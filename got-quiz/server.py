"""
LPU Student Helpdesk Ticketing System
FastAPI backend with dual LLM architecture:
- LLM Call 1: Routes user message to appropriate handler
- LLM Call 2: Executes the action (RAG answer, DB query, ticket creation)
"""

import os
import json
import pickle
import hashlib
from datetime import datetime
from typing import Optional
import fitz  # PyMuPDF
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ============== Configuration ==============
OLLAMA_URL = "http://localhost:11434"
LLM_MODEL = "qwen3:0.6b"
EMBED_MODEL = "qwen3-embedding:0.6b"
PDF_PATH = os.path.join(os.path.dirname(__file__), "..", "download.pdf")
FAKE_DB_PATH = os.path.join(os.path.dirname(__file__), "fake.json")
CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
TOP_K = 3

# ============== FastAPI App ==============
app = FastAPI(
    title="LPU Student Helpdesk API",
    description="Backend for LPU Student Helpdesk Ticketing System",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============== Pydantic Models ==============
class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str
    route: str
    details: Optional[dict] = None

class LoginRequest(BaseModel):
    enrollment_no: str
    otp: str

# ============== Global State ==============
chunks = []
embeddings = []
db = None
pending_auth = {}  # Tracks pending authentication: {session_id: {"enrollment_no": "...", "student": {...}}}
pending_pdf_search = {}  # Track if we're waiting for user to confirm PDF search
pending_action = {}  # Track what action user wanted before login was required

# ============== Helper Functions ==============

def load_database():
    """Load the fake database"""
    global db
    with open(FAKE_DB_PATH, "r", encoding="utf-8") as f:
        db = json.load(f)
    print(f"✓ Loaded database with {len(db['students'])} students and {len(db['tickets'])} tickets")

def save_database():
    """Save database back to file"""
    with open(FAKE_DB_PATH, "w", encoding="utf-8") as f:
        json.dump(db, f, indent=2)

def get_current_student():
    """Get the currently logged in student"""
    student_id = db["current_student_id"]
    for student in db["students"]:
        if student["id"] == student_id:
            return student
    return None

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from PDF"""
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    return text

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list:
    """Split text into overlapping chunks"""
    words = text.split()
    chunks_list = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk.strip():
            chunks_list.append(chunk)
    return chunks_list

def get_embedding(text: str) -> list:
    """Get embedding from Ollama"""
    response = requests.post(
        f"{OLLAMA_URL}/api/embed",
        json={"model": EMBED_MODEL, "input": text}
    )
    return response.json()["embeddings"][0]

def cosine_similarity(a: list, b: list) -> float:
    """Compute cosine similarity between two vectors"""
    dot_product = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    return dot_product / (norm_a * norm_b) if norm_a and norm_b else 0

def get_top_chunks(query: str, k: int = TOP_K) -> list:
    """Get top k most relevant chunks for a query"""
    query_embedding = get_embedding(query)
    similarities = []
    for i, emb in enumerate(embeddings):
        sim = cosine_similarity(query_embedding, emb)
        similarities.append((sim, chunks[i]))
    similarities.sort(reverse=True, key=lambda x: x[0])
    return [chunk for _, chunk in similarities[:k]]

def call_llm(prompt: str, system: str = "") -> str:
    """Call Ollama LLM"""
    full_prompt = prompt
    if system:
        full_prompt = f"{system}\n\n{prompt}"
    
    response = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={"model": LLM_MODEL, "prompt": full_prompt, "stream": False}
    )
    return response.json()["response"]

def load_or_create_embeddings():
    """Load embeddings from cache or create new ones"""
    global chunks, embeddings
    
    os.makedirs(CACHE_DIR, exist_ok=True)
    
    # Create hash of PDF for cache validation
    with open(PDF_PATH, "rb") as f:
        pdf_hash = hashlib.md5(f.read()).hexdigest()
    
    chunks_cache = os.path.join(CACHE_DIR, f"chunks_{pdf_hash}.pkl")
    embeddings_cache = os.path.join(CACHE_DIR, f"embeddings_{pdf_hash}.pkl")
    
    if os.path.exists(chunks_cache) and os.path.exists(embeddings_cache):
        print("Loading from cache...")
        with open(chunks_cache, "rb") as f:
            chunks = pickle.load(f)
        with open(embeddings_cache, "rb") as f:
            embeddings = pickle.load(f)
        print(f"✓ Loaded {len(chunks)} chunks from cache")
    else:
        print("Creating embeddings (this may take a while)...")
        text = extract_text_from_pdf(PDF_PATH)
        chunks = chunk_text(text)
        print(f"Created {len(chunks)} chunks")
        
        embeddings = []
        for i, chunk in enumerate(chunks):
            emb = get_embedding(chunk)
            embeddings.append(emb)
            if (i + 1) % 10 == 0:
                print(f"  Embedded {i + 1}/{len(chunks)}")
        
        with open(chunks_cache, "wb") as f:
            pickle.dump(chunks, f)
        with open(embeddings_cache, "wb") as f:
            pickle.dump(embeddings, f)
        print("✓ Embeddings cached")

# ============== LLM Router (Call 1) ==============

ROUTE_SYSTEM_PROMPT = """You are a routing assistant for an LPU Student Helpdesk system.
Analyze the user's message intent and classify it into ONE of these categories:

STUDENT DATA ROUTES (for personal/account queries):
- check_fees: Anything about fees, payment, dues, money, scholarship, financial status (e.g., "show my fee", "how much do I owe", "fee status", "payment due", "my fees")
- check_attendance: Anything about attendance, classes attended, presence (e.g., "my attendance", "how many classes", "attendance percentage")
- placement_info: Anything about placement, job, company, package, career (e.g., "placement status", "am I placed", "which company")
- profile: Anything about personal info, details, enrollment, course, branch, semester, CGPA (e.g., "my profile", "my details", "show my info", "what course am I in")
- create_ticket: User has a problem, complaint, issue, something not working, needs help (e.g., "I have a problem", "can't access", "not working", "help me with")
- list_tickets: User wants to see tickets, complaints, previous issues (e.g., "my tickets", "show tickets", "ticket status")

GENERAL ROUTES:
- greeting: Hello, hi, hey, good morning, how are you, thanks, thank you
- logout: Logout, sign out, bye, exit, end session
- provide_enrollment: Just an 8-digit number
- provide_otp: Just a 6-digit number
- confirm_search: Short affirmative like yes, ok, sure, go ahead
- decline_search: Short negative like no, cancel, nevermind

INFORMATION ROUTE (only for general LPU info, NOT personal data):
- ask_pdf: Questions about LPU university in general - admissions process, available courses, campus facilities, hostel info, general fee structure (NOT "my fees")

IMPORTANT:
- If user says ANYTHING about "my" or "mine" + fees/attendance/profile/placement → use the corresponding personal route
- "show my fee" = check_fees (NOT unknown, NOT ask_pdf)
- "my attendance" = check_attendance
- Be flexible with phrasing - understand the INTENT

Respond with ONLY the category name. No explanation.
/no_think"""

# Routes that require authentication
PROTECTED_ROUTES = ["check_fees", "check_attendance", "placement_info", "profile", "create_ticket", "list_tickets"]

def route_message(message: str) -> str:
    """Route user message to appropriate handler"""
    import re
    
    clean_msg = message.strip().lower()
    
    # Quick check for pure numbers - skip LLM call
    if re.match(r'^\d{8}$', clean_msg):
        return "provide_enrollment"
    if re.match(r'^\d{6}$', clean_msg):
        return "provide_otp"
    
    # Keyword-based routing for common patterns (faster than LLM)
    fee_keywords = ["fee", "fees", "payment", "due", "dues", "pay", "scholarship", "money", "financial"]
    attendance_keywords = ["attendance", "attended", "classes", "present", "absent"]
    profile_keywords = ["profile", "my info", "my details", "my data", "enrollment", "cgpa", "gpa", "semester"]
    placement_keywords = ["placement", "placed", "job", "company", "package", "career", "recruit"]
    ticket_keywords = ["ticket", "tickets", "complaint", "complaints", "issue status"]
    problem_keywords = ["problem", "issue", "not working", "can't access", "cannot access", "help me", "error", "wrong", "incorrect"]
    logout_keywords = ["logout", "log out", "sign out", "signout", "end session", "bye", "exit"]
    greeting_keywords = ["hello", "hi", "hey", "good morning", "good afternoon", "good evening", "thanks", "thank you"]
    
    # Check for personal data requests (contains "my" or similar + keyword)
    personal_indicators = ["my", "mine", "show me", "check my", "what is my", "what's my", "tell me my"]
    is_personal = any(ind in clean_msg for ind in personal_indicators)
    
    if any(kw in clean_msg for kw in fee_keywords):
        return "check_fees"
    if any(kw in clean_msg for kw in attendance_keywords):
        return "check_attendance"
    if any(kw in clean_msg for kw in placement_keywords):
        return "placement_info"
    if any(kw in clean_msg for kw in profile_keywords) and is_personal:
        return "profile"
    if "show" in clean_msg and "ticket" in clean_msg:
        return "list_tickets"
    if any(kw in clean_msg for kw in ticket_keywords) and ("my" in clean_msg or "show" in clean_msg or "list" in clean_msg):
        return "list_tickets"
    if any(kw in clean_msg for kw in problem_keywords):
        return "create_ticket"
    if any(kw in clean_msg for kw in logout_keywords):
        return "logout"
    if any(kw in clean_msg for kw in greeting_keywords) and len(clean_msg) < 30:
        return "greeting"
    
    # Confirmation/decline for short messages
    confirm_words = ["yes", "yeah", "yep", "sure", "ok", "okay", "go ahead"]
    decline_words = ["no", "nope", "nevermind", "cancel", "don't", "stop"]
    
    if any(word == clean_msg or clean_msg.startswith(word + " ") for word in confirm_words) and len(clean_msg) < 20:
        return "confirm_search"
    if any(word == clean_msg or clean_msg.startswith(word + " ") for word in decline_words) and len(clean_msg) < 20:
        return "decline_search"
    
    # Fall back to LLM for complex queries
    route = call_llm(message, ROUTE_SYSTEM_PROMPT).strip().lower()
    
    # Clean up response - extract just the route keyword
    valid_routes = ["check_fees", "check_attendance", "placement_info", 
                    "profile", "create_ticket", "list_tickets", "greeting", 
                    "provide_enrollment", "provide_otp", "ask_pdf",
                    "confirm_search", "decline_search", "logout"]
    
    for valid in valid_routes:
        if valid in route:
            return valid
    
    # Default to ask_pdf for general questions (not unknown)
    return "ask_pdf"

def handle_logout(message: str) -> tuple:
    """Handle logout request via chat"""
    global pending_auth, pending_action, pending_pdf_search
    
    student = get_current_student()
    if not student:
        return "You are not currently logged in. No action needed!", {}
    
    student_name = student["name"].split()[0]
    
    # Clear all session data
    db["current_student_id"] = None
    save_database()
    
    # Clear any pending states
    pending_auth.clear()
    pending_action.clear()
    pending_pdf_search.clear()
    
    return f"✅ Goodbye, **{student_name}**! You have been logged out successfully.\n\nYou can still ask general questions about LPU. When you need to access your personal data again, just ask and I'll help you log in.", {"logged_out": True}

def handle_confirm_search(message: str) -> tuple:
    """User confirmed they want to search the website"""
    global pending_pdf_search
    
    if "query" not in pending_pdf_search:
        return "What would you like me to search for on the LPU website?", {}
    
    original_query = pending_pdf_search["query"]
    del pending_pdf_search["query"]
    
    # Now do the PDF search
    return handle_ask_pdf(original_query)

def handle_decline_search(message: str) -> tuple:
    """User declined the search"""
    global pending_pdf_search
    if "query" in pending_pdf_search:
        del pending_pdf_search["query"]
    
    return "No problem! Is there anything else I can help you with? You can ask about your fees, attendance, profile, or create a support ticket.", {}

def find_student_by_enrollment(enrollment_no: str):
    """Find a student by enrollment number"""
    for student in db["students"]:
        if student["enrollment_no"] == enrollment_no:
            return student
    return None

def handle_provide_enrollment(message: str) -> tuple:
    """Handle when user provides enrollment number"""
    import re
    global pending_auth
    
    # Extract 8-digit enrollment number
    match = re.search(r'\d{8}', message)
    if not match:
        return "Please provide your **8-digit enrollment number** to continue.", {"needs_enrollment": True}
    
    enrollment_no = match.group()
    student = find_student_by_enrollment(enrollment_no)
    
    if not student:
        return f"❌ Enrollment number **{enrollment_no}** was not found in our database.\n\nPlease verify your enrollment number or contact the registrar office if you believe this is an error.", {"not_found": True}
    
    # Store pending authentication
    pending_auth["current"] = {
        "enrollment_no": enrollment_no,
        "student": student
    }
    
    return f"✅ Found your record, **{student['name']}**!\n\n🔐 Please enter the **6-digit OTP** sent to your registered email ({student['email']}).", {"awaiting_otp": True, "student_name": student["name"]}

def handle_provide_otp(message: str) -> tuple:
    """Handle when user provides OTP"""
    import re
    global pending_auth, pending_action
    
    # Check if we have a pending enrollment
    if "current" not in pending_auth:
        return "Please provide your **enrollment number** first before entering the OTP.", {"needs_enrollment": True}
    
    # Extract 6-digit OTP
    match = re.search(r'\d{6}', message)
    if not match:
        return "Please enter your **6-digit OTP**.", {"needs_otp": True}
    
    otp = match.group()
    pending = pending_auth["current"]
    student = pending["student"]
    
    if student.get("otp") != otp:
        return "❌ Invalid OTP. Please check and try again.", {"invalid_otp": True}
    
    # Successful login!
    db["current_student_id"] = student["id"]
    save_database()
    del pending_auth["current"]
    
    # Check if there's a pending action to execute
    if "route" in pending_action and "message" in pending_action:
        route = pending_action["route"]
        original_message = pending_action["message"]
        del pending_action["route"]
        del pending_action["message"]
        
        # Execute the pending action
        handlers = {
            "check_fees": handle_check_fees,
            "check_attendance": handle_check_attendance,
            "placement_info": handle_placement_info,
            "profile": handle_profile,
            "create_ticket": handle_create_ticket,
            "list_tickets": handle_list_tickets,
        }
        
        handler = handlers.get(route)
        if handler:
            action_response, action_details = handler(original_message)
            return f"✅ Login successful! Welcome, **{student['name']}**!\n\n---\n\n{action_response}", {"logged_in": True, "student_name": student["name"], **action_details}
    
    return f"✅ Login successful! Welcome, **{student['name']}**!\n\nHow can I help you today? You can now:\n ", {"logged_in": True, "student_name": student["name"]}

def require_login_response(route: str, original_message: str) -> tuple:
    """Response when login is required"""
    global pending_action
    
    action_names = {
        "check_fees": "check your fee status",
        "check_attendance": "check your attendance",
        "placement_info": "view placement information",
        "profile": "view your profile",
        "create_ticket": "create a support ticket",
        "list_tickets": "view your tickets"
    }
    action = action_names.get(route, "access this feature")
    
    # Store the pending action so we can execute it after login
    pending_action["route"] = route
    pending_action["message"] = original_message
    
    return f"""🔐 **Authentication Required**

To {action}, I need to verify your identity.

Please enter your **8-digit enrollment number**.""", {"requires_login": True, "requested_route": route}

# ============== Route Handlers ==============

def handle_greeting(message: str) -> tuple:
    """Handle greeting messages"""
    student = get_current_student()
    if student:
        name = student["name"].split()[0]
        response = f"Hello {name}! 👋 I'm your LPU Student Helpdesk assistant. How can I help you today?\n\nI can help you with:\n• University information (admissions, courses, fees)\n• Check your fee status\n• Check your attendance\n• View your profile\n• Check placement status\n• Create support tickets\n• View existing tickets"
    else:
        response = "Hello! 👋 I'm your LPU Student Helpdesk assistant. How can I help you today?\n\nI can help you with:\n• University information (admissions, courses, fees structure)\n• General queries about LPU\n\n🔐 For personalized services (fees, attendance, profile, tickets), you'll need to login with your enrollment number and OTP."
    return response, {}

def handle_ask_pdf(message: str) -> tuple:
    """Answer questions from PDF using RAG"""
    relevant_chunks = get_top_chunks(message)
    context = "\n\n".join(relevant_chunks)
    
    system = """You are an LPU Student Helpdesk assistant. Answer the student's question based on the provided context from the LPU prospectus.
Be helpful, concise, and accurate. If the information is not in the context, say you don't have that specific information.
Format your response nicely with bullet points where appropriate. /no_think"""
    
    prompt = f"""Context from LPU Prospectus:
{context}

Student's Question: {message}

Answer:"""
    
    response = call_llm(prompt, system)
    return response, {"source": "prospectus", "chunks_used": len(relevant_chunks)}

def handle_check_fees(message: str) -> tuple:
    """Check student's fee status"""
    student = get_current_student()
    if not student:
        return "Error: Could not find your student record.", {}
    
    response = f"""💰 **Fee Status for {student['name']}**

• **Total Fees:** ₹{student['total_fees']:,}
• **Fees Paid:** ₹{student['fees_paid']:,}
• **Fees Due:** ₹{student['fees_due']:,}
• **Last Payment:** {student['last_payment_date']}"""
    
    if student['scholarship_amount'] > 0:
        response += f"""

🎓 **Scholarship Details:**
• **Type:** {student['scholarship_type']}
• **Amount:** ₹{student['scholarship_amount']:,}"""
    
    if student['fees_due'] > 0:
        response += f"\n\n⚠️ You have ₹{student['fees_due']:,} pending. Please clear your dues to avoid late fees."
    else:
        response += "\n\n✅ All fees are cleared. Great job!"
    
    return response, {"fees_due": student['fees_due'], "total": student['total_fees']}

def handle_check_attendance(message: str) -> tuple:
    """Check student's attendance"""
    student = get_current_student()
    if not student:
        return "Error: Could not find your student record.", {}
    
    attendance = student['attendance_percent']
    status = "✅ Good" if attendance >= 75 else "⚠️ Low (min 75% required)"
    
    response = f"""📊 **Attendance Status for {student['name']}**

• **Current Attendance:** {attendance}%
• **Status:** {status}
• **Course:** {student['course']} - {student['branch']}
• **Semester:** {student['semester']}"""
    
    if attendance < 75:
        shortage = 75 - attendance
        response += f"\n\n⚠️ Your attendance is {shortage:.1f}% below the required minimum. Please attend classes regularly to avoid detention."
    
    return response, {"attendance": attendance, "status": "good" if attendance >= 75 else "low"}

def handle_placement_info(message: str) -> tuple:
    """Check student's placement status"""
    student = get_current_student()
    if not student:
        return "Error: Could not find your student record.", {}
    
    status = student['placement_status']
    
    response = f"""💼 **Placement Status for {student['name']}**

• **Status:** {status}"""
    
    if status == "Placed":
        response += f"""
• **Company:** {student['company_placed']}
• **Package:** ₹{student['package_lpa']} LPA
• **Placement Year:** {student['placement_year']}

🎉 Congratulations on your placement!"""
    elif status == "Interview Scheduled":
        response += "\n\n📅 You have upcoming interviews. Check your email for details and prepare well!"
    elif status == "Not Eligible":
        response += "\n\n❌ You are currently not eligible for placements. Please clear your backlogs and meet attendance requirements."
    else:
        response += "\n\n📋 Keep checking the placement portal for new opportunities. Update your resume and prepare for upcoming drives!"
    
    return response, {"placement_status": status, "company": student.get('company_placed')}

def handle_profile(message: str) -> tuple:
    """Get student profile information"""
    student = get_current_student()
    if not student:
        return "Error: Could not find your student record.", {}
    
    response = f"""👤 **Student Profile**

**Personal Details:**
• **Name:** {student['name']}
• **Email:** {student['email']}
• **Phone:** {student['phone']}
• **DOB:** {student['dob']}

**Academic Details:**
• **Enrollment No:** {student['enrollment_no']}
• **Course:** {student['course']} - {student['branch']}
• **Semester:** {student['semester']}
• **Batch:** {student['batch']}
• **CGPA:** {student['cgpa']}
• **Academic Status:** {student['academic_status']}

**Address:**
• {student['address_line1']}
{f"• {student['address_line2']}" if student['address_line2'] else ""}
• {student['city']}, {student['state']} - {student['pincode']}"""
    
    return response, {"enrollment": student['enrollment_no'], "cgpa": student['cgpa']}

def handle_create_ticket(message: str) -> tuple:
    """Create a support ticket"""
    student = get_current_student()
    if not student:
        return "Error: Could not find your student record.", {}
    
    # Use LLM to summarize and categorize the issue
    categorize_prompt = f"""Analyze this student complaint/issue and provide:
1. A brief summary (max 15 words)
2. Category (one of: academic, financial, technical, admission, placement, general)
3. Priority (one of: low, medium, high, urgent)

Student's message: {message}

Respond in this exact format:
SUMMARY: <summary>
CATEGORY: <category>
PRIORITY: <priority>

/no_think"""
    
    analysis = call_llm(categorize_prompt)
    
    # Parse the response
    summary = "Student issue reported"
    category = "general"
    priority = "medium"
    
    for line in analysis.split("\n"):
        line = line.strip()
        if line.startswith("SUMMARY:"):
            summary = line.replace("SUMMARY:", "").strip()
        elif line.startswith("CATEGORY:"):
            cat = line.replace("CATEGORY:", "").strip().lower()
            if cat in db["metadata"]["categories"]:
                category = cat
        elif line.startswith("PRIORITY:"):
            pri = line.replace("PRIORITY:", "").strip().lower()
            if pri in db["metadata"]["priorities"]:
                priority = pri
    
    # Create ticket
    db["metadata"]["last_ticket_number"] += 1
    ticket_id = f"TKT-{db['metadata']['last_ticket_number']:03d}"
    now = datetime.utcnow().isoformat() + "Z"
    
    ticket = {
        "id": ticket_id,
        "student_id": student["id"],
        "created_at": now,
        "updated_at": now,
        "status": "open",
        "question": message,
        "summary": summary,
        "category": category,
        "priority": priority,
        "resolution": None
    }
    
    db["tickets"].append(ticket)
    save_database()
    
    response = f"""🎫 **Support Ticket Created**

• **Ticket ID:** {ticket_id}
• **Status:** Open
• **Category:** {category.title()}
• **Priority:** {priority.title()}

**Your Issue:**
{message}

**Summary:** {summary}

Our support team will review your ticket and respond shortly. You can check the status by asking "show my tickets"."""
    
    return response, {"ticket_id": ticket_id, "category": category, "priority": priority}

def handle_list_tickets(message: str) -> tuple:
    """List student's tickets"""
    student = get_current_student()
    if not student:
        return "Error: Could not find your student record.", {}
    
    student_tickets = [t for t in db["tickets"] if t["student_id"] == student["id"]]
    
    if not student_tickets:
        return "📋 You don't have any support tickets yet. If you have an issue, just describe it and I'll create a ticket for you!", {}
    
    response = f"📋 **Your Support Tickets** ({len(student_tickets)} total)\n\n"
    
    status_emoji = {"open": "🟡", "in_progress": "🔵", "resolved": "🟢", "closed": "⚫"}
    
    for ticket in sorted(student_tickets, key=lambda x: x["created_at"], reverse=True):
        emoji = status_emoji.get(ticket["status"], "⚪")
        response += f"""{emoji} **{ticket['id']}** - {ticket['status'].replace('_', ' ').title()}
   *{ticket['summary']}*
   Created: {ticket['created_at'][:10]}
"""
        if ticket["resolution"]:
            response += f"   ✅ Resolution: {ticket['resolution']}\n"
        response += "\n"
    
    return response, {"ticket_count": len(student_tickets)}

# ============== Main Chat Endpoint ==============

@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Main chat endpoint - routes message and executes appropriate action
    """
    message = request.message.strip()
    
    if not message:
        raise HTTPException(status_code=400, detail="Message cannot be empty")
    
    # Check if we're waiting for OTP (user already provided enrollment)
    if "current" in pending_auth:
        # User might be providing OTP
        import re
        if re.search(r'\d{6}', message):
            response_text, details = handle_provide_otp(message)
            return ChatResponse(response=response_text, route="provide_otp", details=details)
    
    # LLM Call 1: Route the message
    route = route_message(message)
    print(f"Route: {route}")
    
    # Handle enrollment number
    if route == "provide_enrollment":
        response_text, details = handle_provide_enrollment(message)
        return ChatResponse(response=response_text, route=route, details=details)
    
    # Handle OTP
    if route == "provide_otp":
        response_text, details = handle_provide_otp(message)
        return ChatResponse(response=response_text, route=route, details=details)
    
    # Check if route requires authentication
    if route in PROTECTED_ROUTES and not get_current_student():
        response_text, details = require_login_response(route, message)
        return ChatResponse(response=response_text, route="require_login", details=details)
    
    # LLM Call 2: Execute the appropriate handler
    handlers = {
        "greeting": handle_greeting,
        "ask_pdf": handle_ask_pdf,
        "check_fees": handle_check_fees,
        "check_attendance": handle_check_attendance,
        "placement_info": handle_placement_info,
        "profile": handle_profile,
        "create_ticket": handle_create_ticket,
        "list_tickets": handle_list_tickets,
        "confirm_search": handle_confirm_search,
        "decline_search": handle_decline_search,
        "logout": handle_logout,
    }
    
    handler = handlers.get(route, handle_ask_pdf)
    response_text, details = handler(message)
    
    return ChatResponse(response=response_text, route=route, details=details)

# ============== Additional API Endpoints ==============

@app.get("/api/student")
async def get_student():
    """Get current student info"""
    student = get_current_student()
    if not student:
        raise HTTPException(status_code=404, detail="Student not found")
    return student

@app.get("/api/tickets")
async def get_tickets():
    """Get current student's tickets"""
    student = get_current_student()
    if not student:
        raise HTTPException(status_code=404, detail="Student not found")
    return [t for t in db["tickets"] if t["student_id"] == student["id"]]

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "chunks_loaded": len(chunks),
        "students_loaded": len(db["students"]) if db else 0,
        "current_student": db["current_student_id"] if db else None
    }

@app.post("/api/login")
async def login(request: LoginRequest):
    """Login with enrollment number and OTP"""
    enrollment_no = request.enrollment_no.strip()
    otp = request.otp.strip()
    
    # Find student by enrollment number
    for student in db["students"]:
        if student["enrollment_no"] == enrollment_no:
            # Check OTP
            if student.get("otp") == otp:
                db["current_student_id"] = student["id"]
                save_database()
                return {
                    "success": True,
                    "message": f"Welcome back, {student['name']}!",
                    "student_name": student["name"],
                    "student_id": student["id"]
                }
            else:
                raise HTTPException(
                    status_code=401,
                    detail="Invalid OTP. Please check your credentials and try again."
                )
    
    # Student not found
    raise HTTPException(
        status_code=404,
        detail="Enrollment number not found in our database. Please verify your enrollment number or contact the registrar office."
    )

@app.post("/api/logout")
async def logout():
    """Logout current student"""
    db["current_student_id"] = None
    save_database()
    return {"success": True, "message": "Logged out successfully"}

@app.get("/api/session")
async def get_session():
    """Check if user is logged in"""
    student = get_current_student()
    if student:
        return {
            "logged_in": True,
            "student_name": student["name"],
            "student_id": student["id"],
            "enrollment_no": student["enrollment_no"]
        }
    return {"logged_in": False}

# ============== Startup ==============

@app.on_event("startup")
async def startup():
    """Initialize on startup"""
    print("=" * 50)
    print("LPU Student Helpdesk System Starting...")
    print("=" * 50)
    
    # Load database
    load_database()
    
    # Load/create embeddings
    load_or_create_embeddings()
    
    current = get_current_student()
    print("=" * 50)
    print("✓ System Ready!")
    if current:
        print(f"Current student: {current['name']}")
    else:
        print("No student logged in - authentication required for personal data")
    print("=" * 50)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
