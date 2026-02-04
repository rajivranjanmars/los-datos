# LPU Student Helpdesk Ticketing System

A chatbot-powered student helpdesk system for Lovely Professional University built with React + FastAPI and powered by local LLMs via Ollama.


## Prerequisites

### 1. Ollama (Required)

This project uses Ollama for local LLM inference. Install Ollama from [ollama.com](https://ollama.com).

After installation, pull the required models:

```bash
ollama pull qwen3:0.6b
ollama pull qwen3-embedding:0.6b
```

**Configuration:**
```
OLLAMA_URL = "http://localhost:11434"
LLM_MODEL = "qwen3:0.6b"
EMBED_MODEL = "qwen3-embedding:0.6b"
```

Make sure Ollama is running before starting the backend server.

### 2. Python 3.10+

### 3. Node.js 18+

## Installation

### Backend Setup

```bash
cd got-quiz

# Create virtual environment (optional but recommended)
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### Frontend Setup

```bash
cd got-quiz

# Install dependencies
npm install
```
