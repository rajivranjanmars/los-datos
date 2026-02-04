import { useState, useRef, useEffect } from 'react'
import './App.css'

function App() {
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [studentName, setStudentName] = useState('')
  const [sessionChecked, setSessionChecked] = useState(false)
  const messagesEndRef = useRef(null)

  // Check existing session on load
  useEffect(() => {
    checkSession()
  }, [])

  const WELCOME_MESSAGE = `Hello! 👋 I'm your LPU Student Helpdesk assistant. How can I help you today?\n\nI can help you with University information (admissions, courses, fees structure)\n`

  const checkSession = async () => {
    try {
      await fetch('http://localhost:8000/api/logout', { method: 'POST' })
    } catch (error) {
      console.error('Session reset failed:', error)
    }
    setMessages([{ role: 'assistant', content: WELCOME_MESSAGE }])
    setSessionChecked(true)
  }

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  const sendMessage = async () => {
    if (!input.trim() || loading) return

    const userMessage = input.trim()
    setInput('')
    setMessages(prev => [...prev, { role: 'user', content: userMessage }])
    setLoading(true)

    try {
      const response = await fetch('http://localhost:8000/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: userMessage })
      })

      if (!response.ok) throw new Error('Failed to get response')

      const data = await response.json()
      
      // Check if user just logged in via chat
      if (data.details?.logged_in && data.details?.student_name) {
        setStudentName(data.details.student_name)
      }
      
      // Check if user logged out via chat
      if (data.details?.logged_out) {
        setStudentName('')
      }
      
      setMessages(prev => [...prev, { 
        role: 'assistant', 
        content: data.response,
        route: data.route 
      }])
    } catch (error) {
      setMessages(prev => [...prev, { 
        role: 'assistant', 
        content: 'Sorry, I encountered an error. Please try again after 4 hours, as the issue has been logged and will be fixed soon. You will get an email once the issue is resolved.',
        error: true
      }])
    } finally {
      setLoading(false)
    }
  }

  const formatMessage = (content) => {
    return content
      .split('\n')
      .map(line => {
        line = line.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
        line = line.replace(/\*(.*?)\*/g, '<em>$1</em>')
        if (line.startsWith('•')) return `<li>${line.substring(1).trim()}</li>`
        return line
      })
      .join('<br/>')
  }

  if (!sessionChecked) {
    return (
      <div className="app">
        <div className="loading-screen">
          <div className="logo">🎓</div>
          <p>Loading...</p>
        </div>
      </div>
    )
  }

  return (
    <div className="app">
      <header className="header">
        <div className="logo">🎓</div>
        <div className="title">
          <h1>LPU Student Helpdesk</h1>
          <span className="subtitle">Lovely Professional University</span>
        </div>
        {studentName && (
          <div className="user-info">
            <span className="user-name">👤 {studentName}</span>
          </div>
        )}
      </header>

      <main className="chat-container">
        <div className="messages">
          {messages.map((msg, idx) => (
            <div key={idx} className={`message ${msg.role} ${msg.error ? 'error' : ''}`}>
              <div className="avatar">
                {msg.role === 'user' ? '👤' : '🤖'}
              </div>
              <div className="content">
                <div 
                  className="text"
                  dangerouslySetInnerHTML={{ __html: formatMessage(msg.content) }}
                />
                {msg.route && (
                  <span className="route-tag">{msg.route.replace('_', ' ')}</span>
                )}
              </div>
            </div>
          ))}
          {loading && (
            <div className="message assistant">
              <div className="avatar">🤖</div>
              <div className="content">
                <div className="typing">
                  <span></span><span></span><span></span>
                </div>
              </div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>
      </main>

      <footer className="input-area">
        <div className="input-container">
          <textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && !e.shiftKey && (e.preventDefault(), sendMessage())}
            placeholder="Ask about fees, attendance, placements, or report an issue..."
            disabled={loading}
            rows={1}
          />
          <button onClick={sendMessage} disabled={loading || !input.trim()}>
            {loading ? '⏳' : '➤'}
          </button>
        </div>
        <div className="hints">
          Try: "What's my fee status?" • "Show my attendance" • "I can't access the portal"
        </div>
      </footer>
    </div>
  )
}

export default App
