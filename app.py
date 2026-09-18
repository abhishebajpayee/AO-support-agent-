"""
FastAPI wrapper around the agent so it can run as a Render Web Service.

Exposes:
  GET  /            -> a minimal chat UI (for a live demo link)
  POST /api/chat     -> {"query": "..."} -> {"answer": "..."}
  GET  /health        -> simple healthcheck

Run locally:
  uvicorn app:app --reload

Deploy on Render:
  Build Command: pip install -r requirements.txt
  Start Command: uvicorn app:app --host 0.0.0.0 --port $PORT
"""
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from src.agent import run_agent

app = FastAPI(title="Agentic Support Assistant")


class ChatRequest(BaseModel):
    query: str


class ChatResponse(BaseModel):
    answer: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    answer = run_agent(req.query)
    return ChatResponse(answer=answer)


@app.get("/", response_class=HTMLResponse)
def index():
    return """
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Agentic Support Assistant</title>
  <style>
    body { font-family: system-ui, sans-serif; max-width: 640px; margin: 40px auto; padding: 0 16px; }
    h1 { font-size: 1.3rem; }
    #log { border: 1px solid #ddd; border-radius: 8px; padding: 12px; height: 400px; overflow-y: auto; margin-bottom: 12px; }
    .msg { margin: 8px 0; }
    .user { font-weight: 600; }
    .agent { color: #333; }
    form { display: flex; gap: 8px; }
    input { flex: 1; padding: 8px; border: 1px solid #ccc; border-radius: 6px; }
    button { padding: 8px 16px; border: none; border-radius: 6px; background: #111; color: #fff; cursor: pointer; }
    button:disabled { opacity: 0.5; }
  </style>
</head>
<body>
  <h1>Agentic Support Assistant</h1>
  <p>Try: "How long does standard shipping take?" or "What's the status of order ORD1002?"</p>
  <div id="log"></div>
  <form id="form">
    <input id="input" placeholder="Ask about shipping, returns, or an order..." autocomplete="off" />
    <button id="btn" type="submit">Send</button>
  </form>
  <script>
    const log = document.getElementById('log');
    const form = document.getElementById('form');
    const input = document.getElementById('input');
    const btn = document.getElementById('btn');

    function addMsg(who, text) {
      const div = document.createElement('div');
      div.className = 'msg';
      div.innerHTML = `<span class="${who === 'You' ? 'user' : 'agent'}">${who}:</span> ${text}`;
      log.appendChild(div);
      log.scrollTop = log.scrollHeight;
    }

    form.addEventListener('submit', async (e) => {
      e.preventDefault();
      const query = input.value.trim();
      if (!query) return;
      addMsg('You', query);
      input.value = '';
      btn.disabled = true;
      addMsg('Agent', 'Thinking...');
      try {
        const res = await fetch('/api/chat', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ query })
        });
        const data = await res.json();
        log.lastChild.innerHTML = `<span class="agent">Agent:</span> ${data.answer}`;
      } catch (err) {
        log.lastChild.innerHTML = `<span class="agent">Agent:</span> Error: ${err}`;
      } finally {
        btn.disabled = false;
      }
    });
  </script>
</body>
</html>
"""
