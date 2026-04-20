import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'agents'))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

from graph import run_pipeline
from agent_memory import get_history, clear_session

load_dotenv(override=True)

app = FastAPI(
    title="Puls-Events API",
    description="API du chatbot RAG multi-agents pour la découverte d'événements culturels",
    version="1.0.0"
)

# CORS — permet à Chainlit de communiquer avec l'API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Modèles Pydantic ──────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    query: str
    session_id: str = "default"
    city: str | None = None
    radius_km: float = 50.0


class ChatResponse(BaseModel):
    response: str
    session_id: str


class HistoryResponse(BaseModel):
    session_id: str
    messages: list[dict]


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health_check():
    """Vérifie que l'API est opérationnelle."""
    return {"status": "ok", "service": "puls-events-api"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    """
    Endpoint principal — envoie une question au pipeline LangGraph.
    """
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="La question ne peut pas être vide")

    try:
        response = run_pipeline(
            query=request.query,
            session_id=request.session_id,
            city=request.city,
            radius_km=request.radius_km
        )
        return ChatResponse(response=response, session_id=request.session_id)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/history/{session_id}", response_model=HistoryResponse)
def get_conversation_history(session_id: str):
    """
    Récupère l'historique d'une session.
    """
    history = get_history(session_id)
    return HistoryResponse(session_id=session_id, messages=history)


@app.delete("/history/{session_id}")
def delete_conversation_history(session_id: str):
    """
    Supprime l'historique d'une session.
    """
    clear_session(session_id)
    return {"message": f"Session {session_id} supprimée"}


@app.get("/")
def root():
    return {
        "service": "Puls-Events API",
        "version": "1.0.0",
        "docs": "/docs"
    }