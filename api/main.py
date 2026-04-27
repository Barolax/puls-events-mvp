import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'agents'))
sys.path.append(os.path.dirname(__file__))

from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr
from sqlalchemy.orm import Session
from dotenv import load_dotenv

from graph import run_pipeline
from agent_memory import get_history, clear_session
from database import get_db, init_db, User
from auth import (
    create_user, create_otp, verify_otp, create_access_token,
    get_current_user, check_rate_limit, record_attempt,
    send_otp_simulation, verify_password
)

load_dotenv(override=True)

app = FastAPI(
    title="Puls-Events API",
    description="API du chatbot RAG multi-agents pour la découverte d'événements culturels",
    version="1.0.0"
)

# Initialisation DB au démarrage
init_db()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Modèles Pydantic ──────────────────────────────────────────────────────────

class RegisterRequest(BaseModel):
    email: str

class OTPRequest(BaseModel):
    email: str

class OTPVerifyRequest(BaseModel):
    email: str
    otp_code: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


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


# ── Endpoints publics ─────────────────────────────────────────────────────────

@app.get("/health")
def health_check():
    return {"status": "ok", "service": "puls-events-api"}


@app.get("/")
def root():
    return {
        "service": "Puls-Events API",
        "version": "1.0.0",
        "docs": "/docs"
    }


@app.post("/register")
def register(request: RegisterRequest, db: Session = Depends(get_db)):
    """
    Inscription avec juste l'email.
    """
    try:
        user = create_user(request.email, db)
        return {"message": f"Compte créé pour {user.email}"}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/request-otp")
def request_otp(request: OTPRequest, db: Session = Depends(get_db)):
    """
    Demande un OTP — rate limiting 3 tentatives max.
    """
    if not check_rate_limit(request.email, db):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Trop de tentatives — réessaie dans 15 minutes")

    # Vérifie que l'utilisateur existe
    user = db.query(User).filter(User.email == request.email).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Email non trouvé — inscris-toi d'abord"
        )

    record_attempt(request.email, True, db)

    # Génération et envoi OTP simulé
    otp_code = create_otp(request.email, db)
    send_otp_simulation(request.email, otp_code)

    return {
        "message": f"Code OTP envoyé à {request.email}",
        "expires_in": "5 minutes",
        "note": "OTP stocké haché bcrypt — usage unique"
    }

@app.post("/verify-otp", response_model=TokenResponse)
def verify_otp_endpoint(request: OTPVerifyRequest, db: Session = Depends(get_db)):
    """
    Vérifie le code OTP et retourne un JWT.
    Le token OTP est invalidé après usage unique.
    """
    if not check_rate_limit(request.email, db):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Trop de tentatives — réessaie dans 15 minutes"
        )

    if not verify_otp(request.email, request.otp_code, db):
        record_attempt(request.email, False, db)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Code OTP invalide ou expiré"
        )

    record_attempt(request.email, True, db)

    # Génération JWT
    token = create_access_token({"sub": request.email})
    return TokenResponse(access_token=token)

# ── Endpoints protégés (JWT requis) ──────────────────────────────────────────

@app.post("/chat", response_model=ChatResponse)
def chat(
    request: ChatRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Endpoint principal — protégé par JWT.
    """
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="La question ne peut pas être vide")

    try:
        response = run_pipeline(
            query=request.query,
            session_id=f"{current_user.email}_{request.session_id}",
            city=request.city,
            radius_km=request.radius_km
        )
        return ChatResponse(response=response, session_id=request.session_id)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/history/{session_id}", response_model=HistoryResponse)
def get_conversation_history(
    session_id: str,
    current_user: User = Depends(get_current_user)
):
    """
    Récupère l'historique — protégé par JWT.
    """
    full_session_id = f"{current_user.email}_{session_id}"
    history = get_history(full_session_id)
    return HistoryResponse(session_id=session_id, messages=history)


@app.delete("/history/{session_id}")
def delete_conversation_history(
    session_id: str,
    current_user: User = Depends(get_current_user)
):
    """
    Supprime l'historique — protégé par JWT.
    """
    full_session_id = f"{current_user.email}_{session_id}"
    clear_session(full_session_id)
    return {"message": f"Session {session_id} supprimée"}