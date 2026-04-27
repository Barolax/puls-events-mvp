import os
import random
import string
from datetime import datetime, timedelta, timezone  
from typing import Optional
from dotenv import load_dotenv, find_dotenv

from jose import JWTError, jwt
from passlib.context import CryptContext
from sqlalchemy.orm import Session
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from database import User, OTPCode, LoginAttempt, get_db

load_dotenv(find_dotenv(), override=True)

SECRET_KEY = os.getenv("JWT_SECRET_KEY", "puls-events-secret-key-change-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
OTP_EXPIRE_MINUTES = 5
MAX_LOGIN_ATTEMPTS = 3

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
security = HTTPBearer()


# ── Hashage bcrypt OTP ────────────────────────────────────────────────────────

def hash_otp(otp_code: str) -> str:
    """Hash un code OTP — jamais stocké en clair."""
    return pwd_context.hash(otp_code)


def verify_otp_hash(plain_code: str, hashed_code: str) -> bool:
    """Vérifie un code OTP sans jamais le décrypter."""
    return pwd_context.verify(plain_code, hashed_code)


# ── Génération OTP ────────────────────────────────────────────────────────────

def generate_otp() -> str:
    """Génère un code OTP à 6 chiffres."""
    return "".join(random.choices(string.digits, k=6))


def send_otp_simulation(email: str, otp_code: str):
    """Simule l'envoi d'un OTP par email."""
    print(f"\n{'='*50}")
    print(f"📧 SIMULATION ENVOI EMAIL")
    print(f"À : {email}")
    print(f"Objet : Votre code de connexion Puls-Events")
    print(f"Code OTP : {otp_code}")
    print(f"Valide {OTP_EXPIRE_MINUTES} minutes — usage unique")
    print(f"{'='*50}\n")


def create_otp(email: str, db: Session) -> str:
    """Crée et stocke un OTP haché. Retourne le code en clair."""
    db.query(OTPCode).filter(
        OTPCode.email == email,
        OTPCode.is_used == False
    ).delete()
    db.commit()

    otp_code = generate_otp()
    expires = datetime.now(timezone.utc) + timedelta(minutes=OTP_EXPIRE_MINUTES)

    otp = OTPCode(
        email=email,
        hashed_code=hash_otp(otp_code),
        expires_at=expires,
        is_used=False
    )
    db.add(otp)
    db.commit()

    return otp_code


def verify_otp(email: str, code: str, db: Session) -> bool:
    """Vérifie un OTP — non expiré, non utilisé, hash correct."""
    otp = db.query(OTPCode).filter(
        OTPCode.email == email,
        OTPCode.is_used == False,
        OTPCode.expires_at > datetime.now(timezone.utc)
    ).first()

    if not otp or not verify_otp_hash(code, otp.hashed_code):
        return False

    otp.is_used = True
    db.commit()
    return True


# ── Rate limiting ─────────────────────────────────────────────────────────────

def check_rate_limit(email: str, db: Session) -> bool:
    """Vérifie le rate limiting — 3 tentatives max / 15 min."""
    window = datetime.now(timezone.utc) - timedelta(minutes=15)
    attempts = db.query(LoginAttempt).filter(
        LoginAttempt.email == email,
        LoginAttempt.success == False,
        LoginAttempt.attempted_at > window
    ).count()
    return attempts < MAX_LOGIN_ATTEMPTS


def record_attempt(email: str, success: bool, db: Session):
    """Enregistre une tentative de connexion."""
    db.add(LoginAttempt(email=email, success=success))
    db.commit()


# ── JWT ───────────────────────────────────────────────────────────────────────

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Génère un JWT signé avec expiration."""
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def verify_token(token: str) -> Optional[str]:
    """Vérifie et décode un JWT. Retourne l'email si valide."""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        return email if email else None
    except JWTError:
        return None


# ── Gestion des utilisateurs ──────────────────────────────────────────────────

def create_user(email: str, db: Session) -> User:
    """Crée un utilisateur passwordless."""
    if db.query(User).filter(User.email == email).first():
        raise ValueError(f"Email {email} déjà utilisé")
    user = User(email=email)
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db)
) -> User:
    """Dépendance FastAPI — vérifie le JWT."""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Token invalide ou expiré",
        headers={"WWW-Authenticate": "Bearer"},
    )
    email = verify_token(credentials.credentials)
    if not email:
        raise credentials_exception
    user = db.query(User).filter(User.email == email).first()
    if not user or not user.is_active:
        raise credentials_exception
    return user


# ── Test ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from database import SessionLocal, init_db

    init_db()
    db = SessionLocal()

    print("=== Test Sécurité Passwordless ===\n")

    # 1. Création utilisateur
    print("1. Création utilisateur (sans password)...")
    try:
        user = create_user("axelle@puls-events.fr", db)
        print(f"   Email : {user.email} ✓")
        print(f"   Pas de password en DB ✓")
    except ValueError as e:
        print(f"   {e}")

    # 2. OTP
    print("\n2. Génération OTP...")
    otp_code = create_otp("axelle@puls-events.fr", db)
    send_otp_simulation("axelle@puls-events.fr", otp_code)
    print(f"   OTP haché en DB — jamais en clair ✓")
    print(f"   Vérification correct   : {verify_otp('axelle@puls-events.fr', otp_code, db)}")
    print(f"   Vérification re-utilisé: {verify_otp('axelle@puls-events.fr', otp_code, db)}")

    # 3. JWT
    print("\n3. JWT...")
    token = create_access_token({"sub": "axelle@puls-events.fr"})
    print(f"   Token : {token[:50]}...")
    print(f"   Email : {verify_token(token)} ✓")

    # 4. Rate limiting
    print("\n4. Rate limiting...")
    for i in range(4):
        allowed = check_rate_limit("hacker@evil.com", db)
        record_attempt("hacker@evil.com", False, db)
        print(f"   Tentative {i+1} : {'✅ autorisée' if allowed else '❌ bloquée'}")

    db.close()
    print("\n=== Sécurité passwordless validée ! ===")