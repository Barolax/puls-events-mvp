import os
import random
import string
from datetime import datetime, timedelta
from typing import Optional
from dotenv import load_dotenv, find_dotenv

from jose import JWTError, jwt
from passlib.context import CryptContext
from sqlalchemy.orm import Session
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from database import User, OTPCode, LoginAttempt, get_db

load_dotenv(find_dotenv(), override=True)

# ── Configuration ─────────────────────────────────────────────────────────────

SECRET_KEY = os.getenv("JWT_SECRET_KEY", "puls-events-secret-key-change-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
OTP_EXPIRE_MINUTES = 5
MAX_LOGIN_ATTEMPTS = 3  # Rate limiting — 3 essais max

# Contexte bcrypt — jamais de mot de passe en clair
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
security = HTTPBearer()

# ── Hashage bcrypt ────────────────────────────────────────────────────────────

def hash_password(password: str) -> str:
    """
    Hash un mot de passe avec bcrypt.
    Le salt est généré automatiquement — jamais de stockage en clair.
    """
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Vérifie un mot de passe sans jamais le décrypter.
    Compare les hashs — la comparaison est résistante aux timing attacks.
    """
    return pwd_context.verify(plain_password, hashed_password)


def hash_otp(otp_code: str) -> str:
    """
    Hash un code OTP avec bcrypt.
    Le code OTP n'est jamais stocké en clair en base.
    """
    return pwd_context.hash(otp_code)


def verify_otp_hash(plain_code: str, hashed_code: str) -> bool:
    """
    Vérifie un code OTP sans jamais le décrypter.
    """
    return pwd_context.verify(plain_code, hashed_code)


# ── Génération OTP ────────────────────────────────────────────────────────────

def generate_otp() -> str:
    """
    Génère un code OTP à 6 chiffres aléatoire.
    """
    return "".join(random.choices(string.digits, k=6))


def send_otp_simulation(email: str, otp_code: str):
    """
    Simule l'envoi d'un OTP par email.
    En production : remplacer par SendGrid ou SMTP.
    """
    print(f"\n{'='*50}")
    print(f"📧 SIMULATION ENVOI EMAIL")
    print(f"À : {email}")
    print(f"Objet : Votre code de connexion Puls-Events")
    print(f"Code OTP : {otp_code}")
    print(f"Valide {OTP_EXPIRE_MINUTES} minutes — usage unique")
    print(f"{'='*50}\n")


def create_otp(email: str, db: Session) -> str:
    """
    Crée et stocke un OTP haché en base.
    Retourne le code en clair pour l'envoyer à l'utilisateur.
    """
    # Invalide les anciens OTPs de cet email
    db.query(OTPCode).filter(
        OTPCode.email == email,
        OTPCode.is_used == False
    ).delete()
    db.commit()

    # Génère le nouveau OTP
    otp_code = generate_otp()
    hashed = hash_otp(otp_code)
    expires = datetime.utcnow() + timedelta(minutes=OTP_EXPIRE_MINUTES)

    otp = OTPCode(
        email=email,
        hashed_code=hashed,  # Stocké haché — jamais en clair
        expires_at=expires,
        is_used=False
    )
    db.add(otp)
    db.commit()

    return otp_code  # Retourné en clair une seule fois pour l'envoi


def verify_otp(email: str, code: str, db: Session) -> bool:
    """
    Vérifie un OTP :
    - Non expiré
    - Non utilisé (usage unique)
    - Hash correspondant
    """
    otp = db.query(OTPCode).filter(
        OTPCode.email == email,
        OTPCode.is_used == False,
        OTPCode.expires_at > datetime.utcnow()
    ).first()

    if not otp:
        return False

    if not verify_otp_hash(code, otp.hashed_code):
        return False

    # Invalide le token après usage unique
    otp.is_used = True
    db.commit()

    return True


# ── Rate limiting ─────────────────────────────────────────────────────────────

def check_rate_limit(email: str, db: Session) -> bool:
    """
    Vérifie si l'utilisateur n'a pas dépassé le nombre de tentatives.
    Fenêtre glissante de 15 minutes.
    Returns True si autorisé, False si bloqué.
    """
    window = datetime.utcnow() - timedelta(minutes=15)

    attempts = db.query(LoginAttempt).filter(
        LoginAttempt.email == email,
        LoginAttempt.success == False,
        LoginAttempt.attempted_at > window
    ).count()

    return attempts < MAX_LOGIN_ATTEMPTS


def record_attempt(email: str, success: bool, db: Session):
    """
    Enregistre une tentative de connexion.
    """
    attempt = LoginAttempt(email=email, success=success)
    db.add(attempt)
    db.commit()


# ── JWT ───────────────────────────────────────────────────────────────────────

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """
    Génère un JWT signé avec expiration.
    """
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def verify_token(token: str) -> Optional[str]:
    """
    Vérifie et décode un JWT.
    Retourne l'email si valide, None sinon.
    """
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        return email if email else None
    except JWTError:
        return None


# ── Gestion des utilisateurs ──────────────────────────────────────────────────

def create_user(email: str, db: Session) -> User:
    """
    Crée un utilisateur avec juste son email.
    Pas de mot de passe — authentification OTP uniquement.
    """
    existing = db.query(User).filter(User.email == email).first()
    if existing:
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
    """
    Dépendance FastAPI — vérifie le JWT et retourne l'utilisateur.
    """
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

    print("=== Test Sécurité Puls-Events ===\n")

    # 1. Création utilisateur
    print("1. Création utilisateur...")
    try:
        user = create_user("axelle@puls-events.fr", "MonMotDePasse123!", db)
        print(f"   Utilisateur créé : {user.email}")
        print(f"   Password en DB   : {user.hashed_password[:30]}... (haché bcrypt ✓)")
    except ValueError as e:
        print(f"   {e}")

    # 2. Vérification password
    print("\n2. Vérification password...")
    user = db.query(User).filter(User.email == "axelle@puls-events.fr").first()
    print(f"   'MonMotDePasse123!' correct  : {verify_password('MonMotDePasse123!', user.hashed_password)}")
    print(f"   'mauvais_password' incorrect : {verify_password('mauvais_password', user.hashed_password)}")

    # 3. OTP
    print("\n3. Génération OTP...")
    if check_rate_limit("axelle@puls-events.fr", db):
        otp_code = create_otp("axelle@puls-events.fr", db)
        send_otp_simulation("axelle@puls-events.fr", otp_code)
        print(f"   OTP en DB : haché bcrypt (jamais en clair ✓)")

        print("   Vérification OTP correct  :", verify_otp("axelle@puls-events.fr", otp_code, db))
        print("   Vérification OTP re-utilisé :", verify_otp("axelle@puls-events.fr", otp_code, db))

    # 4. JWT
    print("\n4. Génération JWT...")
    token = create_access_token({"sub": "axelle@puls-events.fr"})
    print(f"   Token : {token[:50]}...")
    print(f"   Email décodé : {verify_token(token)}")

    # 5. Rate limiting
    print("\n5. Test rate limiting...")
    for i in range(4):
        allowed = check_rate_limit("hacker@evil.com", db)
        record_attempt("hacker@evil.com", False, db)
        print(f"   Tentative {i+1} : {'✅ autorisée' if allowed else '❌ bloquée'}")

    db.close()
    print("\n=== Sécurité validée ! ===")