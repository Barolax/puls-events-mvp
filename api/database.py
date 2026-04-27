import os
from datetime import datetime
from sqlalchemy import create_engine, Column, String, Integer, Boolean, DateTime
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv

load_dotenv(override=True)

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./puls_events.db")

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class User(Base):
    """
    Modèle utilisateur.
    Le mot de passe n'est JAMAIS stocké en clair — uniquement le hash bcrypt.
    """
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class OTPCode(Base):
    """
    Codes OTP pour l'authentification à deux facteurs.
    Le code n'est JAMAIS stocké en clair — uniquement le hash bcrypt.
    """
    __tablename__ = "otp_codes"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, index=True, nullable=False)
    hashed_code = Column(String, nullable=False)  # bcrypt hash du code OTP
    expires_at = Column(DateTime, nullable=False)  # Expiration 5 min
    is_used = Column(Boolean, default=False)        # Usage unique
    created_at = Column(DateTime, default=datetime.utcnow)


class LoginAttempt(Base):
    """
    Suivi des tentatives de connexion pour le rate limiting.
    """
    __tablename__ = "login_attempts"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, index=True, nullable=False)
    success = Column(Boolean, default=False)
    attempted_at = Column(DateTime, default=datetime.utcnow)


def get_db():
    """
    Générateur de session DB pour FastAPI Depends.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """
    Crée toutes les tables si elles n'existent pas.
    """
    Base.metadata.create_all(bind=engine)
    print("Base de données initialisée ✓")


if __name__ == "__main__":
    init_db()
    print("Tables créées : users, otp_codes, login_attempts")