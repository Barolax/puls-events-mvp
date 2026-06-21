import os
import httpx
import chainlit as cl
from chainlit.data.sql_alchemy import SQLAlchemyDataLayer
from dotenv import load_dotenv

load_dotenv(override=True)

API_URL = os.getenv("API_URL", "http://api:8000")
INTERNAL_API_KEY = os.getenv("INTERNAL_API_KEY", "puls-events-internal-2026")

# ── Users hardcodés ───────────────────────────────────────────────────────────
USERS = {
    "admin": ("admin123", "Admin"),
    "demo": ("demo123", "Demo User")
}

# ── Data layer SQLite pour la mémoire des chats ───────────────────────────────
@cl.data_layer
def get_data_layer():
    return SQLAlchemyDataLayer(
        conninfo="sqlite+aiosqlite:///./chainlit.db",
        ssl_require=False,
        show_logger=True
    )

# ── Authentification ──────────────────────────────────────────────────────────
@cl.password_auth_callback
def auth_callback(username: str, password: str):
    if username in USERS:
        password_check, display_name = USERS[username]
        if password == password_check:
            user = cl.User(
                identifier=username,
                display_name=display_name,
                metadata={"role": "user"}
            )
            return user
    return None


async def get_internal_token() -> str:
    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.post(
            f"{API_URL}/internal/token",
            headers={"x-api-key": INTERNAL_API_KEY}
        )
        return response.json()["access_token"]


@cl.on_chat_start
async def on_chat_start():
    user = cl.user_session.get("user")

    if user:
        data_layer = get_data_layer()
        await data_layer.create_user(user)

    session_id = cl.user_session.get("id")
    cl.user_session.set("session_id", session_id)
    cl.user_session.set("city", None)

    token = await get_internal_token()
    cl.user_session.set("jwt_token", token)

    name = user.display_name if user else "visiteur"
    await cl.Message(
        content=(
            f"👋 Bienvenue **{name}** sur **Puls-Events** !\n\n"
            "Je suis **Puls**, ton assistant culturel intelligent. "
            "Je peux t'aider à découvrir des événements culturels partout en France — "
            "concerts, expositions, spectacles, festivals et bien plus encore !\n\n"
            "💡 **Astuce** : Dis-moi ta ville pour des recommandations personnalisées.\n\n"
            "Que cherches-tu aujourd'hui ? 🎭🎵🎨"
        )
    ).send()

@cl.on_chat_resume
async def on_chat_resume(thread):
    """Reprend une conversation existante."""
    session_id = cl.user_session.get("id")
    cl.user_session.set("session_id", session_id)
    cl.user_session.set("city", None)
    
    token = await get_internal_token()
    cl.user_session.set("jwt_token", token)

@cl.on_message
async def on_message(message: cl.Message):
    session_id = cl.user_session.get("session_id")
    city = cl.user_session.get("city")
    jwt_token = cl.user_session.get("jwt_token")

    city = detect_city(message.content, city)
    cl.user_session.set("city", city)

    async with cl.Step(name="Recherche en cours...") as step:
        step.output = "Interrogation des agents (RAG, Géo, Web)..."

    msg = cl.Message(content="")
    await msg.send()

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            async with client.stream(
                "POST",
                f"{API_URL}/chat/stream",
                json={
                    "query": message.content,
                    "session_id": str(session_id),
                    "city": city,
                    "radius_km": 50.0
                },
                headers={"Authorization": f"Bearer {jwt_token}"}
            ) as response:
                if response.status_code == 401:
                    jwt_token = await get_internal_token()
                    cl.user_session.set("jwt_token", jwt_token)
                    await msg.stream_token("Session rafraîchie, repose ta question !")
                else:
                    async for chunk in response.aiter_text():
                        await msg.stream_token(chunk)

    except Exception as e:
        await msg.stream_token(f"Désolé, une erreur est survenue : {str(e)}")

    await msg.update()


@cl.on_chat_end
async def on_chat_end():
    pass


def detect_city(message: str, current_city: str = None) -> str | None:
    cities = [
        "Lille", "Paris", "Lyon", "Marseille", "Bordeaux",
        "Toulouse", "Nantes", "Strasbourg", "Rennes", "Montpellier",
        "Nice", "Grenoble", "Toulon", "Dijon", "Angers",
        "Nîmes", "Aix-en-Provence", "Saint-Étienne", "Le Havre", "Reims"
    ]
    message_lower = message.lower()
    for city in cities:
        if city.lower() in message_lower:
            return city
    return current_city