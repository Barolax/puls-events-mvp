import os
import httpx
import chainlit as cl
from dotenv import load_dotenv

load_dotenv(override=True)

API_URL = os.getenv("API_URL", "http://api:8000")
INTERNAL_API_KEY = os.getenv("INTERNAL_API_KEY", "puls-events-internal-2026")


async def get_internal_token() -> str:
    """
    Récupère un JWT via la clé API interne.
    """
    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.post(
            f"{API_URL}/internal/token",
            headers={"x-api-key": INTERNAL_API_KEY}
        )
        return response.json()["access_token"]


@cl.on_chat_start
async def on_chat_start():
    """
    Initialise la session et récupère un JWT interne.
    """
    session_id = cl.user_session.get("id")
    cl.user_session.set("session_id", session_id)
    cl.user_session.set("city", None)

    # Récupère le JWT une seule fois au démarrage
    token = await get_internal_token()
    cl.user_session.set("jwt_token", token)

    await cl.Message(
        content=(
            "👋 Bienvenue sur **Puls-Events** !\n\n"
            "Je suis **Puls**, ton assistant culturel intelligent. "
            "Je peux t'aider à découvrir des événements culturels partout en France — "
            "concerts, expositions, spectacles, festivals et bien plus encore !\n\n"
            "💡 **Astuce** : Dis-moi ta ville pour des recommandations personnalisées.\n\n"
            "Que cherches-tu aujourd'hui ? 🎭🎵🎨"
        )
    ).send()


@cl.on_message
async def on_message(message: cl.Message):
    """
    Traite chaque message via l'API FastAPI.
    """
    session_id = cl.user_session.get("session_id")
    city = cl.user_session.get("city")
    jwt_token = cl.user_session.get("jwt_token")

    # Détection automatique de la ville
    city = detect_city(message.content, city)
    cl.user_session.set("city", city)

    async with cl.Step(name="Recherche en cours...") as step:
        step.output = "Interrogation des agents (RAG, Géo, Web)..."

        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    f"{API_URL}/chat",
                    json={
                        "query": message.content,
                        "session_id": str(session_id),
                        "city": city,
                        "radius_km": 50.0
                    },
                    headers={"Authorization": f"Bearer {jwt_token}"}
                )

                if response.status_code == 200:
                    answer = response.json()["response"]
                elif response.status_code == 401:
                    # Token expiré — en recrée un
                    jwt_token = await get_internal_token()
                    cl.user_session.set("jwt_token", jwt_token)
                    answer = "Session rafraîchie, repose ta question !"
                else:
                    answer = f"Erreur : {response.status_code}"

        except Exception as e:
            answer = f"Désolé, une erreur est survenue : {str(e)}"

    await cl.Message(content=answer).send()


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