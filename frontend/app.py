import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'agents'))

import chainlit as cl
from dotenv import load_dotenv
from graph import run_pipeline
from agent_memory import clear_session

load_dotenv(override=True)


@cl.on_chat_start
async def on_chat_start():
    """
    Initialise la session au démarrage du chat.
    """
    session_id = cl.user_session.get("id")
    cl.user_session.set("session_id", session_id)
    cl.user_session.set("city", None)

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
    Traite chaque message de l'utilisateur.
    """
    session_id = cl.user_session.get("session_id")
    city = cl.user_session.get("city")

    # Détection automatique de la ville dans le message
    city = detect_city(message.content, city)
    cl.user_session.set("city", city)

    # Indicateur de chargement
    async with cl.Step(name="Recherche en cours...") as step:
        step.output = f"Interrogation des agents (RAG, Géo, Web)..."

        response = run_pipeline(
            query=message.content,
            session_id=session_id,
            city=city,
            radius_km=50
        )

    await cl.Message(content=response).send()


@cl.on_chat_end
async def on_chat_end():
    """
    Nettoie la session à la fermeture du chat.
    """
    session_id = cl.user_session.get("session_id")
    if session_id:
        clear_session(session_id)


def detect_city(message: str, current_city: str = None) -> str | None:
    """
    Détecte la ville mentionnée dans le message.
    Liste des villes principales françaises.
    """
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


if __name__ == "__main__":
    from chainlit.cli import run_chainlit
    run_chainlit(__file__)