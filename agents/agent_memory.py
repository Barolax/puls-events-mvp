import os
import json
import redis
from datetime import datetime
from dotenv import load_dotenv

load_dotenv(override=True)

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
SESSION_TTL = 3600  # 1 heure — expiration automatique des sessions
MAX_HISTORY = 10    # Nombre maximum de messages par session


def get_redis_client() -> redis.Redis:
    return redis.Redis(
        host=REDIS_HOST,
        port=REDIS_PORT,
        decode_responses=True
    )


def get_session_key(session_id: str) -> str:
    return f"puls_events:session:{session_id}"


def get_history(session_id: str) -> list[dict]:
    """
    Récupère l'historique de la conversation.
    """
    client = get_redis_client()
    key = get_session_key(session_id)
    raw = client.get(key)

    if not raw:
        return []

    return json.loads(raw)


def save_message(session_id: str, role: str, content: str):
    """
    Sauvegarde un message dans l'historique.
    role: 'user' ou 'assistant'
    """
    client = get_redis_client()
    key = get_session_key(session_id)

    history = get_history(session_id)

    history.append({
        "role": role,
        "content": content,
        "timestamp": datetime.now().isoformat()
    })

    # Limite la taille de l'historique
    if len(history) > MAX_HISTORY:
        history = history[-MAX_HISTORY:]

    client.setex(key, SESSION_TTL, json.dumps(history))


def format_history_for_llm(history: list[dict]) -> list[dict]:
    """
    Formate l'historique pour l'envoyer au LLM.
    """
    return [
        {"role": msg["role"], "content": msg["content"]}
        for msg in history
    ]


def clear_session(session_id: str):
    """
    Supprime une session.
    """
    client = get_redis_client()
    client.delete(get_session_key(session_id))


def run_memory_agent(state: dict) -> dict:
    """
    Agent Mémoire — appelé par LangGraph.
    Récupère l'historique et sauvegarde le message utilisateur.
    """
    session_id = state.get("session_id", "default")
    query = state.get("query", "")

    print(f"Agent Mémoire — session : {session_id}")

    # Récupère l'historique
    history = get_history(session_id)
    print(f"  → {len(history)} messages en mémoire")

    # Sauvegarde le message utilisateur
    save_message(session_id, "user", query)

    return {
        **state,
        "history": format_history_for_llm(history),
        "memory_done": True
    }


if __name__ == "__main__":
    session_id = "test_session_001"

    # Simule une conversation
    print("=== Test Agent Mémoire ===\n")

    save_message(session_id, "user", "Quels concerts y a-t-il à Lille ?")
    save_message(session_id, "assistant", "Il y a plusieurs concerts à Lille ce weekend...")
    save_message(session_id, "user", "Et des expositions ?")

    history = get_history(session_id)
    print(f"{len(history)} messages en mémoire :")
    for msg in history:
        print(f"  [{msg['role']}] {msg['content'][:60]}...")

    # Test via LangGraph state
    state = {"session_id": session_id, "query": "Et du théâtre ?"}
    result = run_memory_agent(state)
    print(f"\nHistorique formaté pour LLM : {len(result['history'])} messages")

    # Nettoyage
    clear_session(session_id)
    print("Session nettoyée ✓")