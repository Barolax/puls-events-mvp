import os
import sys
from dotenv import load_dotenv, find_dotenv
from mistralai import Mistral

load_dotenv(find_dotenv(), override=True)

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
MISTRAL_MODEL = "mistral-small-latest"  # Modèle léger pour les guardrails

# Sujets autorisés
ALLOWED_TOPICS = [
    "événement", "événements", "concert", "exposition", "spectacle",
    "festival", "théâtre", "musée", "cinéma", "danse", "musique",
    "sortie", "agenda", "culture", "culturel", "culturelle",
    "ville", "lieu", "date", "billet", "réservation", "gratuit",
    "famille", "enfant", "adulte", "soirée", "weekend", "vacances"
]

# Sujets interdits
FORBIDDEN_TOPICS = [
    "recette", "cuisine", "drogue", "arme", "violence", "politique",
    "religion", "sexe", "pornographie", "hack", "piratage", "virus",
    "arnaque", "fraude", "illégal", "criminel"
]

# Nombre maximum de messages hors sujet avant blocage
MAX_OFF_TOPIC_COUNT = 2


def is_cultural_query(query: str) -> bool:
    """
    Vérification rapide par mots-clés — premier niveau de filtre.
    """
    query_lower = query.lower()

    # Vérifie les sujets interdits
    for topic in FORBIDDEN_TOPICS:
        if topic in query_lower:
            return False

    # Vérifie si au moins un sujet autorisé est présent
    for topic in ALLOWED_TOPICS:
        if topic in query_lower:
            return True

    return None  # Indéterminé — on laisse le LLM décider


def classify_query_with_llm(query: str) -> bool:
    """
    Classification par LLM — second niveau de filtre.
    Utilisé quand la vérification par mots-clés est indéterminée.
    """
    client = Mistral(api_key=MISTRAL_API_KEY)

    prompt = f"""Tu es un classificateur de requêtes pour une plateforme d'événements culturels.
    
Ta tâche : déterminer si la question suivante est liée aux événements culturels, sorties, 
loisirs ou activités en France.

Réponds UNIQUEMENT par "OUI" ou "NON".

Question : {query}

Est-ce une question liée aux événements culturels ou sorties ?"""

    response = client.chat.complete(
        model=MISTRAL_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=5
    )

    answer = response.choices[0].message.content.strip().upper()
    return answer == "OUI"


def check_off_topic_drift(history: list[dict], threshold: int = MAX_OFF_TOPIC_COUNT) -> bool:
    """
    Vérifie si la conversation dérive hors sujet.
    Retourne True si trop de messages hors sujet détectés.
    """
    off_topic_count = 0
    for msg in history[-5:]:  # Vérifie les 5 derniers messages
        if msg.get("role") == "user":
            result = is_cultural_query(msg.get("content", ""))
            if result is False:
                off_topic_count += 1

    return off_topic_count >= threshold


def validate_query(query: str, history: list[dict] = None) -> dict:
    """
    Valide une requête avant de l'envoyer au pipeline.
    
    Returns:
        {
            "allowed": bool,
            "reason": str,
            "warning": str | None
        }
    """
    # 1. Vérification dérive conversationnelle
    if history and check_off_topic_drift(history):
        return {
            "allowed": False,
            "reason": "drift",
            "warning": "La conversation a dévié de notre sujet principal."
        }

    # 2. Vérification par mots-clés
    keyword_result = is_cultural_query(query)

    if keyword_result is False:
        return {
            "allowed": False,
            "reason": "forbidden_topic",
            "warning": "Cette question ne correspond pas à notre domaine."
        }

    if keyword_result is True:
        return {"allowed": True, "reason": "keyword_match", "warning": None}

    # 3. Classification LLM si indéterminé
    llm_result = classify_query_with_llm(query)

    if not llm_result:
        return {
            "allowed": False,
            "reason": "off_topic",
            "warning": "Cette question sort de notre domaine des événements culturels."
        }

    return {"allowed": True, "reason": "llm_approved", "warning": None}


def get_refusal_message() -> str:
    """
    Message de refus poli pour les requêtes hors sujet.
    """
    return (
        "Je suis **Puls**, ton assistant dédié aux événements culturels en France 🎭\n\n"
        "Je ne peux pas t'aider sur ce sujet, mais je serais ravi de te recommander :\n"
        "- Des **concerts** près de chez toi 🎵\n"
        "- Des **expositions** à visiter 🎨\n"
        "- Des **spectacles** et **festivals** 🎪\n\n"
        "Qu'est-ce qui t'intéresse comme sortie culturelle ? 😊"
    )


if __name__ == "__main__":
    print("=== Test Guardrails ===\n")

    test_queries = [
        "Quels concerts y a-t-il à Lille ce weekend ?",
        "Donne-moi la recette du gâteau au chocolat",
        "Y a-t-il des expositions à Paris ?",
        "Comment fabriquer une bombe ?",
        "C'est quoi le meilleur festival de jazz en France ?",
        "Explique-moi la politique française",
    ]

    for query in test_queries:
        result = validate_query(query)
        status = "✅" if result["allowed"] else "❌"
        print(f"{status} '{query[:50]}...' → {result['reason']}")