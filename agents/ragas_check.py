import os
from dotenv import load_dotenv

load_dotenv(override=True)


FAITHFULNESS_THRESHOLD = 0.5

def compute_faithfulness(response: str, documents: list[dict]) -> float:
    """
    Calcule un score de faithfulness simplifié.
    Vérifie si les éléments clés de la réponse sont ancrés dans les documents.
    """
    if not documents:
        return 0.0

    # Extrait les textes des documents
    doc_texts = " ".join([
        doc.get("text", "") + " " + doc.get("title", "")
        for doc in documents[:5]
    ]).lower()

    if not doc_texts.strip():
        return 0.0

    # Extrait les mots significatifs de la réponse (> 4 lettres)
    response_words = [
        w.lower().strip(".,!?;:")
        for w in response.split()
        if len(w) > 4
    ]

    if not response_words:
        return 0.0

    # Calcule le ratio de mots ancrés dans les documents
    grounded = sum(1 for w in response_words if w in doc_texts)
    score = grounded / len(response_words)

    return round(score, 3)


def run_ragas_check(state: dict) -> dict:
    """
    Node LangGraph — Output guardrail avec vérification faithfulness.
    Log dans Langfuse avec tag WARNING si score bas.
    """
    response = state.get("response", "")
    documents = state.get("documents", [])
    query = state.get("query", "")
    session_id = state.get("session_id", "default")

    score = compute_faithfulness(response, documents)

    is_grounded = score >= FAITHFULNESS_THRESHOLD
    tags = ["ragas", "faithfulness"]

    if not is_grounded:
        tags.append("WARNING")
        print(f"⚠️  RAGAS faithfulness bas : {score} (seuil: {FAITHFULNESS_THRESHOLD})")
        print(f"   Query: {query[:80]}")
    else:
        print(f"✅ RAGAS faithfulness : {score}")

    # Log
    print(f"[LANGFUSE] ragas_faithfulness | score={score} | grounded={is_grounded} | tags={tags} | query={query[:80]}")

    # On ne bloque pas l'utilisateur — on log seulement
    return {**state, "faithfulness_score": score}