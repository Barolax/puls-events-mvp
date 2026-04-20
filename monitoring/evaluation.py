import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'agents'))

from datetime import datetime
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv(), override=True)

from langfuse import Langfuse

langfuse = Langfuse(
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
)


def trace_pipeline(
    query: str,
    response: str,
    session_id: str,
    city: str = None,
    documents: list[dict] = None,
    latency_ms: int = None
) -> str:
    trace_id = langfuse.create_trace_id()

    # Span RAG
    rag_span = langfuse.start_observation(
        name="rag-retrieval",
        as_type="retriever",
        input={"query": query},
        output={"documents_count": len(documents) if documents else 0},
        metadata={"city_filter": city}
    )
    rag_span.end()

    # Span génération LLM
    gen_span = langfuse.start_observation(
        name="mistral-generation",
        as_type="generation",
        model="mistral-large-latest",
        input=query,
        output=response,
        metadata={"session_id": session_id}
    )
    gen_span.end()

    langfuse.flush()
    return trace_id


def score_response(trace_id: str, score: float, comment: str = None):
    """
    Ajoute un score de qualité à une trace.
    """
    langfuse.create_score(
        trace_id=trace_id,
        name="user-satisfaction",
        value=score,
        comment=comment
    )
    langfuse.flush()


def check_connection() -> bool:
    try:
        langfuse.auth_check()
        return True
    except Exception as e:
        print(f"Erreur connexion Langfuse : {e}")
        return False


if __name__ == "__main__":
    print("=== Test Monitoring Langfuse ===\n")

    if check_connection():
        print("Connexion Langfuse OK ✓")
    else:
        print("Erreur connexion Langfuse ✗")
        exit(1)

    trace_id = trace_pipeline(
        query="Quels concerts y a-t-il à Lille ce weekend ?",
        response="Il y a plusieurs concerts à Lille ce weekend...",
        session_id="test_monitoring",
        city="Lille",
        documents=[{"title": "Concert test"}],
        latency_ms=1200
    )
    print(f"Trace créée : {trace_id}")

    score_response(trace_id, score=0.9, comment="Réponse très pertinente")
    print("Score ajouté ✓")

    print("\nVa sur cloud.langfuse.com pour voir les traces !")