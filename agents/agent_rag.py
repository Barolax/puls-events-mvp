import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import SearchRequest
from mistralai.client import Mistral

load_dotenv(override=True)

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "puls_events")
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
EMBEDDING_MODEL = "mistral-embed"
TOP_K = 5  # Nombre de résultats à retourner


def get_qdrant_client() -> QdrantClient:
    return QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)


def embed_query(query: str) -> list[float]:
    """
    Vectorise la question de l'utilisateur.
    """
    client = Mistral(api_key=MISTRAL_API_KEY)
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        inputs=[query]
    )
    return response.data[0].embedding


def search_events(query: str, city_filter: str = None) -> list[dict]:
    """
    Recherche les événements pertinents dans Qdrant.
    """
    client = get_qdrant_client()
    query_vector = embed_query(query)

    # Filtre optionnel par ville
    query_filter = None
    if city_filter:
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        query_filter = Filter(
            must=[FieldCondition(
                key="city",
                match=MatchValue(value=city_filter)
            )]
        )

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=TOP_K,
        query_filter=query_filter,
        with_payload=True
    ).points

    return [
        {
            "score": hit.score,
            "text": hit.payload.get("text", ""),
            "title": hit.payload.get("title", ""),
            "city": hit.payload.get("city", ""),
            "address": hit.payload.get("address", ""),
            "date_begin": hit.payload.get("date_begin", ""),
            "tags": hit.payload.get("tags", []),
            "latitude": hit.payload.get("latitude"),
            "longitude": hit.payload.get("longitude"),
        }
        for hit in results
    ]


def run_rag_agent(state: dict) -> dict:
    """
    Agent RAG — appelé par LangGraph.
    Reçoit le state et retourne les documents pertinents.
    """
    query = state.get("query", "")
    city_filter = state.get("city")

    print(f"Agent RAG — recherche : '{query}' (ville: {city_filter or 'toutes'})")

    documents = search_events(query, city_filter=city_filter)

    return {
        **state,
        "documents": documents,
        "rag_done": True
    }


if __name__ == "__main__":
    # Test direct de l'agent
    test_state = {
        "query": "spectacle de danse à Lille",
        "city": None
    }

    result = run_rag_agent(test_state)

    print(f"\n{len(result['documents'])} documents trouvés :\n")
    for doc in result["documents"]:
        print(f"- {doc['title']} ({doc['city']}) — score: {doc['score']:.3f}")