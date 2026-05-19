import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from mistralai import Mistral
import cohere

load_dotenv(override=True)

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "puls_events")
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
EMBEDDING_MODEL = "mistral-embed"
TOP_K = 10   # On récupère plus de résultats pour le reranking
TOP_K_RERANK = 5  # On garde les 5 meilleurs après reranking


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


def rerank_documents(query: str, documents: list[dict]) -> list[dict]:
    """
    Réordonne les documents par pertinence avec Cohere Rerank.
    """
    if not COHERE_API_KEY or not documents:
        return documents

    try:
        co = cohere.Client(COHERE_API_KEY)
        texts = [f"{doc['title']} : {doc['text'][:300]}" for doc in documents]

        results = co.rerank(
            model="rerank-multilingual-v3.0",
            query=query,
            documents=texts,
            top_n=TOP_K_RERANK
        )

        reranked = []
        for result in results.results:
            doc = documents[result.index]
            doc["rerank_score"] = result.relevance_score
            reranked.append(doc)

        print(f"  → Cohere Rerank : {len(reranked)} documents reranked")
        return reranked

    except Exception as e:
        print(f"  ⚠️ Cohere Rerank error : {e} — fallback sur résultats Qdrant")
        return documents[:TOP_K_RERANK]


def search_events(query: str, city_filter: str = None) -> list[dict]:
    """
    Recherche les événements pertinents dans Qdrant + Cohere Rerank.
    """
    client = get_qdrant_client()
    query_vector = embed_query(query)

    query_filter = None
    if city_filter:
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

    documents = [
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

    # Cohere Rerank
    return rerank_documents(query, documents)


def run_rag_agent(state: dict) -> dict:
    """
    Agent RAG — appelé par LangGraph.
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