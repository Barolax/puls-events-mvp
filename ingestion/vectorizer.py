import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from mistralai.client import Mistral

load_dotenv(override=True)

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "puls_events")
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
EMBEDDING_MODEL = "mistral-embed"
VECTOR_SIZE = 1024  # Dimension des embeddings Mistral


def get_qdrant_client() -> QdrantClient:
    return QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)


def init_collection(client: QdrantClient):
    """
    Crée la collection Qdrant si elle n'existe pas.
    """
    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME not in existing:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=VECTOR_SIZE,
                distance=Distance.COSINE
            )
        )
        print(f"Collection '{COLLECTION_NAME}' créée")
    else:
        print(f"Collection '{COLLECTION_NAME}' déjà existante")


def generate_embeddings(texts: list[str]) -> list[list[float]]:
    """
    Génère les embeddings via Mistral AI.
    Traitement par batch de 32 pour éviter les timeouts.
    """
    client = Mistral(api_key=MISTRAL_API_KEY)
    embeddings = []
    batch_size = 32

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        print(f"  Embedding batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}...")
        response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            inputs=batch
        )
        embeddings.extend([e.embedding for e in response.data])

    return embeddings


def vectorize_and_store(chunks: list[dict]):
    """
    Vectorise les chunks et les stocke dans Qdrant.
    """
    client = get_qdrant_client()
    init_collection(client)

    texts = [chunk["text"] for chunk in chunks]
    print(f"Génération des embeddings pour {len(texts)} chunks...")
    embeddings = generate_embeddings(texts)

    points = [
        PointStruct(
            id=abs(hash(chunk["id"])) % (2**63),
            vector=embedding,
            payload={
                "text": chunk["text"],
                **chunk["metadata"]
            }
        )
        for chunk, embedding in zip(chunks, embeddings)
    ]

    client.upsert(
        collection_name=COLLECTION_NAME,
        points=points
    )

    print(f"{len(points)} vecteurs stockés dans Qdrant ✓")


if __name__ == "__main__":
    from open_agenda import fetch_all_events
    from chunking import chunk_events
    from validation import validate_events

    print("=== Pipeline d'ingestion ===\n")

    print("1. Récupération des événements...")
    events = fetch_all_events(size_per_agenda=10)

    print("\n2. Validation des données...")
    valid_events, invalid_events = validate_events(events)

    print("\n3. Chunking...")
    chunks = chunk_events(valid_events)

    print("\n4. Vectorisation et stockage dans Qdrant...")
    vectorize_and_store(chunks)

    print("\n=== Pipeline terminé ! ===")