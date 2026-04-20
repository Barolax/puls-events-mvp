import os
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv(override=True)

# Configuration du chunking
CHUNK_SIZE = 512
CHUNK_OVERLAP = 64  # Sliding window — chevauchement pour ne pas couper le contexte


def event_to_text(event: dict) -> str:
    """
    Convertit un événement en texte brut pour la vectorisation.
    """
    parts = []

    if event.get("title"):
        parts.append(f"Titre : {event['title']}")
    if event.get("city"):
        parts.append(f"Ville : {event['city']}")
    if event.get("address"):
        parts.append(f"Adresse : {event['address']}")
    if event.get("date_begin"):
        parts.append(f"Date de début : {event['date_begin']}")
    if event.get("date_end"):
        parts.append(f"Date de fin : {event['date_end']}")
    if event.get("tags"):
        parts.append(f"Catégories : {', '.join(event['tags'])}")
    if event.get("description"):
        parts.append(f"Description : {event['description']}")
    if event.get("long_description"):
        parts.append(f"Détails : {event['long_description']}")

    return "\n".join(parts)


def chunk_event(event: dict) -> list[dict]:
    """
    Découpe un événement en chunks avec métadonnées.
    Utilise un chunking sémantique + sliding window.
    """
    text = event_to_text(event)

    # Si le texte est court, pas besoin de découper
    if len(text) <= CHUNK_SIZE:
        return [{
            "id": f"{event['id']}_0",
            "text": text,
            "metadata": {
                "event_id": event["id"],
                "title": event["title"],
                "city": event["city"],
                "address": event.get("address", ""),
                "latitude": event.get("latitude"),
                "longitude": event.get("longitude"),
                "date_begin": event.get("date_begin", ""),
                "date_end": event.get("date_end", ""),
                "tags": event.get("tags", []),
                "source": event.get("source", "open_agenda"),
                "chunk_index": 0,
                "total_chunks": 1
            }
        }]

    # Chunking sémantique avec sliding window
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    chunks = splitter.split_text(text)

    return [
        {
            "id": f"{event['id']}_{i}",
            "text": chunk,
            "metadata": {
                "event_id": event["id"],
                "title": event["title"],
                "city": event["city"],
                "address": event.get("address", ""),
                "latitude": event.get("latitude"),
                "longitude": event.get("longitude"),
                "date_begin": event.get("date_begin", ""),
                "date_end": event.get("date_end", ""),
                "tags": event.get("tags", []),
                "source": event.get("source", "open_agenda"),
                "chunk_index": i,
                "total_chunks": len(chunks)
            }
        }
        for i, chunk in enumerate(chunks)
    ]


def chunk_events(events: list[dict]) -> list[dict]:
    """
    Découpe tous les événements en chunks.
    """
    all_chunks = []
    for event in events:
        chunks = chunk_event(event)
        all_chunks.extend(chunks)

    print(f"{len(events)} événements → {len(all_chunks)} chunks")
    return all_chunks


if __name__ == "__main__":
    from open_agenda import fetch_all_events

    events = fetch_all_events(size_per_agenda=10)
    chunks = chunk_events(events)

    print(f"\nExemple de chunk :")
    print(f"ID      : {chunks[0]['id']}")
    print(f"Texte   : {chunks[0]['text'][:200]}...")
    print(f"Metadata: {chunks[0]['metadata']}")