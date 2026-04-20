import os
from dotenv import load_dotenv
from smolagents import DuckDuckGoSearchTool

load_dotenv(override=True)

MAX_RESULTS = 5


def search_web(query: str) -> list[dict]:
    """
    Recherche des événements en temps réel via DuckDuckGo.
    Pas de clé API requise.
    """
    tool = DuckDuckGoSearchTool()

    try:
        raw_results = tool(query)
        return parse_results(raw_results, query)
    except Exception as e:
        print(f"Erreur recherche web : {e}")
        return []


def parse_results(raw: str, query: str) -> list[dict]:
    """
    Parse les résultats bruts en liste de documents.
    """
    if not raw:
        return []

    # DuckDuckGoSearchTool retourne un string formaté
    results = []
    entries = raw.strip().split("\n\n")

    for i, entry in enumerate(entries[:MAX_RESULTS]):
        if entry.strip():
            results.append({
                "title": f"Résultat web {i+1}",
                "text": entry.strip(),
                "city": "",
                "address": "",
                "date_begin": "",
                "tags": [],
                "latitude": None,
                "longitude": None,
                "distance_km": None,
                "score": 0.5,
                "source": "web"
            })

    return results


def should_search_web(state: dict) -> bool:
    """
    Décide si une recherche web est nécessaire.
    On cherche sur le web si :
    - Pas assez de résultats RAG
    - La requête contient des mots clés temps réel
    """
    documents = state.get("documents", [])
    query = state.get("query", "").lower()

    real_time_keywords = [
        "aujourd'hui", "ce soir", "ce weekend", "demain",
        "cette semaine", "prochainement", "bientôt"
    ]

    has_real_time = any(kw in query for kw in real_time_keywords)
    not_enough_results = len(documents) < 3

    return has_real_time or not_enough_results


def run_web_agent(state: dict) -> dict:
    """
    Agent Web — appelé par LangGraph.
    Enrichit les résultats RAG avec une recherche web temps réel.
    """
    query = state.get("query", "")
    city = state.get("city", "")

    print(f"Agent Web — recherche : '{query}'")

    if not should_search_web(state):
        print("  → Recherche web non nécessaire")
        return {**state, "web_done": True}

    # Enrichit la requête avec la ville si disponible
    search_query = f"événements culturels {city} {query}" if city else f"événements culturels France {query}"

    web_results = search_web(search_query)
    print(f"  → {len(web_results)} résultats web")

    # Fusionne résultats RAG + web
    existing_docs = state.get("documents", [])
    all_docs = existing_docs + web_results

    return {
        **state,
        "documents": all_docs,
        "web_done": True
    }


if __name__ == "__main__":
    print("=== Test Agent Web ===\n")

    # Test 1 — requête temps réel
    state = {
        "query": "concerts ce weekend à Lille",
        "city": "Lille",
        "documents": []
    }

    result = run_web_agent(state)
    print(f"\n{len(result['documents'])} documents au total\n")
    for doc in result["documents"][:3]:
        print(f"- [{doc.get('source', 'rag')}] {doc['text'][:100]}...")