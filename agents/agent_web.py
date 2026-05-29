import os
from dotenv import load_dotenv
from smolagents import DuckDuckGoSearchTool

load_dotenv(override=True)

MAX_RESULTS = 5

# Whitelist de domaines fiables pour les événements culturels
TRUSTED_DOMAINS = [
    "openagenda.com",
    "sortiraparis.com",
    "lilleaddict.fr",
    "tourisme-lille.fr",
    "petitfute.com",
    "linternaute.com",
    "telerama.fr",
    "evene.lefigaro.fr",
    "fnac.com",
    "billetreduc.com",
    "shotgun.live",
    "facebook.com/events",
    "metropole-europeenne-lille.fr",
    "culture.fr",
    "tourisme-nordpasdecalais.fr",
    "auvergnerhonealpes-tourisme.com",
    "gironde.fr",
    "iledefrance.fr",
    "grandest.fr",
    "mairie-paris.fr",
]


def is_trusted_source(text: str) -> bool:
    """
    Vérifie si le résultat provient d'une source de confiance.
    """
    text_lower = text.lower()
    return any(domain in text_lower for domain in TRUSTED_DOMAINS)


def search_web(query: str) -> list[dict]:
    """
    Recherche des événements en temps réel via DuckDuckGo.
    Filtre les résultats selon la whitelist de domaines fiables.
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
    Filtre selon la whitelist si des résultats fiables existent.
    """
    if not raw:
        return []

    results = []
    trusted_results = []
    entries = raw.strip().split("\n\n")

    for i, entry in enumerate(entries[:MAX_RESULTS * 2]):
        if entry.strip():
            doc = {
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
            }
            if is_trusted_source(entry):
                doc["score"] = 0.8
                trusted_results.append(doc)
            else:
                results.append(doc)

    final_results = trusted_results if trusted_results else results

    if trusted_results:
        print(f"  → {len(trusted_results)} résultats de sources fiables (whitelist)")
    else:
        print(f"  → Aucune source whitelistée, fallback sur résultats généraux")

    return final_results[:MAX_RESULTS]


def should_search_web(state: dict) -> bool:
    """
    Décide si une recherche web est nécessaire.
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

    search_query = f"événements culturels {city} {query}" if city else f"événements culturels France {query}"
    web_results = search_web(search_query)
    print(f"  → {len(web_results)} résultats web")

    all_documents = state.get("documents", []) + web_results
    return {**state, "documents": all_documents, "web_done": True}