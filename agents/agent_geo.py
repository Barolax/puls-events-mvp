import os
import math
import requests
from dotenv import load_dotenv

load_dotenv(override=True)

DEFAULT_RADIUS_KM = 50  # Rayon de recherche par défaut en km


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calcule la distance en km entre deux points GPS.
    """
    R = 6371  # Rayon de la Terre en km
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
    return R * 2 * math.asin(math.sqrt(a))


def geocode_city(city: str) -> tuple[float, float] | None:
    """
    Convertit un nom de ville en coordonnées GPS via API Nominatim (OpenStreetMap).
    Gratuit, pas de clé API requise.
    """
    try:
        response = requests.get(
            "https://nominatim.openstreetmap.org/search",
            params={"q": city, "format": "json", "limit": 1, "countrycodes": "fr"},
            headers={"User-Agent": "puls-events-mvp"},
            timeout=5
        )
        data = response.json()
        if data:
            return float(data[0]["lat"]), float(data[0]["lon"])
    except Exception as e:
        print(f"Erreur géocodage '{city}' : {e}")
    return None


def filter_by_proximity(
    documents: list[dict],
    user_lat: float,
    user_lon: float,
    radius_km: float = DEFAULT_RADIUS_KM
) -> list[dict]:
    """
    Filtre et enrichit les documents avec la distance utilisateur.
    """
    results = []
    for doc in documents:
        lat = doc.get("latitude")
        lon = doc.get("longitude")

        if lat is None or lon is None:
            results.append({**doc, "distance_km": None})
            continue

        distance = haversine_distance(user_lat, user_lon, lat, lon)

        if distance <= radius_km:
            results.append({**doc, "distance_km": round(distance, 1)})

    # Trie par distance
    results.sort(key=lambda x: x["distance_km"] if x["distance_km"] is not None else 9999)
    return results


def run_geo_agent(state: dict) -> dict:
    """
    Agent Géo — appelé par LangGraph.
    Enrichit les documents avec la distance et filtre par proximité.
    """
    documents = state.get("documents", [])
    user_city = state.get("city")
    user_lat = state.get("latitude")
    user_lon = state.get("longitude")
    radius_km = state.get("radius_km", DEFAULT_RADIUS_KM)

    print(f"Agent Géo — ville: {user_city or 'non renseignée'}")

    # Géocodage si on a une ville mais pas de coordonnées
    if user_city and not (user_lat and user_lon):
        coords = geocode_city(user_city)
        if coords:
            user_lat, user_lon = coords
            print(f"  → Coordonnées : {user_lat:.4f}, {user_lon:.4f}")

    # Filtrage par proximité si on a des coordonnées
    if user_lat and user_lon:
        documents = filter_by_proximity(documents, user_lat, user_lon, radius_km)
        print(f"  → {len(documents)} événements dans un rayon de {radius_km}km")
    else:
        print("  → Pas de coordonnées, pas de filtrage géographique")

    return {
        **state,
        "documents": documents,
        "latitude": user_lat,
        "longitude": user_lon,
        "geo_done": True
    }


if __name__ == "__main__":
    from agent_rag import run_rag_agent

    print("=== Test Agent Géo ===\n")

    # D'abord on récupère des documents via RAG
    state = {
        "query": "concert musique",
        "city": "Lille",
        "radius_km": 30
    }

    state = run_rag_agent(state)
    print(f"\nRAG : {len(state['documents'])} documents")

    # Puis on filtre par géo
    state = run_geo_agent(state)
    print(f"\nRésultats après filtrage géo :")
    for doc in state["documents"]:
        dist = doc.get("distance_km")
        dist_str = f"{dist}km" if dist else "distance inconnue"
        print(f"  - {doc['title']} ({doc['city']}) — {dist_str}")