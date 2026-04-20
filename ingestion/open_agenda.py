import requests
import os
from dotenv import load_dotenv

load_dotenv(override=True)

OPEN_AGENDA_API_KEY = os.getenv("OPENAGENDA_API_KEY")
BASE_URL = "https://api.openagenda.com/v2"

AGENDA_UIDS = {
    "hauts_de_france": 57621068,
    "hauts_de_france_musees": 501473,
    "bordeaux": 1108324,
    "auvergne_rhone_alpes": 828001,
}

def fetch_events_from_agenda(agenda_uid: int, size: int = 100, after: str = None) -> list[dict]:
    """
    Récupère les événements d'un agenda spécifique.
    """
    params = {
        "key": OPEN_AGENDA_API_KEY,
        "size": size,
        "includeLabels": 1,
        "lang": "fr"
    }

    if after:
        params["timings[gte]"] = after

    try:
        response = requests.get(
            f"{BASE_URL}/agendas/{agenda_uid}/events",
            params=params,
            timeout=10
        )
        response.raise_for_status()
        data = response.json()
        events = data.get("events", [])
        return [format_event(e) for e in events]

    except requests.exceptions.RequestException as e:
        print(f"Erreur agenda {agenda_uid} : {e}")
        return []


def fetch_all_events(size_per_agenda: int = 100, after: str = None) -> list[dict]:
    """
    Récupère les événements de tous les agendas configurés.
    """
    all_events = []
    for region, uid in AGENDA_UIDS.items():
        print(f"Récupération agenda : {region}...")
        events = fetch_events_from_agenda(uid, size=size_per_agenda, after=after)
        print(f"  → {len(events)} événements")
        all_events.extend(events)

    # Dédoublonnage par id
    seen = set()
    unique_events = []
    for e in all_events:
        if e["id"] not in seen:
            seen.add(e["id"])
            unique_events.append(e)

    print(f"\nTotal : {len(unique_events)} événements uniques")
    return unique_events


def format_event(event: dict) -> dict:
    """
    Formate un événement brut Open Agenda en structure propre.
    """
    title = event.get("title", {}).get("fr", "Sans titre")
    description = event.get("description", {}).get("fr", "")
    long_description = event.get("longDescription", {}).get("fr", "")

    location = event.get("location", {})
    city = location.get("city", "")
    address = location.get("address", "")
    latitude = location.get("latitude")
    longitude = location.get("longitude")

    timings = event.get("timings", [])
    date_begin = timings[0].get("begin", "") if timings else ""
    date_end = timings[-1].get("end", "") if timings else ""

    tags = [
        label.get("label", {}).get("fr", "")
        for label in event.get("labels", [])
    ]

    return {
        "id": str(event.get("uid", "")),
        "title": title,
        "description": description,
        "long_description": long_description,
        "city": city,
        "address": address,
        "latitude": latitude,
        "longitude": longitude,
        "date_begin": date_begin,
        "date_end": date_end,
        "tags": tags,
        "source": "open_agenda"
    }


if __name__ == "__main__":
    events = fetch_all_events(size_per_agenda=10)
    for e in events[:3]:
        print(f"\n- {e['title']}")
        print(f"  Ville : {e['city']}")
        print(f"  Date  : {e['date_begin']}")
        print(f"  Tags  : {e['tags']}")