import pytest
import os
import sys

sys.path.insert(0, '/app/agents')

from dotenv import load_dotenv
load_dotenv(override=True)


class TestAgentGeo:
    """Tests de l'agent géolocalisation."""

    def test_geo_known_city(self):
        """Vérifie que l'agent géocode une ville connue."""
        from agent_geo import run_geo_agent
        state = {
            "query": "concerts à Lille",
            "city": "Lille",
            "latitude": None,
            "longitude": None,
            "radius_km": 50,
            "documents": [],
            "geo_done": False
        }
        result = run_geo_agent(state)
        assert result.get("geo_done") is True
        assert result.get("latitude") is not None
        assert result.get("longitude") is not None

    def test_geo_lille_coordinates(self):
        """Vérifie les coordonnées de Lille."""
        from agent_geo import run_geo_agent
        state = {
            "query": "événements",
            "city": "Lille",
            "latitude": None,
            "longitude": None,
            "radius_km": 50,
            "documents": [],
            "geo_done": False
        }
        result = run_geo_agent(state)
        lat = result.get("latitude")
        lon = result.get("longitude")
        if lat and lon:
            assert 50.0 < lat < 51.0, "Latitude de Lille incorrecte"
            assert 2.5 < lon < 3.5, "Longitude de Lille incorrecte"

    def test_geo_no_city(self):
        """Vérifie le comportement sans ville."""
        from agent_geo import run_geo_agent
        state = {
            "query": "événements en France",
            "city": None,
            "latitude": None,
            "longitude": None,
            "radius_km": 50,
            "documents": [],
            "geo_done": False
        }
        result = run_geo_agent(state)
        assert result.get("geo_done") is True
        assert result.get("latitude") is None
        assert result.get("longitude") is None

    def test_geo_paris_coordinates(self):
        """Vérifie les coordonnées de Paris."""
        from agent_geo import run_geo_agent
        state = {
            "query": "concerts",
            "city": "Paris",
            "latitude": None,
            "longitude": None,
            "radius_km": 50,
            "documents": [],
            "geo_done": False
        }
        result = run_geo_agent(state)
        lat = result.get("latitude")
        lon = result.get("longitude")
        if lat and lon:
            assert 48.0 < lat < 49.5, "Latitude de Paris incorrecte"
            assert 1.5 < lon < 3.0, "Longitude de Paris incorrecte"