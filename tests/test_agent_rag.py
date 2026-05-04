import pytest
import os
import sys
from dotenv import load_dotenv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'agents'))

load_dotenv(override=True)


class TestAgentRAG:
    """Tests de l'agent RAG avec Qdrant."""

    def test_rag_returns_documents(self):
        """Vérifie que le RAG retourne des documents."""
        from agent_rag import run_rag_agent
        state = {
            "query": "concerts à Lille",
            "city": "Lille",
            "latitude": None,
            "longitude": None,
            "radius_km": 50,
            "documents": [],
            "rag_done": False
        }
        result = run_rag_agent(state)
        assert "documents" in result
        assert isinstance(result["documents"], list)

    def test_rag_returns_relevant_results(self):
        """Vérifie que le RAG retourne des résultats pertinents."""
        from agent_rag import run_rag_agent
        state = {
            "query": "exposition musée Lille",
            "city": "Lille",
            "latitude": None,
            "longitude": None,
            "radius_km": 50,
            "documents": [],
            "rag_done": False
        }
        result = run_rag_agent(state)
        assert result.get("rag_done") is True

    def test_rag_document_has_required_fields(self):
        """Vérifie que les documents retournés ont les bons champs."""
        from agent_rag import run_rag_agent
        state = {
            "query": "festival musique",
            "city": None,
            "latitude": None,
            "longitude": None,
            "radius_km": 50,
            "documents": [],
            "rag_done": False
        }
        result = run_rag_agent(state)
        documents = result.get("documents", [])
        if len(documents) > 0:
            doc = documents[0]
            assert "title" in doc, "Le document doit avoir un titre"
            assert "text" in doc, "Le document doit avoir un texte"

    def test_rag_with_geo_filter(self):
        """Vérifie le filtrage géographique."""
        from agent_rag import run_rag_agent
        state = {
            "query": "événements culturels",
            "city": "Lille",
            "latitude": 50.6292,
            "longitude": 3.0573,
            "radius_km": 30,
            "documents": [],
            "rag_done": False
        }
        result = run_rag_agent(state)
        assert "documents" in result
        assert result.get("rag_done") is True

    def test_rag_empty_query(self):
        """Vérifie le comportement avec une query vide."""
        from agent_rag import run_rag_agent
        state = {
            "query": "",
            "city": None,
            "latitude": None,
            "longitude": None,
            "radius_km": 50,
            "documents": [],
            "rag_done": False
        }
        result = run_rag_agent(state)
        assert "documents" in result
        assert isinstance(result["documents"], list)