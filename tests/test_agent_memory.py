import pytest
import os
import sys

sys.path.insert(0, '/app/agents')

from dotenv import load_dotenv
load_dotenv(override=True)


class TestAgentMemory:
    """Tests de l'agent mémoire Redis."""

    def test_save_and_retrieve_message(self):
        """Vérifie qu'on peut sauvegarder et récupérer un message."""
        from agent_memory import save_message, run_memory_agent
        session_id = "pytest_memory_001"
        save_message(session_id, "user", "Bonjour je cherche des concerts")
        state = {
            "query": "et demain ?",
            "session_id": session_id,
            "history": [],
            "memory_done": False
        }
        result = run_memory_agent(state)
        assert result.get("memory_done") is True
        assert isinstance(result.get("history"), list)

    def test_memory_returns_history(self):
        """Vérifie que la mémoire retourne l'historique."""
        from agent_memory import save_message, run_memory_agent
        session_id = "pytest_memory_002"
        save_message(session_id, "user", "concerts à Lille")
        save_message(session_id, "assistant", "Voici les concerts à Lille")
        state = {
            "query": "et à Paris ?",
            "session_id": session_id,
            "history": [],
            "memory_done": False
        }
        result = run_memory_agent(state)
        assert len(result.get("history", [])) > 0

    def test_memory_empty_session(self):
        """Vérifie le comportement avec une session vide."""
        from agent_memory import run_memory_agent
        state = {
            "query": "test",
            "session_id": "session_inexistante_xyz",
            "history": [],
            "memory_done": False
        }
        result = run_memory_agent(state)
        assert result.get("memory_done") is True
        assert isinstance(result.get("history"), list)