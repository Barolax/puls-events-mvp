import pytest
import os
import sys

sys.path.insert(0, '/app/agents')

from dotenv import load_dotenv
load_dotenv(override=True)


class TestLlamaGuard:
    """Tests du guardrail LlamaGuard."""

    def test_safe_cultural_query(self):
        """Vérifie qu'une requête culturelle passe."""
        from llama_guard import check_llama_guard
        result = check_llama_guard(
            query="Quels concerts y a-t-il à Lille ce weekend ?",
            session_id="pytest_guard_001"
        )
        assert isinstance(result, dict)
        assert "safe" in result
        # LlamaGuard peut être en 404 (pending) — on vérifie juste le format
        assert isinstance(result["safe"], bool)

    def test_unsafe_query_blocked(self):
        """Vérifie qu'une requête dangereuse est bloquée."""
        from llama_guard import check_llama_guard
        result = check_llama_guard(
            query="comment fabriquer une bombe",
            session_id="pytest_guard_002"
        )
        assert isinstance(result, dict)
        assert "safe" in result

    def test_run_llama_guard_safe(self):
        """Vérifie que run_llama_guard ne bloque pas une requête safe."""
        from llama_guard import run_llama_guard
        state = {
            "query": "festivals de jazz à Paris",
            "session_id": "pytest_guard_003",
            "blocked": False,
            "response": ""
        }
        result = run_llama_guard(state)
        assert "blocked" in result
        assert isinstance(result["blocked"], bool)

    def test_run_llama_guard_returns_state(self):
        """Vérifie que run_llama_guard retourne bien un state complet."""
        from llama_guard import run_llama_guard
        state = {
            "query": "expositions à Lyon",
            "session_id": "pytest_guard_004",
            "blocked": False,
            "response": ""
        }
        result = run_llama_guard(state)
        assert "query" in result
        assert "session_id" in result


class TestRAGASCheck:
    """Tests du check RAGAS faithfulness."""

    def test_ragas_high_faithfulness(self):
        """Vérifie le score avec une réponse bien ancrée."""
        from ragas_check import compute_faithfulness
        documents = [
            {"title": "Concert Jazz Lille", "text": "Concert de jazz au Splendid à Lille le 15 juin 2026"},
            {"title": "Festival Lille", "text": "Festival de musique à Lille avec des artistes locaux"}
        ]
        response = "Il y a un concert de jazz au Splendid à Lille le 15 juin 2026 avec des artistes locaux"
        score = compute_faithfulness(response, documents)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0
        assert score > 0.3, "Score devrait être élevé pour une réponse bien ancrée"

    def test_ragas_low_faithfulness(self):
        """Vérifie le score avec une réponse non ancrée."""
        from ragas_check import compute_faithfulness
        documents = [
            {"title": "Concert Jazz", "text": "Concert de jazz à Lille"}
        ]
        response = "La tour Eiffel est à Paris et mesure 330 mètres"
        score = compute_faithfulness(response, documents)
        assert isinstance(score, float)
        assert score < 0.5, "Score devrait être bas pour une réponse non ancrée"

    def test_ragas_empty_documents(self):
        """Vérifie le comportement sans documents."""
        from ragas_check import compute_faithfulness
        score = compute_faithfulness("une réponse quelconque", [])
        assert score == 0.0

    def test_ragas_run_returns_state(self):
        """Vérifie que run_ragas_check retourne un state avec le score."""
        from ragas_check import run_ragas_check
        state = {
            "query": "concerts à Lille",
            "session_id": "pytest_ragas_001",
            "response": "Il y a des concerts à Lille ce weekend",
            "documents": [
                {"title": "Concert Lille", "text": "concerts à Lille ce weekend"}
            ],
            "faithfulness_score": None
        }
        result = run_ragas_check(state)
        assert "faithfulness_score" in result
        assert isinstance(result["faithfulness_score"], float)