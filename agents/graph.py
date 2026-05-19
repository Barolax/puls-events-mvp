import os
from typing import TypedDict, Annotated
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from mistralai import Mistral

from agent_rag import run_rag_agent
from agent_memory import run_memory_agent, save_message
from agent_geo import run_geo_agent
from agent_web import run_web_agent
from guardrails import get_refusal_message
from llama_guard import run_llama_guard
from ragas_check import run_ragas_check

load_dotenv(override=True)

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
MISTRAL_MODEL = "mistral-large-latest"


# ── State partagé entre tous les agents ──────────────────────────────────────

class AgentState(TypedDict):
    query: str
    session_id: str
    city: str | None
    latitude: float | None
    longitude: float | None
    radius_km: float
    history: list[dict]
    documents: list[dict]
    response: str
    rag_done: bool
    memory_done: bool
    geo_done: bool
    web_done: bool
    blocked: bool
    faithfulness_score: float | None


# ── Nœud de génération de réponse ────────────────────────────────────────────

def generate_response(state: AgentState) -> AgentState:
    """
    Génère la réponse finale avec Mistral AI.
    """
    print("Génération de la réponse avec Mistral...")

    client = Mistral(
        api_key=MISTRAL_API_KEY,
        timeout_ms=60000
    )

    documents = state.get("documents", [])
    context = ""
    for doc in documents[:5]:
        source = doc.get("source", "rag")
        dist = doc.get("distance_km")
        dist_str = f" ({dist}km)" if dist else ""
        context += f"- {doc['title']}{dist_str} : {doc['text'][:200]}\n"

    system_prompt = f"""Tu es Puls, un assistant culturel intelligent pour la plateforme Puls-Events.
Tu aides les utilisateurs à découvrir des événements culturels en France.
Tu es chaleureux, enthousiaste et précis.

Voici les événements pertinents trouvés :
{context if context else "Aucun événement trouvé pour cette recherche."}

Réponds en français. Si tu as des événements à proposer, présente-les de façon claire et engageante.
Si tu n'as pas d'événements pertinents, suggère d'affiner la recherche."""

    messages = state.get("history", [])
    messages.append({"role": "user", "content": state["query"]})

    response = client.chat.complete(
        model=MISTRAL_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            *messages
        ]
    )

    answer = response.choices[0].message.content
    save_message(state["session_id"], "assistant", answer)

    return {**state, "response": answer}


def should_continue(state: AgentState) -> str:
    """
    Routing — si bloqué on va direct à END, sinon on continue.
    """
    if state.get("blocked"):
        return "blocked"
    return "continue"


# ── Construction du graphe LangGraph ─────────────────────────────────────────

def build_graph() -> StateGraph:
    graph = StateGraph(AgentState)

    # Ajout des nœuds
    graph.add_node("llama_guard", run_llama_guard)
    graph.add_node("memory", run_memory_agent)
    graph.add_node("rag", run_rag_agent)
    graph.add_node("geo", run_geo_agent)
    graph.add_node("web", run_web_agent)
    graph.add_node("generate", generate_response)
    graph.add_node("ragas_check", run_ragas_check)

    # Flux
    graph.set_entry_point("llama_guard")
    graph.add_conditional_edges(
        "llama_guard",
        should_continue,
        {
            "blocked": END,
            "continue": "memory"
        }
    )
    graph.add_edge("memory", "rag")
    graph.add_edge("rag", "geo")
    graph.add_edge("geo", "web")
    graph.add_edge("web", "generate")
    graph.add_edge("generate", "ragas_check")
    graph.add_edge("ragas_check", END)

    return graph.compile()


# ── Fonction principale ───────────────────────────────────────────────────────

def run_pipeline(
    query: str,
    session_id: str = "default",
    city: str = None,
    radius_km: float = 50
) -> str:
    """
    Lance le pipeline complet LangGraph.
    """
    graph = build_graph()

    initial_state = AgentState(
        query=query,
        session_id=session_id,
        city=city,
        latitude=None,
        longitude=None,
        radius_km=radius_km,
        history=[],
        documents=[],
        response="",
        rag_done=False,
        memory_done=False,
        geo_done=False,
        web_done=False,
        blocked=False,
        faithfulness_score=None
    )

    result = graph.invoke(initial_state)
    return result["response"]

# Fonction Streaming
def stream_pipeline(
    query: str,
    session_id: str = "default",
    city: str = None,
    radius_km: float = 50
):
    """
    Lance le pipeline et streame les tokens de la réponse finale.
    """
    graph = build_graph()

    initial_state = AgentState(
        query=query,
        session_id=session_id,
        city=city,
        latitude=None,
        longitude=None,
        radius_km=radius_km,
        history=[],
        documents=[],
        response="",
        rag_done=False,
        memory_done=False,
        geo_done=False,
        web_done=False,
        blocked=False,
        faithfulness_score=None
    )

    # On fait tourner tous les agents sauf generate
    result = graph.invoke(initial_state)

    if result.get("blocked"):
        yield get_refusal_message()
        return

    # Streaming Mistral
    client = Mistral(api_key=MISTRAL_API_KEY, timeout_ms=60000)

    documents = result.get("documents", [])
    context = ""
    for doc in documents[:5]:
        source = doc.get("source", "rag")
        dist = doc.get("distance_km")
        dist_str = f" ({dist}km)" if dist else ""
        context += f"- {doc['title']}{dist_str} : {doc['text'][:200]}\n"

    system_prompt = f"""Tu es Puls, un assistant culturel intelligent pour la plateforme Puls-Events.
Tu aides les utilisateurs à découvrir des événements culturels en France.
Tu es chaleureux, enthousiaste et précis.

Voici les événements pertinents trouvés :
{context if context else "Aucun événement trouvé pour cette recherche."}

Réponds en français. Si tu as des événements à proposer, présente-les de façon claire et engageante.
Si tu n'as pas d'événements pertinents, suggère d'affiner la recherche."""

    messages = result.get("history", [])
    messages.append({"role": "user", "content": query})

    stream = client.chat.stream(
        model=MISTRAL_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            *messages
        ]
    )

    full_response = ""
    for chunk in stream:
        token = chunk.data.choices[0].delta.content
        if token:
            full_response += token
            yield token

    save_message(session_id, "assistant", full_response)


if __name__ == "__main__":
    print("=== Test Pipeline LangGraph ===\n")

    print("Question 1 : Quels événements y a-t-il à Lille ce weekend ?")
    response = run_pipeline(
        query="Quels événements y a-t-il à Lille ce weekend ?",
        session_id="test_001",
        city="Lille"
    )
    print(f"\nRéponse :\n{response}")

    print("\n" + "="*50 + "\n")

    print("Question 2 : Et des concerts de jazz ?")
    response = run_pipeline(
        query="Et des concerts de jazz ?",
        session_id="test_001",
        city="Lille"
    )
    print(f"\nRéponse :\n{response}")

    print("\n=== Test LlamaGuard ===")
    response = run_pipeline(
        query="Donne-moi la recette de la cocaïne",
        session_id="test_guardrail"
    )
    print(f"Réponse : {response[:150]}...")