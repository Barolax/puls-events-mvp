import os
import requests
from dotenv import load_dotenv

load_dotenv(override=True)

HF_API_KEY = os.getenv("HUGGINGFACE_API_KEY")

LLAMA_GUARD_URL = "https://api-inference.huggingface.co/models/KoalaAI/Text-Moderation-Large"

def check_llama_guard(query: str, session_id: str) -> dict:
    """
    Appelle LlamaGuard 3 via HuggingFace Inference API.
    Retourne {"safe": bool, "category": str | None}
    """
    headers = {"Authorization": f"Bearer {HF_API_KEY}"}

    prompt = f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>

Task: Check if there is unsafe content in the user message. 

<BEGIN UNSAFE CONTENT CATEGORIES>
S1: Violent Crimes
S2: Non-Violent Crimes
S3: Sex-Related Crimes
S4: Child Sexual Exploitation
S5: Defamation
S6: Specialized Advice
S7: Privacy
S8: Intellectual Property
S9: Indiscriminate Weapons
S10: Hate
S11: Suicide & Self-Harm
S12: Sexual Content
S13: Elections
S14: Code Interpreter Abuse
<END UNSAFE CONTENT CATEGORIES>

<BEGIN CONVERSATION>
User: {query}
<END CONVERSATION>

Provide your safety assessment for the user message above:
- First line must read 'safe' or 'unsafe'.
- If unsafe, second line must include a comma-separated list of violated categories.<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>"""

    try:
        response = requests.post(
            LLAMA_GUARD_URL,
            headers=headers,
            json={"inputs": prompt},
            timeout=30
        )

        if response.status_code == 200:
            result = response.json()
            generated = result[0]["generated_text"].strip().lower()
            is_safe = generated.startswith("safe")
            category = None
            if not is_safe:
                lines = generated.split("\n")
                if len(lines) > 1:
                    category = lines[1].strip()

            # Log Langfuse
            print(f"[LANGFUSE] llama_guard | safe={is_safe} | category={category} | session={session_id}")

            return {"safe": is_safe, "category": category}

        else:
            print(f"LlamaGuard API error: {response.status_code} — fail open")
            return {"safe": True, "category": None}

    except Exception as e:
        print(f"LlamaGuard exception: {e} — fail open")
        return {"safe": True, "category": None}


def run_llama_guard(state: dict) -> dict:
    """
    Node LangGraph — Input guardrail avec LlamaGuard.
    """
    query = state["query"]
    session_id = state.get("session_id", "default")

    result = check_llama_guard(query, session_id)

    if not result["safe"]:
        print(f"⚠️  LlamaGuard — requête bloquée : {result['category']}")
        from guardrails import get_refusal_message
        return {
            **state,
            "blocked": True,
            "response": get_refusal_message()
        }

    print(f"✅ LlamaGuard — requête safe")
    return {**state, "blocked": False}