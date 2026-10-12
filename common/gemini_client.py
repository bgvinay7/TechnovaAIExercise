import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")


def get_gemini_client():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY is missing. Add it to your .env file."
        )
    return genai.Client(api_key=api_key)


def generate_basic_response(question: str) -> str:
    """Part A: Send the user question without a custom system prompt."""
    client = get_gemini_client()
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=question,
    )
    return response.text or "(The model returned an empty response.)"


def generate_role_based_response(question: str) -> str:
    """Part B: Send the same question with structured support instructions."""
    from google.genai import types

    system_prompt = """
ROLE
You are a senior customer support representative for TechNova.

OBJECTIVE
Help customers resolve e-commerce issues clearly and professionally.

CONTEXT
Customers may ask about damaged products, delivery delays, refunds,
order status, or requests to speak with a human.

RULES
1. Be empathetic, professional, and concise.
2. Never invent company policies, order status, or delivery information.
3. Never promise a refund unless the applicable policy is provided.
4. Ask for an order number when it is needed to investigate an issue.
5. Escalate serious safety incidents to a human support agent.
6. Never claim that a refund or escalation was completed unless confirmed.
7. For a dangerous product, prioritize immediate safety.

PROCESS
Understand the issue, identify safety risks, acknowledge the concern,
give a safe and policy-compliant next step, and ask for missing information
only when necessary. For serious safety incidents, prioritize safety and
urgent human assistance.

OUTPUT FORMAT
Use a brief acknowledgment, a clear next step, and a request for required
information if necessary. Keep the answer concise.
"""
    client = get_gemini_client()
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=question,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=0.2,
        ),
    )
    return response.text or "(The model returned an empty response.)"
