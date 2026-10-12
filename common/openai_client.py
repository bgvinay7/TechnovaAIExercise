import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


def generate_finetuned_response(question: str) -> str:
    """Optional Part C inference using an already-trained OpenAI model."""
    api_key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("OPENAI_FINE_TUNED_MODEL")

    if not api_key:
        raise ValueError("OPENAI_API_KEY is missing from .env.")
    if not model:
        raise ValueError(
            "OPENAI_FINE_TUNED_MODEL is not configured. Train a model first "
            "and set its returned fine-tuned model ID in .env."
        )

    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": question}],
        temperature=0.2,
    )
    return response.choices[0].message.content or "(Empty response.)"
