
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common.gemini_client import generate_response
from part_b_role_prompting.role_prompting import (
    get_support_response,
)


def main():
    question = "My laptop arrived damaged."

    print("CUSTOMER:", question)

    print("\n=== MODEL A: BASIC LLM ===")
    print(generate_response(question))

    print("\n=== MODEL B: ROLE-BASED PROMPT ===")
    print(get_support_response(question))

    print("\n=== MODEL C: FINE-TUNED MODEL ===")
    print(
        "Not configured yet. Run this comparison after a "
        "supported fine-tuning provider returns a trained model ID."
    )


if __name__ == "__main__":
    main()
