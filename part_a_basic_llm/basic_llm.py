
import sys
from pathlib import Path

# Make the project root importable when this file runs directly.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common.gemini_client import generate_response


def main():
    question = (
        "A customer says their laptop arrived damaged. "
        "Respond to them."
    )

    answer = generate_response(question)

    print("\n--- PART A: BASIC LLM ---")
    print("Customer:", question)
    print("\nAssistant:", answer)


if __name__ == "__main__":
    main()
