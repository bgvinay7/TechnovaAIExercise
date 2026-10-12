
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

# Load .env from the project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
MODEL_ID = os.getenv("OPENAI_FINE_TUNED_MODEL", "").strip()


TEST_CASES = [
    {
        "name": "Damaged product",
        "question": "My laptop arrived with a cracked screen. What should I do?",
    },
    {
        "name": "Refund policy",
        "question": "Can you guarantee that I will get a full refund today?",
    },
    {
        "name": "Safety issue",
        "question": "My charger is sparking and smells burnt. Should I keep using it?",
    },
    {
        "name": "Human support",
        "question": (
            "I contacted support twice and nobody solved my issue. "
            "I want to speak to a human agent."
        ),
    },
]


def get_response(client, question):
    response = client.chat.completions.create(
        model=MODEL_ID,
        messages=[
            {"role": "user", "content": question}
        ],
        temperature=0.2,
    )

    return response.choices[0].message.content or ""


def main():
    if not API_KEY:
        print("ERROR: Add OPENAI_API_KEY to your .env file.")
        return

    if not MODEL_ID:
        print("ERROR: Add OPENAI_FINE_TUNED_MODEL to your .env file.")
        print("Use the exact fine-tuned model ID returned by OpenAI.")
        return

    client = OpenAI(api_key=API_KEY)

    print("\nPART C - FINE-TUNED OPENAI MODEL TEST")
    print("Model:", MODEL_ID)

    while True:
        print("\nSelect a test:")
        for index, case in enumerate(TEST_CASES, start=1):
            print(f"{index}. {case['name']}")

        print("5. Enter your own question")
        print("0. Run all test cases")
        print("q. Quit")

        choice = input("\nChoice: ").strip().lower()

        if choice == "q":
            break

        if choice == "0":
            selected = TEST_CASES
        elif choice in {"1", "2", "3", "4"}:
            selected = [TEST_CASES[int(choice) - 1]]
        elif choice == "5":
            question = input("Customer question: ").strip()

            if not question:
                print("Please enter a question.")
                continue

            selected = [
                {"name": "Custom test", "question": question}
            ]
        else:
            print("Invalid selection.")
            continue

        for case in selected:
            print("\n" + "=" * 60)
            print("TEST:", case["name"])
            print("CUSTOMER:", case["question"])

            try:
                answer = get_response(client, case["question"])
                print("\nFINE-TUNED MODEL RESPONSE:")
                print(answer)

            except Exception as error:
                print("\nAPI ERROR:", error)
                print(
                    "Check your API key, model ID, fine-tuning "
                    "job status, and account access."
                )

    print("\nPart C testing completed.")


if __name__ == "__main__":
    main()
