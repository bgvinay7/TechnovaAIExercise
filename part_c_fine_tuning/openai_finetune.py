"""
Optional OpenAI fine-tuning workflow.

IMPORTANT:
- Check your OpenAI account's fine-tuning access and current model eligibility.
- OpenAI documentation has indicated the fine-tuning platform is winding down
  and unavailable to new users. Do not run this until the dashboard/API confirms
  your account can create jobs.
- This script can incur charges. It is intentionally NOT called by Streamlit.
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

from validate_dataset import validate_dataset, DATASET_PATH

load_dotenv()


def main():
    validate_dataset()

    api_key = os.getenv("OPENAI_API_KEY")
    base_model = os.getenv("OPENAI_BASE_MODEL", "gpt-4o-mini")
    if not api_key:
        raise ValueError("Set OPENAI_API_KEY in .env before running.")

    print("This will upload training data and request a fine-tuning job.")
    print("Confirm current account eligibility and costs before proceeding.")
    confirmation = input("Type START to continue: ").strip()
    if confirmation != "START":
        print("Cancelled. No upload or job was created.")
        return

    client = OpenAI(api_key=api_key)
    uploaded = client.files.create(
        file=Path(DATASET_PATH),
        purpose="fine-tune",
    )
    print(f"Uploaded training file: {uploaded.id}")

    job = client.fine_tuning.jobs.create(
        model=base_model,
        training_file=uploaded.id,
    )
    print(f"Fine-tuning job ID: {job.id}")
    print(f"Initial status: {job.status}")
    print("Check job status in the OpenAI dashboard or API.")
    print("When completed, copy the fine_tuned_model ID into .env:")
    print("OPENAI_FINE_TUNED_MODEL=ft:...")


if __name__ == "__main__":
    main()
