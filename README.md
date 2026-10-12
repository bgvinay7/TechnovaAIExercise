# TechNova AI Support Lab

A Streamlit learning project comparing three approaches to an e-commerce customer-support assistant.

- **Part A — Basic LLM:** direct Gemini prompt.
- **Part B — Role-Based Prompting:** Gemini with structured support instructions.
- **Part C — Fine-Tuning:** optional OpenAI fine-tuned model inference, plus a JSONL validator and a separate fine-tuning job script.

## Requirements

Python 3.10 or newer is recommended. Create and activate a virtual environment, then install:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Configure keys

Copy `.env.example` to `.env` and add your Gemini API key:

```dotenv
GEMINI_API_KEY=your_gemini_key
GEMINI_MODEL=gemini-2.5-flash
OPENAI_API_KEY=
OPENAI_FINE_TUNED_MODEL=
OPENAI_BASE_MODEL=gpt-4o-mini
```

Keep `.env` private. Never commit it to Git.

## Launch Streamlit

```powershell
streamlit run streamlit_app.py
```

Part C is OFF by default. For instructor evaluation of Parts A and B, leave it OFF. With Part C OFF, the app does not call OpenAI.

## Validate the fine-tuning dataset

```powershell
python part_c_fine_tuning/validate_dataset.py
```

The included five examples are educational only; provider-specific validation and larger, high-quality datasets may be required.

## Optional OpenAI fine-tuning

Check the current OpenAI dashboard/docs for your account's fine-tuning eligibility and supported base models before proceeding. OpenAI's published fine-tuning best-practices page has stated that the platform is winding down and unavailable to new users, so an API key and prepaid credit do not by themselves guarantee access.

If your account has access, configure `OPENAI_API_KEY`, verify the selected base model is eligible, and run:

```powershell
python part_c_fine_tuning/openai_finetune.py
```

The script validates the data, asks for an explicit `START` confirmation, uploads the JSONL, and requests a fine-tuning job. This may incur charges. It is intentionally separate from the Streamlit UI and is never triggered by the UI toggle.

When the job succeeds, set the returned model ID as `OPENAI_FINE_TUNED_MODEL` in `.env`. Restart Streamlit, enable Part C in the sidebar, and compare responses.

## GitHub safety

`.env` is ignored by `.gitignore`; `.env.example` is safe to commit. Check `git status` before committing.
