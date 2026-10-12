import os
import time
from dotenv import load_dotenv
import streamlit as st

load_dotenv()

from common.gemini_client import (
    generate_basic_response,
    generate_role_based_response,
)

st.set_page_config(
    page_title="TechNova | AI Support Lab",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .block-container {padding-top: 1.7rem; padding-bottom: 2.5rem;}
    .hero {
        padding: 1.5rem 1.7rem; border-radius: 18px;
        background: linear-gradient(120deg, #102a43 0%, #145374 60%, #1d7185 100%);
        color: white; margin-bottom: 1rem;
    }
    .hero h1 {color: white; margin-bottom: .3rem; font-size: 2.15rem;}
    .hero p {color: #e5f4f7; margin-bottom: 0;}
    .small-note {color: #64748b; font-size: .9rem;}
    div[data-testid="stMetric"] {
        background: var(--secondary-background-color);
        border: 1px solid rgba(128,128,128,.18);
        padding: 12px 16px; border-radius: 12px;
    }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.title("⚙️ Experiment controls")
    st.caption("TechNova · Prompting to Fine-Tuning")
    st.divider()
    enable_part_c = st.toggle(
        "Enable Part C — OpenAI",
        value=False,
        help="OFF by default. When disabled, no OpenAI API call is made.",
    )
    st.caption(
        "For instructor evaluation, leave Part C OFF. Parts A and B use Gemini."
    )
    st.divider()
    st.markdown("**Model setup**")
    gemini_model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    st.code(f"Gemini: {gemini_model}", language="text")
    if enable_part_c:
        if os.getenv("OPENAI_FINE_TUNED_MODEL"):
            st.success("Fine-tuned model ID found in .env.")
        else:
            st.warning("Set OPENAI_FINE_TUNED_MODEL after a training job succeeds.")
    else:
        st.info("OpenAI is disabled. No OpenAI request will be sent.")

st.markdown("""
<div class="hero">
  <h1>💬 TechNova AI Support Lab</h1>
  <p>Compare a basic LLM, role-based prompting, and optional fine-tuning
  using the same customer question.</p>
</div>
""", unsafe_allow_html=True)

metric1, metric2, metric3 = st.columns(3)
with metric1:
    st.metric("Part A", "Basic LLM", "Gemini")
with metric2:
    st.metric("Part B", "Role Prompt", "Gemini")
with metric3:
    st.metric("Part C", "Fine-Tuning", "Optional OpenAI")

st.markdown("### Ask a customer-support question")
examples = {
    "Damaged laptop": "My laptop arrived damaged.",
    "Damaged + order number": "My laptop arrived damaged. My order number is TN12345.",
    "Safety incident": "The laptop exploded while charging.",
    "Refund request": "I want a refund immediately.",
    "Human support": "Can I speak to a human?",
    "Custom question": "",
}
choice = st.selectbox("Choose a test case", list(examples.keys()))
default_question = examples[choice]
question = st.text_area(
    "Customer message",
    value=default_question,
    height=105,
    placeholder="Type a customer message here...",
)
run_comparison = st.button("▶ Compare responses", type="primary", use_container_width=True)

if run_comparison:
    if not question.strip():
        st.warning("Please enter a customer question first.")
        st.stop()

    st.session_state["last_question"] = question.strip()
    st.session_state["comparison_results"] = {}

    with st.spinner("Running Part A — Basic Gemini response..."):
        try:
            st.session_state["comparison_results"]["Part A · Basic LLM"] = (
                generate_basic_response(question.strip())
            )
        except Exception as exc:
            st.session_state["comparison_results"]["Part A · Basic LLM"] = (
                f"Request failed: {exc}"
            )

    with st.spinner("Running Part B — Role-based Gemini response..."):
        try:
            st.session_state["comparison_results"]["Part B · Role Prompt"] = (
                generate_role_based_response(question.strip())
            )
        except Exception as exc:
            st.session_state["comparison_results"]["Part B · Role Prompt"] = (
                f"Request failed: {exc}"
            )

    if enable_part_c:
        from common.openai_client import generate_finetuned_response
        with st.spinner("Running Part C — OpenAI fine-tuned model..."):
            try:
                st.session_state["comparison_results"]["Part C · Fine-Tuned"] = (
                    generate_finetuned_response(question.strip())
                )
            except Exception as exc:
                st.session_state["comparison_results"]["Part C · Fine-Tuned"] = (
                    f"Request failed: {exc}"
                )
    else:
        st.session_state["comparison_results"]["Part C · Fine-Tuned"] = (
            "Disabled for this evaluation. Turn on Part C in the sidebar and "
            "configure a successfully trained model ID to test it."
        )

if "comparison_results" in st.session_state:
    st.divider()
    st.markdown("### Response comparison")
    st.markdown(
        f"<div class='small-note'>Customer question: "
        f"<b>{st.session_state.get('last_question', '')}</b></div>",
        unsafe_allow_html=True,
    )
    results = st.session_state["comparison_results"]
    columns = st.columns(3, gap="medium")
    labels = [
        ("Part A · Basic LLM", "🟦", "Direct prompt"),
        ("Part B · Role Prompt", "🟩", "Structured instructions"),
        ("Part C · Fine-Tuned", "🟪", "OpenAI · optional"),
    ]
    for col, (key, emoji, subtitle) in zip(columns, labels):
        with col:
            st.markdown(f"#### {emoji} {key}")
            st.caption(subtitle)
            st.markdown("---")
            st.write(results.get(key, "Not run."))
            if key.startswith("Part C") and not enable_part_c:
                st.caption("OpenAI API was not called.")

    st.divider()
    st.markdown("### Instructor evaluation checklist")
    st.caption("Score the actual responses; do not assume one approach is always best.")
    criteria = [
        "Empathetic tone",
        "Professional and concise",
        "No invented refund or delivery policy",
        "Asks for order number when appropriate",
        "Prioritizes safety and human escalation",
        "Follows the requested output format",
    ]
    with st.form("evaluation_form"):
        scores = {}
        score_cols = st.columns(2)
        for index, criterion in enumerate(criteria):
            with score_cols[index % 2]:
                scores[criterion] = st.select_slider(
                    criterion,
                    options=[1, 2, 3, 4, 5],
                    value=3,
                    key=f"score_{index}",
                )
        notes = st.text_area("Evaluator notes", placeholder="Record observations...")
        submitted = st.form_submit_button("Show evaluation summary")
    if submitted:
        st.markdown("#### Your evaluation scores")
        st.dataframe(
            [{"Criterion": k, "Score (1–5)": v} for k, v in scores.items()],
            use_container_width=True,
            hide_index=True,
        )
        st.write(notes if notes.strip() else "No additional notes.")
        st.download_button(
            "Download evaluation notes",
            data="TechNova AI Support Lab — Evaluation\n\n"
                 + f"Question: {st.session_state.get('last_question', '')}\n\n"
                 + "\n".join(f"{k}: {v}/5" for k, v in scores.items())
                 + f"\n\nNotes: {notes}\n",
            file_name="technova_evaluation.txt",
            mime="text/plain",
        )

with st.expander("What each part demonstrates"):
    st.markdown("""
    - **Part A — Basic LLM:** sends the question directly to Gemini without a custom role prompt.
    - **Part B — Role-Based Prompting:** sends the same question with explicit support instructions.
    - **Part C — Fine-Tuning:** optionally calls a separately trained OpenAI model. This does not start a training job.
    - **Important:** the sample dataset is educational. Real support policies and escalation actions should be verified by trusted systems.
    """)

st.caption("TechNova Institute Exercise · Prompting to Fine-Tuning · Keep API keys out of Git.")
