"""Citizen portal: search, summarize, and draft + submit a consultation submission.

Uses only the configuration that performed best in the evaluation:
the retrieval method saved by Stage 3 (dynamic threshold), Llama summaries and
few-shot drafts. FLAN-T5 is used only as an offline backup when Llama cannot be
reached. The other methods remain in the evaluation scripts (src/retrieve.py,
src/summarize.py, src/generate.py, src/evaluate.py)."""
from __future__ import annotations

import time

import streamlit as st

import config
from generate import choose_examples, generate_submission, summarize_with_llama
from llm import LlamaError, load_api_key
from retrieve import retrieve
from embed import embed_query
from submissions import build_record, check_submission, receipt_text, save_submission
from factcheck import unsupported_names
from summarize import summarize_texts, summary_input
from webapp import get_embedder, get_summarizer, llama_chat, load_data

# The three steps of the citizen flow, shown as numbered tabs. The labels are
# fixed (no counts) so the active step can be set from Session State.
STEPS = ["1. See related submissions", "2. Read a summary", "3. Write and submit"]


def go_to(step: str) -> None:
    """Switch to another step (used by the Next buttons)."""
    st.session_state["step"] = step


def next_button(step: str, label: str) -> None:
    st.button(label, icon=":material/arrow_forward:", on_click=go_to, args=(step,),
              key=f"next_{step}")


def clear_draft() -> None:
    """Empty the draft form. The fields are set to empty values rather than
    removed from session state: the text boxes keep their value across page
    switches (persist_state="session"), so a removed value would come back."""
    st.session_state["draft_problem"] = ""
    st.session_state["draft_suggestion"] = ""
    st.session_state["draft_area"] = None
    st.session_state["draft_confirm"] = False
    st.session_state["draft_area_auto"] = True     # pre-fill the area again
    st.session_state.pop("ai_draft", None)


st.title("Consultation Decision Support System")
st.caption("Tell the government about a problem with a public service. Start with a few "
           "keywords, see what other citizens have said, and send your own submission.")

try:
    clean, store, settings = load_data()
except FileNotFoundError as e:
    st.error(f"Missing file: {e.filename}. Run Stages 1-3 first "
             "(python src/preprocess.py, src/embed.py, src/retrieve.py).")
    st.stop()

if store.get("synthetic"):
    st.warning("Running on SYNTHETIC test data (pipeline testing only, not study data).")

# ---- Sidebar: information only (the configuration is fixed by the evaluation) ----
with st.sidebar:
    st.caption(f"Submissions in the database: {len(store['response_ids'])}")
    st.caption(f"Summaries and drafts: {config.LLAMA_MODEL}")
    if not load_api_key():
        st.warning("No LLAMA_API_KEY in .env: AI drafts are off and summaries use the "
                   "offline backup model.")

# ---- Keyword search ----
with st.form("search"):
    keywords = st.text_input("Keywords (2-3 words describing the issue)",
                             placeholder="e.g. hospital waiting time")
    searched = st.form_submit_button("Search", type="primary", icon=":material/search:")

if searched:
    if not keywords.strip():
        st.info("Please enter some keywords.")
        st.stop()
    st.session_state["keywords"] = keywords
    st.session_state["query_vector"] = embed_query(get_embedder(store["model_name"]), keywords)
    # A new search starts a new topic: clear the old summary and draft
    for k in ["summary", "submit_errors", "last_submission"]:
        st.session_state.pop(k, None)
    clear_draft()
    st.session_state["step"] = STEPS[0]           # a new search starts again at step 1

if "query_vector" not in st.session_state:
    # Before the first search: explain the three steps
    with st.container(border=True):
        st.markdown("**How it works**")
        st.markdown(f"""
- :material/search: **{STEPS[0]}:** enter 2-3 keywords and press Search to see what other
  citizens have said about the same issue.
- :material/summarize: **{STEPS[1]}:** get a short AI summary of those submissions.
- :material/edit_note: **{STEPS[2]}:** write your own submission, or let the AI draft one for
  you to check and edit, then send it to the consultation office.
""")
    st.stop()

hits = retrieve(st.session_state["query_vector"], store, settings)
key = st.session_state["keywords"]
if st.session_state.get("summary", {}).get("key") != key:
    st.session_state.pop("summary", None)          # new keywords: old summary is stale

st.caption("Go through the steps from left to right. You can return to any step.")
tab_results, tab_summary, tab_draft = st.tabs(STEPS, key="step", on_change="rerun")

with tab_results:
    if not hits:
        st.info("No related submissions were found. Try other keywords.")
    else:
        st.markdown(f"**{len(hits)} related submission(s)** from other citizens for "
                    f"*{st.session_state['keywords']}*, most similar first.")
        st.caption("Similarity shows how close each submission is to your keywords "
                   "(higher is closer).")
    if hits and hits[0][1] < config.WEAK_MATCH_SIMILARITY:
        st.warning(f"No closely matching submissions were found (best similarity "
                   f"{hits[0][1]:.2f}). The submissions below are the closest available but may "
                   "be about other issues. Try more specific keywords, e.g. \"bus fare\" "
                   "instead of \"travel price\".", icon=":material/search_off:")
    for rank, (rid, score) in enumerate(hits, start=1):
        row = clean.loc[rid]
        with st.container(border=True):
            st.markdown(f"**{rank}. {row['issue_area']}** · `{rid}` · similarity **{score:.3f}**")
            st.write(row["problem_description"])
            if row["suggested_improvement"]:
                st.markdown(f"*Suggested improvement:* {row['suggested_improvement']}")
    if hits:
        next_button(STEPS[1], "Next: read a summary")

with tab_summary:
    if not hits:
        st.info("Nothing to summarize.")
        next_button(STEPS[2], "Next: write your submission")
    else:
        n_used = min(len(hits), config.SUMMARY_MAX_RESPONSES)
        st.markdown(f"Press **Summarize** for a short AI summary of the {n_used} most similar "
                    "submission(s): the main problems and what citizens want the government to do. "
                    "This step is optional.")
        if st.button("Summarize", type="primary", icon=":material/summarize:"):
            texts = [summary_input(clean.loc[rid]) for rid, _ in hits]
            start = time.perf_counter()
            backup_reason = None
            if load_api_key():
                try:
                    with st.spinner("Writing summary ..."):
                        summary, info = summarize_with_llama(texts, llama_chat)
                    model = config.LLAMA_MODEL
                except LlamaError as e:
                    backup_reason = str(e)
            else:
                backup_reason = "no Llama API key is set"
            if backup_reason and not config.OFFLINE_SUMMARY_BACKUP:
                st.error(f"Llama is unavailable ({backup_reason}), and the offline backup model "
                         "is switched off on this server. Please try again in a few minutes.",
                         icon=":material/cloud_off:")
            else:
                if backup_reason:
                    # Offline backup: FLAN-T5 runs locally (rated lower in the evaluation)
                    with st.spinner("Llama is unavailable: writing summary with the offline model ..."):
                        generate, count_tokens = get_summarizer()
                        summary, info = summarize_texts(texts, generate, count_tokens)
                    model = config.SUMMARY_MODEL
                info["seconds"] = time.perf_counter() - start
                st.session_state["summary"] = {"key": key, "model": model, "backup": backup_reason,
                                               "text": summary, "info": info,
                                               "ids": [rid for rid, _ in hits[:info["n_used"]]]}

        result = st.session_state.get("summary")
        if result:
            info = result["info"]
            if result.get("backup"):
                st.warning(f"Llama was unavailable ({result['backup']}), so this summary was "
                           "written by the offline backup model, which is less accurate.",
                           icon=":material/cloud_off:")
            st.markdown(f"**AI-generated summary** by {result['model']} (may contain errors: "
                        "check it against the submissions)")
            st.info(result["text"] or "(empty summary)")
            invented = unsupported_names(result["text"],
                                         [summary_input(clean.loc[rid]) for rid in result["ids"]])
            if invented:
                st.warning("This summary mentions " + ", ".join(f"**{n}**" for n in invented)
                           + ", which does not appear in any of the submissions it is based on. "
                             "The model may have invented it: check before relying on the summary.",
                           icon=":material/fact_check:")
            details = [f"Based on {info['n_used']} of {info['n_retrieved']} retrieved submissions "
                       f"({', '.join(result['ids'])})"]
            if "stages" in info:
                details.append(f"{info['input_tokens']} input tokens · {info['stages']} stage(s)")
            details.append(f"{info['seconds']:.1f} s")
            st.caption(" · ".join(details))
            if info.get("truncated"):
                st.warning("At least one submission was longer than the model's input "
                           "limit and was cut off.")
        next_button(STEPS[2], "Next: write your submission")


# ---------------------------------------------------------------------------
# Draft & submit
# ---------------------------------------------------------------------------
def current_record() -> dict:
    """The submission as it is now in the form."""
    return build_record(st.session_state.get("draft_problem", ""),
                        st.session_state.get("draft_suggestion", ""),
                        st.session_state.get("draft_area") or "",
                        st.session_state["keywords"],
                        st.session_state.get("ai_draft"))


def submit() -> None:
    """Runs when "Submit" is clicked, before the page is redrawn, so the form
    can be cleared after a successful submission."""
    record = current_record()
    errors = check_submission(record)
    if not st.session_state.get("draft_confirm"):
        errors.append("Please tick the box to confirm you have checked the text.")
    last = st.session_state.get("last_submission")
    if last and (last["problem"], last["suggestion"]) == (record["problem"], record["suggestion"]):
        errors.append(f"This text was already submitted (reference {last['reference']}).")
    st.session_state["submit_errors"] = errors
    if errors:
        return
    st.session_state["last_submission"] = save_submission(record)
    clear_draft()


with tab_draft:
    last = st.session_state.get("last_submission")
    if last:
        with st.container(border=True):
            st.success(f"Your submission was sent to the consultation office. "
                       f"Reference number: **{last['reference']}**", icon=":material/check_circle:")
            st.download_button("Download your copy", receipt_text(last),
                               file_name=f"{last['reference']}.txt", mime="text/plain",
                               on_click="ignore", icon=":material/download:", key="dl_receipt")
            st.caption("You can write another submission below.")

    st.markdown("Write your submission in the boxes below, or press **Generate AI draft** to let "
                f"the AI write one from your keywords (*{st.session_state['keywords']}*). Then check "
                "and edit the text, tick the confirmation box and press **Submit**.")
    # Few-shot drafts only: they were closer to real submissions in the evaluation
    if st.button("Generate AI draft", icon=":material/auto_awesome:",
                 help="Replaces the text in the boxes below", disabled=not load_api_key()):
        example_ids = choose_examples(st.session_state["query_vector"], store,
                                      config.FEW_SHOT_EXAMPLES)
        try:
            with st.spinner("Writing a draft ..."):
                out = generate_submission(st.session_state["keywords"],
                                          [clean.loc[e] for e in example_ids], llama_chat)
            st.session_state["ai_draft"] = {
                "problem": out["problem"], "suggestion": out["suggestion"],
                "prompt_type": "few-shot", "model": config.LLAMA_MODEL,
                "example_ids": example_ids, "parsed_ok": out["parsed_ok"],
                "info": f"{out['completion_tokens']} tokens · {out['seconds']:.1f} s"}
            # The text boxes are drawn further down, so they can still be filled here
            st.session_state["draft_problem"] = out["problem"]
            st.session_state["draft_suggestion"] = out["suggestion"]
            st.session_state["draft_confirm"] = False
        except LlamaError as e:
            st.error(str(e))

    ai_draft = st.session_state.get("ai_draft")
    if ai_draft:
        note = (f"**AI-generated draft** ({ai_draft['info']}, based on "
                f"{len(ai_draft['example_ids'])} similar submissions: {', '.join(ai_draft['example_ids'])}"
                + "). It may not match your experience: check every sentence and change "
                  "anything that is not true for you.")
        st.info(note, icon=":material/smart_toy:")
        if not ai_draft["parsed_ok"]:
            st.caption("The model did not follow the Problem / Suggested improvement format; "
                       "please check how the text is split below.")

    # Default issue area: the area of the most similar existing submission
    auto_area = st.session_state.pop("draft_area_auto", False) or "draft_area" not in st.session_state
    if auto_area and hits:
        top_area = clean.loc[hits[0][0], "issue_area"]
        st.session_state["draft_area"] = top_area if top_area in config.ISSUE_AREAS else None
    st.session_state.setdefault("draft_area", None)

    with st.container(border=True):
        st.selectbox("Issue area", config.ISSUE_AREAS, key="draft_area",
                     placeholder="Choose the area your issue relates to", persist_state="session")
        st.text_area("Problem", key="draft_problem", height=150, persist_state="session",
                     placeholder="Describe the problem you have experienced, in as much detail "
                                 "as you can.")
        st.text_area("Suggested improvement", key="draft_suggestion", height=110,
                     persist_state="session",
                     placeholder="What should the government do about it?")
        st.caption("Do not include your name, phone number, email address or NIC number.")
        st.checkbox("I have read and checked this text, and it describes my own experience "
                    "and views.", key="draft_confirm")

        for error in st.session_state.get("submit_errors", []):
            st.error(error)

        with st.container(horizontal=True):
            st.button("Submit to consultation office", type="primary", icon=":material/send:",
                      on_click=submit)
            st.download_button("Download a copy", receipt_text(current_record()),
                               file_name="consultation_draft.txt", mime="text/plain",
                               on_click="ignore", icon=":material/download:", key="dl_draft",
                               disabled=not st.session_state.get("draft_problem", "").strip())
        st.caption("Prototype: submissions are saved by this research system and are not sent "
                   "to the Information Center's official systems.")
