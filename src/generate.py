"""Stage 5: Generation with Llama 3 (and Llama summaries for comparison).

1. Generation: writes a new consultation-style submission from keywords,
   with two kinds of prompt:
       zero-shot  keywords and instructions only
       few-shot   keywords, instructions and config.FEW_SHOT_EXAMPLES real
                  submissions retrieved for those keywords (style examples)
2. Hold-out run (for Stage 7): for every respondent, a submission is
   generated from *their* keywords, with their own response left out of the
   examples, so it can later be compared with what they actually wrote.
3. Llama summaries: the responses that FLAN-T5 summarized in Stage 4 are
   summarized again with Llama, so the two summarizers can be compared on
   exactly the same input.

Only cleaned text is sent to the API: phone, email and NIC numbers were
replaced with placeholders in Stage 1. All outputs are marked AI-generated.

Run from the project folder (after Stage 4, with LLAMA_API_KEY in .env):
    python src/generate.py

Outputs (in data/processed/):
    generations.csv          hold-out generations (zero-shot and few-shot)
    summaries_llama.csv      Llama summaries of the Stage 4 test queries
    generation_report.txt    counts, lengths, tokens, timing and examples
The run can be stopped and started again: finished generations are kept
and skipped.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402
from preprocess import clean_text  # noqa: E402
from summarize import format_responses, summary_input  # noqa: E402

# A chat function takes (messages, temperature, max_tokens) and returns
# {"text", "prompt_tokens", "completion_tokens", "seconds"} (see llm.chat).
ChatFn = Callable[[list, float, int], dict]

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------
low, high = config.GENERATION_LENGTH_WORDS
SYSTEM_GENERATE = f"""You help citizens of Sri Lanka write submissions to a government public consultation.
Write one submission in the voice of a citizen (first person), in plain English, {low} to {high} words in total.
Describe a realistic problem related to the keywords, then suggest a practical improvement.
Do not include names, phone numbers, email addresses, ID numbers or other personal details.
Answer in exactly this format and write nothing else:
Problem: <the problem>
Suggested improvement: <the suggestion>"""

SYSTEM_SUMMARIZE = """You summarize citizen consultation submissions for government officials.
Use only information that is in the submissions. Do not add facts, places, organizations or numbers that are not in them.
Write one paragraph of 40 to 80 words: first the main problems, then what citizens want the government to do."""


def format_example(row) -> str:
    """A real submission written in the same format the model must answer in."""
    text = f"Problem: {row['problem_description']}"
    if row["suggested_improvement"]:
        text += f"\nSuggested improvement: {row['suggested_improvement']}"
    return text


def build_messages(keywords: str, examples: list) -> list[dict]:
    """Messages for one generation. No examples = zero-shot, examples = few-shot."""
    user = ""
    if examples:
        user += ("Here are real submissions from other citizens. Use them only to see the "
                 "style and level of detail. Do not copy them.\n\n")
        for i, ex in enumerate(examples, start=1):
            user += f"Example {i}:\n{format_example(ex)}\n\n"
    user += f"Keywords: {keywords}\nWrite a new submission about these keywords."
    return [{"role": "system", "content": SYSTEM_GENERATE},
            {"role": "user", "content": user}]


def build_summary_messages(texts: list[str]) -> list[dict]:
    return [{"role": "system", "content": SYSTEM_SUMMARIZE},
            {"role": "user", "content": "Submissions:\n" + format_responses(texts)}]


# ---------------------------------------------------------------------------
# Generation and summarization
# ---------------------------------------------------------------------------
PARSE_RE = re.compile(r"Problem:\s*(.*?)\s*Suggested improvement:\s*(.*)", re.DOTALL | re.IGNORECASE)


def parse_submission(text: str) -> tuple[str, str, bool]:
    """Split the model's answer into (problem, suggestion, parsed_ok).
    If the model ignored the format, the whole answer is kept as the problem."""
    match = PARSE_RE.search(text)
    if not match:
        return clean_text(text), "", False
    problem, suggestion = (clean_text(part.replace("\n", " ")) for part in match.groups())
    return problem, suggestion, bool(problem and suggestion)


def generate_submission(keywords: str, examples: list, chat_fn: ChatFn) -> dict:
    """Generate one submission. clean_text() is applied to the answer as a
    safety net, in case the model writes something that looks like a phone
    number, email or ID number."""
    result = chat_fn(build_messages(keywords, examples),
                     config.GENERATION_TEMPERATURE, config.GENERATION_MAX_TOKENS)
    problem, suggestion, ok = parse_submission(result["text"])
    return {"problem": problem, "suggestion": suggestion, "parsed_ok": ok,
            "generated_text": " ".join(p for p in [problem, suggestion] if p),
            "raw_output": result["text"], "prompt_tokens": result["prompt_tokens"],
            "completion_tokens": result["completion_tokens"], "seconds": result["seconds"]}


def summarize_with_llama(texts: list[str], chat_fn: ChatFn) -> tuple[str, dict]:
    """Summarize with Llama. The same cap on the number of responses as
    FLAN-T5 is used, so both models see the same input."""
    used = texts[:config.SUMMARY_MAX_RESPONSES]
    info = {"n_retrieved": len(texts), "n_used": len(used)}
    if not used:
        return "", info
    result = chat_fn(build_summary_messages(used),
                     config.LLAMA_SUMMARY_TEMPERATURE, config.LLAMA_SUMMARY_MAX_TOKENS)
    info.update(prompt_tokens=result["prompt_tokens"],
                completion_tokens=result["completion_tokens"], seconds=result["seconds"])
    return clean_text(result["text"]), info


def choose_examples(query_vector: np.ndarray, store: dict, n: int,
                    exclude_id: str | None = None) -> list[str]:
    """IDs of the n most similar responses, optionally leaving one out.
    Top-n (rather than a threshold) is used so the prompt always gets n examples."""
    scores = store["embeddings"] @ query_vector
    ids = [store["response_ids"][i] for i in np.argsort(-scores)]
    return [rid for rid in ids if rid != exclude_id][:n]


# ---------------------------------------------------------------------------
# Hold-out run
# ---------------------------------------------------------------------------
def run_holdout(clean: pd.DataFrame, store: dict, query_vectors: dict, chat_fn: ChatFn,
                done: pd.DataFrame, save: Callable[[pd.DataFrame], None]) -> pd.DataFrame:
    """Generate zero-shot and few-shot submissions from each respondent's keywords.

    clean:         responses indexed by response_id
    query_vectors: response_id -> embedded keywords
    done:          rows from an earlier run (skipped, so a stopped run can resume)
    save:          called after every respondent, so progress is never lost
    """
    rows = done.to_dict("records")
    finished = {(r["query_id"], r["prompt_type"]) for r in rows}
    ids = [rid for rid in store["response_ids"] if clean.loc[rid, "keywords"].strip()]
    if config.GENERATION_LIMIT:
        ids = ids[:config.GENERATION_LIMIT]

    for n, rid in enumerate(ids, start=1):
        row = clean.loc[rid]
        example_ids = choose_examples(query_vectors[rid], store, config.FEW_SHOT_EXAMPLES, exclude_id=rid)
        for prompt_type, examples in [("zero-shot", []),
                                      ("few-shot", [clean.loc[e] for e in example_ids])]:
            if (rid, prompt_type) in finished:
                continue
            print(f"[{n}/{len(ids)}] {rid} {prompt_type}: '{row['keywords']}'")
            out = generate_submission(row["keywords"], examples, chat_fn)
            same_area = [clean.loc[e, "issue_area"] == row["issue_area"] for e in example_ids]
            rows.append({
                "query_id": rid, "issue_area": row["issue_area"], "keywords": row["keywords"],
                "prompt_type": prompt_type,
                "example_ids": "; ".join(example_ids) if examples else "",
                "examples_same_area": (sum(same_area) / len(same_area)) if examples else np.nan,
                **out,
                "real_text": summary_input(row),
                "model": config.LLAMA_MODEL, "ai_generated": True,
            })
        save(pd.DataFrame(rows))
    return pd.DataFrame(rows)


def build_report(gen: pd.DataFrame, llama_sum: pd.DataFrame, flan_sum: pd.DataFrame,
                 synthetic: bool) -> str:
    lines = []
    add = lines.append
    add("GENERATION REPORT")
    add("=" * 70)
    if synthetic:
        add("WARNING: this input is SYNTHETIC test data. Do not report these")
        add("numbers as results; rerun on the real questionnaire export.")
    add(f"Model:        {config.LLAMA_MODEL} via {config.LLAMA_API_BASE_URL}")
    add(f"Temperature:  {config.GENERATION_TEMPERATURE} (generation), "
        f"{config.LLAMA_SUMMARY_TEMPERATURE} (summaries); seed {config.LLAMA_SEED}")
    add(f"Few-shot examples per prompt: {config.FEW_SHOT_EXAMPLES} (respondent's own response left out)")
    add(f"Requested length: {low}-{high} words")
    add("All texts below are AI-generated.")
    add("")
    add("Hold-out generation")
    add("-" * 70)
    if len(gen):
        # Rows read back from the CSV (resumed runs) may hold text, so convert
        gen = gen.assign(gen_words=gen["generated_text"].str.split().str.len(),
                         real_words=gen["real_text"].str.split().str.len(),
                         parsed_ok=gen["parsed_ok"].astype(str).str.lower() == "true",
                         examples_same_area=pd.to_numeric(gen["examples_same_area"], errors="coerce"))
        add(f"{'Prompt':<11}{'N':>5}{'Format OK':>11}{'Words':>8}{'Real words':>12}"
            f"{'In tok.':>9}{'Out tok.':>10}{'Secs':>7}")
        for ptype, g in gen.groupby("prompt_type", sort=False):
            add(f"{ptype:<11}{len(g):>5}{g['parsed_ok'].mean():>11.0%}{g['gen_words'].mean():>8.1f}"
                f"{g['real_words'].mean():>12.1f}{g['prompt_tokens'].mean():>9.0f}"
                f"{g['completion_tokens'].mean():>10.0f}{g['seconds'].mean():>7.2f}")
        few = gen[gen["prompt_type"] == "few-shot"]
        if len(few):
            add(f"Few-shot examples from the respondent's own issue area: "
                f"{few['examples_same_area'].mean():.0%}")
        add("Similarity to the real submissions (ROUGE, BLEU, BERTScore) is measured in Stage 7.")
        add("")
        first = gen["query_id"].iloc[0]
        add(f"Example ({first}, keywords: '{gen['keywords'].iloc[0]}')")
        for _, r in gen[gen["query_id"] == first].iterrows():
            add(f"  [{r['prompt_type']}] {r['generated_text']}")
        add(f"  [real]      {gen['real_text'].iloc[0]}")
    else:
        add("No generations.")
    add("")
    add("Summaries: FLAN-T5 (Stage 4) vs Llama, same responses")
    add("-" * 70)
    for (_, f), (_, l) in zip(flan_sum.iterrows(), llama_sum.iterrows()):
        add(f"[{f['query']}]  ({f['retrieved_ids']})")
        add(f"  FLAN-T5: {f['summary']}")
        add(f"  Llama:   {l['summary']}")
        add("")
    return "\n".join(lines)


def main() -> None:
    from embed import embed_query, load_embeddings, load_model
    from llm import LlamaError, chat, load_api_key

    api_key = load_api_key()
    if not api_key:
        print("No API key found.\n"
              "1. Create a Cloudflare API token (Workers AI template) and find your account ID\n"
              "2. Copy .env.example to .env (in the project folder)\n"
              "3. Fill in LLAMA_API_KEY and CLOUDFLARE_ACCOUNT_ID and run this script again.")
        sys.exit(1)

    def chat_fn(messages, temperature, max_tokens):
        return chat(messages, temperature, max_tokens, api_key=api_key)

    out = config.PROCESSED_DIR
    clean = pd.read_csv(out / "clean_responses.csv", encoding="utf-8-sig",
                        dtype=str, keep_default_na=False).set_index("response_id")
    store = load_embeddings()
    synthetic = bool(store.get("synthetic"))

    print(f"Loading {store['model_name']} to embed the keywords ...")
    embedder = load_model(store["model_name"])
    query_vectors = {rid: embed_query(embedder, clean.loc[rid, "keywords"])
                     for rid in store["response_ids"] if clean.loc[rid, "keywords"].strip()}

    # Resume an earlier run with the same model
    gen_file = out / "generations.csv"
    done = pd.DataFrame()
    if gen_file.exists():
        done = pd.read_csv(gen_file, encoding="utf-8-sig", keep_default_na=False)
        done = done[done["model"] == config.LLAMA_MODEL]
        if len(done):
            print(f"Resuming: {len(done)} generations already done.")

    def save(df):
        df.to_csv(gen_file, index=False, encoding="utf-8-sig")

    try:
        gen = run_holdout(clean, store, query_vectors, chat_fn, done, save)

        # Llama summaries of exactly the responses FLAN-T5 summarized in Stage 4
        flan = pd.read_csv(out / "summaries.csv", encoding="utf-8-sig", keep_default_na=False)
        llama_rows = []
        for _, f in flan.iterrows():
            ids = [i for i in f["retrieved_ids"].split("; ") if i]
            print(f"Llama summary: '{f['query']}' ({len(ids)} responses)")
            summary, info = summarize_with_llama([summary_input(clean.loc[i]) for i in ids], chat_fn)
            llama_rows.append({"query": f["query"], "retrieved_ids": f["retrieved_ids"], **info,
                               "summary_words": len(summary.split()), "summary": summary,
                               "model": config.LLAMA_MODEL, "ai_generated": True})
    except LlamaError as e:
        print(f"\nStopped: {e}\nFinished generations are saved; run again to continue.")
        sys.exit(1)

    llama_sum = pd.DataFrame(llama_rows)
    llama_sum.to_csv(out / "summaries_llama.csv", index=False, encoding="utf-8-sig")
    report = build_report(gen, llama_sum, flan, synthetic)
    (out / "generation_report.txt").write_text(report, encoding="utf-8")
    print(report)
    print(f"\nSaved outputs to {out}")


if __name__ == "__main__":
    main()
