"""Stage 4: Summarization.

Summarizes the responses retrieved for a query with FLAN-T5 (abstractive
summarization: the model writes new sentences rather than copying them).

T5 can only read 512 tokens, so:
    1. at most config.SUMMARY_MAX_RESPONSES responses are used, the most
       similar first;
    2. if they still do not fit in one input, they are split into groups
       that fit, each group is summarized, and the group summaries are then
       summarized together (summarizing in stages).
No response is ever cut off silently: if a single response is longer than
the limit it is truncated and this is recorded.

Run from the project folder (after Stage 3):
    python src/summarize.py

Outputs (in data/processed/):
    summaries.csv                     one summary per test query (config.EVAL_QUERIES)
    summarization_report.txt          token counts, stages used, timing and the summaries
    reference_summaries_template.csv  the same retrieved responses, with an empty
                                      column for independent reference summaries (Stage 7)
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Callable

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402

# Small safety margin: special tokens added by the tokenizer (e.g. </s>)
TOKEN_MARGIN = 8


def summary_input(row) -> str:
    """Text of one response as given to the summarizer: the problem and the
    suggestion as plain text. The "Suggested improvement:" label used in
    response_text is left out, because the model tended to copy it."""
    return " ".join(p for p in [row["problem_description"], row["suggested_improvement"]] if p)


def format_responses(texts: list[str]) -> str:
    """One response per line, so the model can see where each one ends."""
    return "\n".join(f"- {t}" for t in texts)


def pack_groups(texts: list[str], count_tokens: Callable[[str], int], budget: int) -> list[list[str]]:
    """Split responses into groups whose formatted text fits in `budget` tokens.
    Order is kept, so the most similar responses come first."""
    groups, current = [], []
    for t in texts:
        if current and count_tokens(format_responses(current + [t])) > budget:
            groups.append(current)
            current = []
        current.append(t)
    if current:
        groups.append(current)
    return groups


def summarize_texts(texts: list[str], generate: Callable[[str], str],
                    count_tokens: Callable[[str], int]) -> tuple[str, dict]:
    """Summarize a list of responses.

    generate(prompt) returns the model's output for one input.
    count_tokens(text) returns the number of model tokens in a text.
    Passing these in keeps this function independent of the model, so the
    tests can use simple stand-ins.

    Returns (summary, info) where info records what was done.
    """
    used = texts[:config.SUMMARY_MAX_RESPONSES]
    info = {"n_retrieved": len(texts), "n_used": len(used), "stages": 0,
            "groups": 0, "input_tokens": 0, "truncated": False}
    if not used:
        return "", info

    limit = config.T5_TOKEN_LIMIT - TOKEN_MARGIN
    budget = limit - count_tokens(config.SUMMARY_PROMPT)
    info["input_tokens"] = count_tokens(format_responses(used))

    groups = pack_groups(used, count_tokens, budget)
    info["groups"] = len(groups)
    info["truncated"] = any(count_tokens(format_responses(g)) > budget for g in groups)

    # Stage 1: one summary per group (just one group if everything fits)
    summaries = [generate(config.SUMMARY_PROMPT + format_responses(g)) for g in groups]
    info["stages"] = 1

    # Stage 2+: combine group summaries until one summary is left
    combine_budget = limit - count_tokens(config.SUMMARY_COMBINE_PROMPT)
    while len(summaries) > 1:
        parts = pack_groups(summaries, count_tokens, combine_budget)
        if len(parts) == len(summaries):
            # Summaries are so long that none can be paired up: merge two at a time
            parts = [summaries[i:i + 2] for i in range(0, len(summaries), 2)]
        summaries = [generate(config.SUMMARY_COMBINE_PROMPT + format_responses(p)) for p in parts]
        info["stages"] += 1
    return summaries[0].strip(), info


# ---------------------------------------------------------------------------
# The real model
# ---------------------------------------------------------------------------
def load_summarizer(name: str = config.SUMMARY_MODEL) -> tuple[Callable, Callable]:
    """Load FLAN-T5 and return (generate, count_tokens) functions for it.
    Imported here so the tests work without downloading the model."""
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(name)
    model = AutoModelForSeq2SeqLM.from_pretrained(name)
    model.eval()

    def count_tokens(text: str) -> int:
        return len(tokenizer(text, add_special_tokens=False)["input_ids"])

    def generate(prompt: str) -> str:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                           max_length=config.T5_TOKEN_LIMIT)
        output = model.generate(**inputs,
                                min_new_tokens=config.SUMMARY_MIN_NEW_TOKENS,
                                max_new_tokens=config.SUMMARY_MAX_NEW_TOKENS,
                                num_beams=config.SUMMARY_NUM_BEAMS,
                                no_repeat_ngram_size=config.SUMMARY_NO_REPEAT_NGRAM,
                                early_stopping=True)
        return tokenizer.decode(output[0], skip_special_tokens=True)

    return generate, count_tokens


def build_report(df: pd.DataFrame, synthetic: bool, method: str) -> str:
    lines = []
    add = lines.append
    add("SUMMARIZATION REPORT")
    add("=" * 70)
    if synthetic:
        add("WARNING: this input is SYNTHETIC test data. Do not report these")
        add("numbers as results; rerun on the real questionnaire export.")
    add(f"Summarization model:  {config.SUMMARY_MODEL}")
    add(f"Retrieval method:     {method} (from retrieval_settings.json)")
    add(f"Max responses used:   {config.SUMMARY_MAX_RESPONSES}")
    add(f"Input limit:          {config.T5_TOKEN_LIMIT} tokens")
    add(f"Decoding:             beam search ({config.SUMMARY_NUM_BEAMS} beams), "
        f"{config.SUMMARY_MIN_NEW_TOKENS}-{config.SUMMARY_MAX_NEW_TOKENS} new tokens, "
        f"no repeated {config.SUMMARY_NO_REPEAT_NGRAM}-grams")
    add("")
    add(f"{'Query':<44}{'Ret.':>5}{'Used':>5}{'Tokens':>8}{'Stages':>7}{'Secs':>6}")
    add("-" * 75)
    for _, r in df.iterrows():
        add(f"{r['query'][:43]:<44}{r['n_retrieved']:>5}{r['n_used']:>5}"
            f"{r['input_tokens']:>8}{r['stages']:>7}{r['seconds']:>6.1f}")
    done = df[df["n_used"] > 0]
    if len(done):
        add("-" * 75)
        add(f"Mean summary length: {done['summary_words'].mean():.1f} words; "
            f"mean time: {done['seconds'].mean():.1f} s per summary")
        add(f"Queries needing more than one stage: {int((done['stages'] > 1).sum())}; "
            f"with a truncated response: {int(done['truncated'].sum())}")
    add("")
    add("Summaries")
    add("-" * 70)
    for _, r in df.iterrows():
        add(f"[{r['query']}]  ({r['retrieved_ids'] or 'nothing retrieved'})")
        add(r["summary"] or "(no summary: nothing was retrieved)")
        add("")
    return "\n".join(lines)


def main() -> None:
    from embed import embed_query, load_embeddings, load_model
    from retrieve import retrieve

    clean = pd.read_csv(config.PROCESSED_DIR / "clean_responses.csv", encoding="utf-8-sig",
                        dtype=str, keep_default_na=False).set_index("response_id")
    store = load_embeddings()
    settings = json.loads(config.RETRIEVAL_SETTINGS_FILE.read_text(encoding="utf-8"))
    synthetic = bool(store.get("synthetic"))

    print(f"Loading {store['model_name']} for retrieval ...")
    embedder = load_model(store["model_name"])
    print(f"Loading {config.SUMMARY_MODEL} (downloads on first run) ...")
    generate, count_tokens = load_summarizer()

    rows, template = [], []
    for query in config.EVAL_QUERIES:
        hits = retrieve(embed_query(embedder, query), store, settings)
        ids = [rid for rid, _ in hits]
        texts = [summary_input(clean.loc[rid]) for rid in ids]
        print(f"Summarizing '{query}' ({len(ids)} responses) ...")
        start = time.perf_counter()
        summary, info = summarize_texts(texts, generate, count_tokens)
        used_ids = ids[:info["n_used"]]
        rows.append({"query": query, "retrieved_ids": "; ".join(used_ids),
                     **info, "seconds": round(time.perf_counter() - start, 1),
                     "summary_words": len(summary.split()), "summary": summary,
                     "model": config.SUMMARY_MODEL})
        # Reference writers see exactly the responses the model summarized,
        # but not the model's summary (so their reference is independent)
        template.append({"query": query, "response_ids": "; ".join(used_ids),
                         "responses": format_responses(texts[:info["n_used"]]),
                         "reference_summary": "", "written_by": ""})

    df = pd.DataFrame(rows)
    out = config.PROCESSED_DIR
    df.to_csv(out / "summaries.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(template).to_csv(out / "reference_summaries_template.csv",
                                  index=False, encoding="utf-8-sig")
    report = build_report(df, synthetic, settings["method"])
    (out / "summarization_report.txt").write_text(report, encoding="utf-8")
    print(report)
    print(f"\nSaved outputs to {out}")


if __name__ == "__main__":
    main()
