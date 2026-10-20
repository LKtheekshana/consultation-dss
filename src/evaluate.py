"""Stage 7: Evaluation.

Brings together the evaluation:

    A. Generation (hold-out): each generated submission is compared with what
       the respondent actually wrote (ROUGE-1, ROUGE-L, BLEU, BERTScore), for
       zero-shot and few-shot prompts. A control row compares generated texts
       with an UNRELATED respondent's submission, to show what the scores look
       like when there is no real match.
    B. Summaries: FLAN-T5 and Llama summaries are compared with reference
       summaries written by people (only once the references exist).
    C. Invented names (src/factcheck.py) and response times per function.
    D. Human ratings: makes a blinded rating sheet; once raters have filled
       it in, reports mean readability, informativeness and flow per system
       and the agreement between raters (Krippendorff's alpha).
    E. System Usability Scale: makes the form; once filled in, scores it.
       Also reports how much users edited the AI drafts in the web app.
    F. The success criteria of the study: met, not met or pending.

Parts that need people (B, D, E) show "pending" until their files exist, so
this script can be run again whenever new ratings or references arrive.

Run from the project folder (after Stages 1-5):
    python src/evaluate.py

Outputs (in data/processed/):
    evaluation_report.txt     all evaluation tables
    generation_scores.csv     scores for every generated submission
    summary_scores.csv        scores for every summary (once references exist)
Forms for people (in data/raw/, created once and never overwritten):
    human_ratings/rating_sheet_TEMPLATE.csv, human_ratings/rating_key.csv
    sus_responses_TEMPLATE.csv, sus_form.txt
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402
from factcheck import unsupported_names  # noqa: E402
from summarize import format_responses, summary_input  # noqa: E402

OUT = config.PROCESSED_DIR
METRICS = ["rouge1", "rougeL", "bleu", "bertscore_p", "bertscore_r", "bertscore_f1"]
METRIC_NAMES = {"rouge1": "ROUGE-1", "rougeL": "ROUGE-L", "bleu": "BLEU",
                "bertscore_p": "BERTScore P", "bertscore_r": "BERTScore R",
                "bertscore_f1": "BERTScore F1"}

# The ten SUS statements (Brooke, 1996), answered from 1 (strongly disagree)
# to 5 (strongly agree)
SUS_ITEMS = [
    "I think that I would like to use this system frequently.",
    "I found the system unnecessarily complex.",
    "I thought the system was easy to use.",
    "I think that I would need the support of a technical person to be able to use this system.",
    "I found the various functions in this system were well integrated.",
    "I thought there was too much inconsistency in this system.",
    "I would imagine that most people would learn to use this system very quickly.",
    "I found the system very cumbersome to use.",
    "I felt very confident using the system.",
    "I needed to learn a lot of things before I could get going with this system.",
]


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig", dtype=str, keep_default_na=False)


# ---------------------------------------------------------------------------
# Text similarity metrics
# ---------------------------------------------------------------------------
def load_bertscore() -> Callable:
    """Return a function (candidates, references) -> (P, R, F1) lists.
    Imported here so the tests can pass a stand-in instead."""
    from bert_score import score

    def bertscore(cands: list[str], refs: list[list[str]]):
        p, r, f = score(cands, refs, lang=config.BERTSCORE_LANG,
                        rescale_with_baseline=config.BERTSCORE_RESCALE, verbose=False)
        return p.tolist(), r.tolist(), f.tolist()
    return bertscore


def score_texts(cands: list[str], refs: list[list[str]], bertscore: Callable) -> pd.DataFrame:
    """Scores of each candidate against its reference(s).

    With several references (e.g. two reference summaries), ROUGE and
    BERTScore use the best-matching reference and BLEU uses all of them,
    which is how these metrics are normally used with multiple references.
    BLEU is on sacreBLEU's 0-100 scale; the other metrics are 0-1.
    """
    from rouge_score import rouge_scorer
    import sacrebleu

    scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
    rows = []
    for cand, rs in zip(cands, refs):
        s = [scorer.score(ref, cand) for ref in rs]
        rows.append({"rouge1": max(x["rouge1"].fmeasure for x in s),
                     "rougeL": max(x["rougeL"].fmeasure for x in s),
                     "bleu": sacrebleu.sentence_bleu(cand, rs).score})
    out = pd.DataFrame(rows)
    p, r, f = bertscore(cands, refs)
    out["bertscore_p"], out["bertscore_r"], out["bertscore_f1"] = p, r, f
    return out


def wilcoxon_text(a: pd.Series, b: pd.Series) -> str:
    """Wilcoxon signed-rank test for paired scores, as text for the report."""
    from scipy.stats import wilcoxon
    diff = a.to_numpy() - b.to_numpy()
    if np.allclose(diff, 0):
        return "no difference"
    stat, p = wilcoxon(a, b)
    return f"W = {stat:.1f}, " + ("p < .001" if p < 0.001 else f"p = {p:.3f}") + f" (n = {len(a)})"


def mean_sd(values: pd.Series, digits: int = 3) -> str:
    return f"{values.mean():.{digits}f} ({values.std():.{digits}f})"


# ---------------------------------------------------------------------------
# A. Generation
# ---------------------------------------------------------------------------
def control_pairs(ids: list[str], seed: int = config.RANDOM_SEED) -> dict[str, str]:
    """Pair every respondent with a different one (a shuffled ring), so each
    generated text can be compared with an unrelated real submission."""
    order = list(np.random.default_rng(seed).permutation(ids))
    return {order[i]: order[(i + 1) % len(order)] for i in range(len(order))}


def evaluate_generation(gen: pd.DataFrame, bertscore: Callable) -> pd.DataFrame:
    """One row per generated text (zero-shot, few-shot) plus control rows."""
    real = gen.drop_duplicates("query_id").set_index("query_id")["real_text"]
    parts = []
    for ptype in ["zero-shot", "few-shot"]:
        g = gen[gen["prompt_type"] == ptype].reset_index(drop=True)
        scores = score_texts(g["generated_text"].tolist(), [[t] for t in g["real_text"]], bertscore)
        parts.append(pd.concat([g[["query_id", "issue_area", "keywords"]].assign(comparison=ptype),
                                scores], axis=1))
    # Control: few-shot text vs an unrelated respondent's real submission
    few = gen[gen["prompt_type"] == "few-shot"].reset_index(drop=True)
    other = control_pairs(few["query_id"].tolist())
    refs = [[real[other[q]]] for q in few["query_id"]]
    scores = score_texts(few["generated_text"].tolist(), refs, bertscore)
    parts.append(pd.concat([few[["query_id", "issue_area", "keywords"]]
                            .assign(comparison="control (unrelated submission)"), scores], axis=1))
    return pd.concat(parts, ignore_index=True)


def generation_table(scores: pd.DataFrame) -> list[str]:
    lines = [f"{'Comparison':<32}{'N':>4}" + "".join(f"{METRIC_NAMES[m]:>15}" for m in METRICS)]
    for comp, g in scores.groupby("comparison", sort=False):
        digits = {m: 1 if m == "bleu" else 3 for m in METRICS}
        lines.append(f"{comp:<32}{len(g):>4}" + "".join(f"{mean_sd(g[m], digits[m]):>15}" for m in METRICS))
    zero = scores[scores["comparison"] == "zero-shot"].set_index("query_id")
    few = scores[scores["comparison"] == "few-shot"].set_index("query_id").loc[zero.index]
    lines.append("")
    lines.append("Few-shot vs zero-shot (Wilcoxon signed-rank, paired by respondent):")
    for m in METRICS:
        lines.append(f"    {METRIC_NAMES[m]:<14} {wilcoxon_text(few[m], zero[m])}")
    return lines


# ---------------------------------------------------------------------------
# B. Summaries
# ---------------------------------------------------------------------------
def load_references(path: Path = config.REFERENCE_SUMMARIES_FILE) -> pd.DataFrame | None:
    """Filled-in reference summaries, or None if there are none yet."""
    if not path.exists():
        return None
    refs = read_csv(path)
    refs = refs[refs["reference_summary"].str.strip() != ""]
    return refs if len(refs) else None


def evaluate_summaries(systems: dict[str, pd.DataFrame], refs: pd.DataFrame,
                       bertscore: Callable) -> pd.DataFrame:
    by_query = refs.groupby("query")["reference_summary"].apply(list)
    parts = []
    for name, s in systems.items():
        s = s[s["query"].isin(by_query.index)].reset_index(drop=True)
        scores = score_texts(s["summary"].tolist(), [by_query[q] for q in s["query"]], bertscore)
        parts.append(pd.concat([s[["query"]].assign(system=name,
                                                    n_references=[len(by_query[q]) for q in s["query"]]),
                                scores], axis=1))
    return pd.concat(parts, ignore_index=True)


# ---------------------------------------------------------------------------
# C. Invented names and response times
# ---------------------------------------------------------------------------
def invented_names(systems: dict[str, pd.DataFrame], clean: pd.DataFrame) -> list[str]:
    lines = []
    for name, s in systems.items():
        flagged = []
        for _, r in s.iterrows():
            sources = [summary_input(clean.loc[i]) for i in r["retrieved_ids"].split("; ") if i]
            names = unsupported_names(r["summary"], sources)
            if names:
                flagged.append(f"'{r['query']}': {', '.join(names)}")
        lines.append(f"    {name:<10} {len(flagged)} of {len(s)} summaries "
                     f"({len(flagged) / max(len(s), 1):.0%})")
        lines += [f"        {f}" for f in flagged]
    return lines


def response_times(gen: pd.DataFrame, systems: dict[str, pd.DataFrame], retrieval_ms: float | None) -> list[str]:
    lines = [f"{'Function':<44}{'N':>5}{'Mean (s)':>10}{'SD':>8}{'Max':>8}"]
    if retrieval_ms is not None:
        lines.append(f"{'Retrieval (embed keywords + find matches)':<44}{'':>5}{retrieval_ms / 1000:>10.3f}")
    for name, s in systems.items():
        t = pd.to_numeric(s["seconds"], errors="coerce").dropna()
        lines.append(f"{'Summary: ' + name:<44}{len(t):>5}{t.mean():>10.2f}{t.std():>8.2f}{t.max():>8.2f}")
    for ptype in ["zero-shot", "few-shot"]:
        t = pd.to_numeric(gen.loc[gen["prompt_type"] == ptype, "seconds"], errors="coerce").dropna()
        lines.append(f"{'Generation: ' + ptype:<44}{len(t):>5}{t.mean():>10.2f}{t.std():>8.2f}{t.max():>8.2f}")
    return lines


def measure_retrieval_ms(clean: pd.DataFrame) -> float:
    """Average time to embed a keyword query and retrieve, over all respondents' keywords."""
    from embed import embed_query, load_embeddings, load_model
    from retrieve import retrieve
    store = load_embeddings()
    settings = json.loads(config.RETRIEVAL_SETTINGS_FILE.read_text(encoding="utf-8"))
    model = load_model(store["model_name"])
    embed_query(model, "warm up")                       # first call is slower
    times = []
    for kw in clean["keywords"]:
        start = time.perf_counter()
        retrieve(embed_query(model, kw), store, settings)
        times.append((time.perf_counter() - start) * 1000)
    return float(np.mean(times))


# ---------------------------------------------------------------------------
# D. Human ratings
# ---------------------------------------------------------------------------
def make_rating_sheet(gen: pd.DataFrame, systems: dict[str, pd.DataFrame], clean: pd.DataFrame,
                      folder: Path = config.HUMAN_RATINGS_DIR) -> bool:
    """Create the blinded rating sheet and its key once. Returns True if created.

    Raters see the input (the submissions, or the keywords) and the text, but
    not which system wrote it; items are shuffled. The key that links items to
    systems is kept separately for the researcher.
    """
    key_file = folder / "rating_key.csv"
    if key_file.exists():
        return False                       # never overwrite once raters may have started
    items = []
    for name, s in systems.items():
        for _, r in s.iterrows():
            ids = [i for i in r["retrieved_ids"].split("; ") if i]
            items.append({"kind": "summary", "system": name, "about": r["query"],
                          "task": "A summary of the citizen submissions shown in 'input'",
                          "input": format_responses([summary_input(clean.loc[i]) for i in ids]),
                          "text": r["summary"]})
    rng = np.random.default_rng(config.RANDOM_SEED)
    respondents = sorted(gen["query_id"].unique())
    chosen = rng.choice(respondents, size=min(config.RATING_SHEET_GENERATIONS, len(respondents)),
                        replace=False)
    for q in sorted(chosen):
        for _, r in gen[gen["query_id"] == q].iterrows():
            items.append({"kind": "generation", "system": f"Llama {r['prompt_type']}", "about": q,
                          "task": "A citizen submission written from the keywords shown in 'input'",
                          "input": f"Keywords: {r['keywords']}", "text": r["generated_text"]})
    order = rng.permutation(len(items))
    items = [items[i] for i in order]
    for n, item in enumerate(items, start=1):
        item["item_id"] = f"I{n:02d}"
    folder.mkdir(parents=True, exist_ok=True)
    sheet = pd.DataFrame([{"item_id": i["item_id"], "task": i["task"], "input": i["input"],
                           "text": i["text"], **{c: "" for c in config.RATING_CRITERIA},
                           "comments": ""} for i in items])
    sheet.to_csv(folder / "rating_sheet_TEMPLATE.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([{k: i[k] for k in ["item_id", "kind", "system", "about"]} for i in items]).to_csv(
        key_file, index=False, encoding="utf-8-sig")
    return True


def load_ratings(folder: Path = config.HUMAN_RATINGS_DIR) -> pd.DataFrame | None:
    """All filled-in rating sheets (files named rating_<rater>.csv), long format:
    item_id, rater, criterion, score. None if there are none yet."""
    rows = []
    for f in sorted(folder.glob("rating_*.csv")):
        if f.name in {"rating_sheet_TEMPLATE.csv", "rating_key.csv"}:
            continue
        sheet = read_csv(f)
        rater = f.stem.replace("rating_", "", 1)
        for _, r in sheet.iterrows():
            for c in config.RATING_CRITERIA:
                score = pd.to_numeric(r.get(c, ""), errors="coerce")
                if 1 <= score <= 5:
                    rows.append({"item_id": r["item_id"], "rater": rater, "criterion": c, "score": score})
    return pd.DataFrame(rows) if rows else None


def krippendorff_alpha(ratings: pd.DataFrame, criterion: str) -> float | None:
    import krippendorff
    sub = ratings[ratings["criterion"] == criterion]
    matrix = sub.pivot_table(index="rater", columns="item_id", values="score")
    if matrix.shape[0] < 2:
        return None
    return float(krippendorff.alpha(reliability_data=matrix.to_numpy(), level_of_measurement="ordinal"))


def ratings_table(ratings: pd.DataFrame, key: pd.DataFrame) -> tuple[list[str], pd.DataFrame]:
    """Mean rating per system and criterion. Returns (report lines, per-system means)."""
    r = ratings.merge(key, on="item_id")
    per_item = r.groupby(["item_id", "kind", "system", "criterion"])["score"].mean().reset_index()
    stats = per_item.groupby(["kind", "system", "criterion"])["score"].agg(["mean", "std"])
    lines = [f"Raters: {ratings['rater'].nunique()} ({', '.join(sorted(ratings['rater'].unique()))}); "
             f"items rated: {ratings['item_id'].nunique()}"]
    lines.append(f"{'Output':<12}{'System':<20}" + "".join(f"{c.capitalize():>18}" for c in config.RATING_CRITERIA)
                 + f"{'Overall':>10}")
    overall = {}
    for kind, system in stats.index.droplevel("criterion").unique():
        row = {c: stats.loc[(kind, system, c)] for c in config.RATING_CRITERIA}
        cells = "".join(f"{row[c]['mean']:>11.2f} ({row[c]['std']:.2f})" for c in config.RATING_CRITERIA)
        ov = np.mean([row[c]["mean"] for c in config.RATING_CRITERIA])
        overall[(kind, system)] = ov
        lines.append(f"{kind:<12}{system:<20}{cells}{ov:>10.2f}")
    lines.append("Values are means (SD) over items, each item averaged over raters; scale 1-5.")
    alphas = {c: krippendorff_alpha(ratings, c) for c in config.RATING_CRITERIA}
    if any(a is not None for a in alphas.values()):
        lines.append("Inter-rater agreement (Krippendorff's alpha, ordinal): "
                     + ", ".join(f"{c} {a:.2f}" for c, a in alphas.items() if a is not None))
    else:
        lines.append("Inter-rater agreement: needs at least two raters.")
    return lines, pd.Series(overall)


def rating_comparisons(ratings: pd.DataFrame, key: pd.DataFrame) -> list[str]:
    """Per-rater overall means and paired tests between systems.

    Each item's score is its mean over raters and criteria. Summaries are
    paired by topic (Llama vs FLAN-T5); drafts by respondent (few- vs zero-shot).
    """
    r = ratings.merge(key, on="item_id")
    per_rater = r.groupby(["kind", "system", "rater"])["score"].mean().unstack("rater")
    lines = ["Overall mean per rater (each rater's own scores, all criteria):",
             f"{'Output':<12}{'System':<20}" + "".join(f"{c:>8}" for c in per_rater.columns)]
    for (kind, system), row in per_rater.iterrows():
        lines.append(f"{kind:<12}{system:<20}" + "".join(f"{v:>8.2f}" for v in row))
    item = r.groupby(["kind", "system", "about"])["score"].mean()
    for kind, a, b, by in [("summary", "Llama", "FLAN-T5", "topic"),
                           ("generation", "Llama few-shot", "Llama zero-shot", "respondent")]:
        x, y = item[(kind, a)], item[(kind, b)]
        x, y = x.loc[x.index.intersection(y.index)], y.loc[x.index.intersection(y.index)]
        lines.append(f"{a} vs {b} (Wilcoxon, paired by {by}): {wilcoxon_text(x, y)}; "
                     f"{a} higher for {(x > y).sum()} of {len(x)}, equal for {(x == y).sum()}")
    return lines


# ---------------------------------------------------------------------------
# E. System Usability Scale and draft edits
# ---------------------------------------------------------------------------
def make_sus_forms() -> bool:
    """Create the SUS form and empty response sheet once."""
    template = config.RAW_DIR / "sus_responses_TEMPLATE.csv"
    if template.exists():
        return False
    pd.DataFrame(columns=["participant"] + [f"q{i}" for i in range(1, 11)]).to_csv(
        template, index=False, encoding="utf-8-sig")
    lines = ["SYSTEM USABILITY SCALE (Brooke, 1996)", "",
             "After using the Consultation DSS web application, please tick one box for each",
             "statement: 1 = strongly disagree ... 5 = strongly agree.", ""]
    for i, item in enumerate(SUS_ITEMS, start=1):
        lines.append(f"{i:>2}. {item}")
        lines.append("    [1]  [2]  [3]  [4]  [5]")
    lines += ["", "Researcher: enter each participant's answers as one row (q1..q10, values 1-5)",
              f"in {config.SUS_FILE.name} (copy {template.name})."]
    (config.RAW_DIR / "sus_form.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return True


def sus_score(answers: list[int]) -> float:
    """Standard SUS score (0-100): odd items score - 1, even items 5 - score,
    sum times 2.5."""
    if len(answers) != 10 or any(a not in (1, 2, 3, 4, 5) for a in answers):
        raise ValueError("SUS needs ten answers from 1 to 5")
    total = sum((a - 1) if i % 2 == 0 else (5 - a) for i, a in enumerate(answers))
    return total * 2.5


def load_sus(path: Path = config.SUS_FILE) -> pd.DataFrame | None:
    if not path.exists():
        return None
    df = read_csv(path)
    rows = []
    for _, r in df.iterrows():
        answers = [int(r[f"q{i}"]) for i in range(1, 11) if str(r[f"q{i}"]).strip()]
        if len(answers) == 10:
            rows.append({"participant": r["participant"], "sus": sus_score(answers)})
    return pd.DataFrame(rows) if rows else None


def draft_edit_lines() -> list[str]:
    """How much web-app users changed the AI drafts before submitting."""
    from submissions import load_submissions
    subs = load_submissions()
    if subs.empty:
        return ["    No submissions in the web app yet (they are recorded during usability testing)."]
    ai = subs[subs["ai_assisted"] == "True"]
    lines = [f"    Submissions: {len(subs)}; AI-assisted: {len(ai)} ({len(ai) / len(subs):.0%})"]
    if len(ai):
        ratio = pd.to_numeric(ai["edit_ratio"], errors="coerce")
        lines.append(f"    AI drafts submitted unchanged: {(ai['edited'] == 'False').mean():.0%}")
        lines.append(f"    Share of the AI draft changed: mean {ratio.mean():.0%}, median {ratio.median():.0%}")
    return lines


# ---------------------------------------------------------------------------
# F. Success criteria
# ---------------------------------------------------------------------------
def retrieval_criterion() -> tuple[str, str]:
    """Best semantic method vs keyword baseline on the issue-area label (Stage 3)."""
    from retrieve import SEMANTIC_METHODS, paired_test, summarize
    results = pd.read_csv(OUT / "retrieval_results.csv", encoding="utf-8-sig")
    table = summarize(results)
    area = table[table["label"] == "area"]
    best = area[area["method"].isin(SEMANTIC_METHODS)].sort_values("f1", ascending=False).iloc[0]
    base = area[area["method"] == "keyword"].iloc[0]
    evidence = (f"{best['method']} F1 {best['f1']:.3f} vs keyword F1 {base['f1']:.3f}; "
                f"{paired_test(results, 'area', best['method'])}")
    return evidence, "Met" if best["f1"] > base["f1"] else "Not met"


def criteria_lines(rating_means: pd.Series | None, sus: pd.DataFrame | None) -> list[str]:
    rows = [("Semantic retrieval outperforms keyword matching (F1)", *retrieval_criterion())]
    if rating_means is None:
        rows.append((f"Mean human rating >= {config.HUMAN_RATING_TARGET}", "no ratings yet", "Pending"))
    else:
        for (kind, system), value in rating_means.items():
            rows.append((f"Mean human rating >= {config.HUMAN_RATING_TARGET}: {kind}, {system}",
                         f"overall mean {value:.2f}",
                         "Met" if value >= config.HUMAN_RATING_TARGET else "Not met"))
    if sus is None:
        rows.append((f"Mean SUS score > {config.SUS_BENCHMARK}", "no SUS responses yet", "Pending"))
    else:
        m = sus["sus"].mean()
        rows.append((f"Mean SUS score > {config.SUS_BENCHMARK}",
                     f"mean {m:.1f} (SD {sus['sus'].std():.1f}), n = {len(sus)}",
                     "Met" if m > config.SUS_BENCHMARK else "Not met"))
    lines = [f"{'Criterion':<62}{'Status':<10}Evidence"]
    lines += [f"{c[:61]:<62}{s:<10}{e}" for c, e, s in rows]
    return lines


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    clean = read_csv(OUT / "clean_responses.csv").set_index("response_id")
    gen = read_csv(OUT / "generations.csv")
    systems = {"FLAN-T5": read_csv(OUT / "summaries.csv"), "Llama": read_csv(OUT / "summaries_llama.csv")}

    print(f"Loading BERTScore ({config.BERTSCORE_LANG}; downloads roberta-large on first run) ...")
    bertscore = load_bertscore()
    lines = ["EVALUATION REPORT", "=" * 100]
    if clean["data_source"].str.contains("SYNTHETIC").any():
        lines += ["WARNING: this input is SYNTHETIC test data. Do not report these numbers."]
    lines += [f"Responses: {len(clean)}; generation model: {config.LLAMA_MODEL}; "
              f"BERTScore: {config.BERTSCORE_LANG}, rescaled = {config.BERTSCORE_RESCALE}", ""]

    # A. Generation
    print("Scoring generated submissions against the real ones ...")
    gen_scores = evaluate_generation(gen, bertscore)
    gen_scores.to_csv(OUT / "generation_scores.csv", index=False, encoding="utf-8-sig")
    lines += ["A. Generated submissions vs the respondent's real submission (hold-out)", "-" * 100]
    lines += generation_table(gen_scores)
    lines += ["Values are mean (SD). ROUGE and BERTScore 0-1; BLEU 0-100. The control row compares",
              "few-shot texts with a different respondent's submission (no real match).", ""]

    # B. Summaries
    lines += ["B. Summaries vs reference summaries", "-" * 100]
    refs = load_references()
    if refs is None:
        lines += [f"    Pending: no reference summaries yet. Copy {OUT.name}/reference_summaries_template.csv",
                  f"    to {config.REFERENCE_SUMMARIES_FILE.relative_to(config.ROOT)} and have independent writers fill it in."]
    else:
        print("Scoring summaries against the reference summaries ...")
        sum_scores = evaluate_summaries(systems, refs, bertscore)
        sum_scores.to_csv(OUT / "summary_scores.csv", index=False, encoding="utf-8-sig")
        writers = ", ".join(sorted(refs["written_by"].replace("", "unnamed").unique()))
        lines.append(f"    References: {len(refs)} from {refs['written_by'].nunique()} writer(s) ({writers})")
        lines.append(f"{'System':<12}{'N':>4}" + "".join(f"{METRIC_NAMES[m]:>15}" for m in METRICS))
        for name, g in sum_scores.groupby("system", sort=False):
            lines.append(f"{name:<12}{len(g):>4}" + "".join(
                f"{mean_sd(g[m], 1 if m == 'bleu' else 3):>15}" for m in METRICS))
        a = sum_scores[sum_scores["system"] == "Llama"].set_index("query")
        b = sum_scores[sum_scores["system"] == "FLAN-T5"].set_index("query").loc[a.index]
        lines.append("Llama vs FLAN-T5 (Wilcoxon, paired by query): "
                     + "; ".join(f"{METRIC_NAMES[m]} {wilcoxon_text(a[m], b[m])}"
                                 for m in ["rouge1", "rougeL", "bertscore_f1"]))
    lines.append("")

    # C. Invented names and response times
    lines += ["C. Summaries mentioning names not found in their source submissions", "-" * 100]
    lines += invented_names(systems, clean)
    lines += ["", "   Response times", "-" * 100]
    print("Timing retrieval ...")
    lines += response_times(gen, systems, measure_retrieval_ms(clean.reset_index()))
    lines.append("Summary and generation times were recorded when Stages 4 and 5 ran.")
    lines.append("")

    # D. Human ratings
    lines += ["D. Human ratings (readability, informativeness, flow; 1-5)", "-" * 100]
    if make_rating_sheet(gen, systems, clean):
        lines.append(f"    Rating sheet created: {config.HUMAN_RATINGS_DIR.relative_to(config.ROOT)}/rating_sheet_TEMPLATE.csv")
    ratings = load_ratings()
    rating_means = None
    if ratings is None:
        lines.append("    Pending: no filled-in rating sheets yet (save each as human_ratings/rating_<name>.csv).")
    else:
        key = read_csv(config.HUMAN_RATINGS_DIR / "rating_key.csv")
        table_lines, rating_means = ratings_table(ratings, key)
        lines += table_lines
        lines += rating_comparisons(ratings, key)
    lines.append("")

    # E. SUS and edits
    lines += ["E. Usability: System Usability Scale and edits to AI drafts", "-" * 100]
    if make_sus_forms():
        lines.append("    SUS form created: data/raw/sus_form.txt (and sus_responses_TEMPLATE.csv)")
    sus = load_sus()
    if sus is None:
        lines.append("    Pending: no SUS responses yet.")
    else:
        lines.append(f"    SUS: n = {len(sus)}, mean {sus['sus'].mean():.1f}, SD {sus['sus'].std():.1f}, "
                     f"min {sus['sus'].min():.1f}, max {sus['sus'].max():.1f} (benchmark {config.SUS_BENCHMARK})")
        if len(sus) > 1:
            from scipy import stats
            half = stats.t.ppf(0.975, len(sus) - 1) * sus["sus"].std() / np.sqrt(len(sus))
            lines.append(f"    95% confidence interval of the mean: {sus['sus'].mean() - half:.1f} to "
                         f"{sus['sus'].mean() + half:.1f}")
    lines += draft_edit_lines()
    lines.append("")

    # F. Success criteria
    lines += ["F. Success criteria", "-" * 100]
    lines += criteria_lines(rating_means, sus)

    report = "\n".join(lines)
    (OUT / "evaluation_report.txt").write_text(report, encoding="utf-8")
    print(report)
    print(f"\nSaved outputs to {OUT}")


if __name__ == "__main__":
    main()
