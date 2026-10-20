"""Stage 3: Retrieval and similarity thresholds.

Finds the responses most related to a user's keywords, and compares ways of
deciding how many responses to return:

    static     similarity >= a fixed threshold (config.STATIC_THRESHOLD)
    dynamic    similarity >= mean + k*std of that query's scores
    grid       one threshold chosen by searching 0..1 in steps of 0.01
    top-k      the k most similar responses
    keyword    baseline: exact keyword matching (Marjanovic et al., 2024)

How the comparison works (leave-one-out):
    Each respondent's own keywords are used as a test query. Their own
    response is removed from the pool, and the remaining responses are
    "relevant" if they share the respondent's label. Two labels are used:
        area   same issue_area
        issue  at least one issue in common in issues_selected
    Precision, recall and F1 are calculated per query and then averaged.

    The grid-search threshold is chosen on training folds and scored on the
    held-out fold (cross-validation), so it is not tuned on its own test
    queries. Thresholds are never chosen by ROUGE against the retrieved
    responses, which would be circular.

Run from the project folder (after Stage 2):
    python src/retrieve.py

Outputs (in data/processed/):
    retrieval_report.txt     results table
    retrieval_results.csv    one row per query, method and label
    grid_search_curve.csv    precision/recall/F1 at every threshold (for a figure)
    retrieval_settings.json  method and threshold the web app will use
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402
from embed import embed_query, load_embeddings, load_model  # noqa: E402

SEMANTIC_METHODS = ["static", "dynamic", "grid", "top-k"]
ALL_METHODS = SEMANTIC_METHODS + ["keyword"]
LABELS = ["area", "issue"]


# ---------------------------------------------------------------------------
# Selecting responses
# ---------------------------------------------------------------------------
def static_select(scores: np.ndarray, threshold: float) -> np.ndarray:
    return scores >= threshold


def dynamic_threshold(scores: np.ndarray, k: float) -> float:
    """T = mean + k * std of this query's scores, so the cut-off adapts to
    queries that are generally more or less similar to everything."""
    return float(scores.mean() + k * scores.std())


def dynamic_select(scores: np.ndarray, k: float) -> np.ndarray:
    return scores >= dynamic_threshold(scores, k)


def topk_select(scores: np.ndarray, k: int) -> np.ndarray:
    chosen = np.zeros(len(scores), dtype=bool)
    chosen[np.argsort(-scores)[:k]] = True
    return chosen


def query_words(text: str) -> list[str]:
    """Lower-case words of a query, without stopwords, for the keyword baseline."""
    words = re.findall(r"[a-z0-9]+", str(text).lower())
    return [w for w in words if w not in config.STOPWORDS]


def keyword_select(query: str, texts: list[str], mode: str = config.KEYWORD_MATCH_MODE) -> np.ndarray:
    """Baseline: a response is retrieved if it contains the query words as
    whole words ("any" = at least one word, "all" = every word)."""
    words = query_words(query)
    if not words:
        return np.zeros(len(texts), dtype=bool)
    patterns = [re.compile(rf"\b{re.escape(w)}\b") for w in words]
    combine = all if mode == "all" else any
    return np.array([combine(p.search(t.lower()) for p in patterns) for t in texts], dtype=bool)


def retrieve(query_vector: np.ndarray, store: dict, settings: dict) -> list[tuple[str, float]]:
    """Return (response_id, similarity) pairs, most similar first, using the
    method saved in retrieval_settings.json. Used by the web app."""
    scores = store["embeddings"] @ query_vector
    method = settings["method"]
    if method == "static":
        chosen = static_select(scores, settings["static_threshold"])
    elif method == "dynamic":
        chosen = dynamic_select(scores, settings["dynamic_k"])
    elif method == "grid":
        chosen = static_select(scores, settings["grid_threshold"])
    else:
        chosen = topk_select(scores, settings["top_k"])
    order = [i for i in np.argsort(-scores) if chosen[i]]
    return [(store["response_ids"][i], float(scores[i])) for i in order]


# ---------------------------------------------------------------------------
# Relevance labels and metrics
# ---------------------------------------------------------------------------
def split_issues(text: str) -> set[str]:
    """Split issues_selected into separate issues.

    Domains are joined with "; " in Stage 1. Inside one domain, Google Forms
    joins ticked boxes with ", ". Some options contain a comma themselves
    ("Difficulty getting appointments, tests or referrals"), so we only split
    where the comma is followed by a capital letter (every option starts with one).
    """
    parts = []
    for chunk in str(text).split(";"):
        parts.extend(re.split(r",\s+(?=[A-Z])", chunk))
    return {p.strip() for p in parts if p.strip()}


def relevance_matrix(clean: pd.DataFrame, label: str) -> np.ndarray:
    """relevant[i, j] is True if response j is relevant to query i."""
    if label == "area":
        areas = clean["issue_area"].to_numpy()
        return areas[:, None] == areas[None, :]
    issues = [split_issues(t) for t in clean["issues_selected"]]
    n = len(issues)
    rel = np.zeros((n, n), dtype=bool)
    for i in range(n):
        for j in range(n):
            rel[i, j] = bool(issues[i] & issues[j])
    return rel


def prf(retrieved: np.ndarray, relevant: np.ndarray) -> tuple[float, float, float]:
    """Precision, recall and F1 for one query. Retrieving nothing scores 0."""
    hits = int((retrieved & relevant).sum())
    n_ret, n_rel = int(retrieved.sum()), int(relevant.sum())
    p = hits / n_ret if n_ret else 0.0
    r = hits / n_rel if n_rel else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f1


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def thresholds() -> np.ndarray:
    steps = int(round(1 / config.GRID_STEP))
    return np.round(np.linspace(0, 1, steps + 1), 4)


def grid_f1_table(sims: np.ndarray, rel: np.ndarray, queries: list[int]) -> np.ndarray:
    """Mean precision, recall and F1 over the given queries at every grid threshold.
    Returns an array of shape (len(thresholds), 3)."""
    grid = thresholds()
    out = np.zeros((len(grid), 3))
    for qi in queries:
        mask = np.arange(sims.shape[1]) != qi           # leave the query's own response out
        s, r = sims[qi, mask], rel[qi, mask]
        for ti, t in enumerate(grid):
            out[ti] += prf(s >= t, r)
    return out / max(len(queries), 1)


def cv_grid_thresholds(sims: np.ndarray, rel: np.ndarray, queries: list[int]) -> dict[int, float]:
    """Pick a threshold for every query using only the other folds."""
    rng = np.random.default_rng(config.RANDOM_SEED)
    shuffled = rng.permutation(queries)
    folds = np.array_split(shuffled, min(config.GRID_FOLDS, len(queries)))
    grid = thresholds()
    chosen = {}
    for fold in folds:
        held_out = set(fold)
        train = [q for q in queries if q not in held_out]
        best = grid[grid_f1_table(sims, rel, train)[:, 2].argmax()] if train else config.STATIC_THRESHOLD
        for q in fold:
            chosen[int(q)] = float(best)
    return chosen


def evaluate(clean: pd.DataFrame, sims: np.ndarray) -> tuple[pd.DataFrame, dict]:
    """Score every method for every query.

    clean: responses in the same order as the rows/columns of sims.
    sims:  sims[i, j] = similarity between query i's keywords and response j.
    Returns (per-query results, extra information for the report).
    """
    texts = clean["response_text"].tolist()
    rows, info = [], {"skipped": {}, "cv_thresholds": {}}
    has_query = [i for i, k in enumerate(clean["keywords"]) if str(k).strip()]

    for label in LABELS:
        rel = relevance_matrix(clean, label)
        np.fill_diagonal(rel, False)
        # A query with no other relevant response cannot have its recall measured
        queries = [i for i in has_query if rel[i].any()]
        info["skipped"][label] = len(clean) - len(queries)
        cv_t = cv_grid_thresholds(sims, rel, queries)
        info["cv_thresholds"][label] = sorted(set(cv_t.values()))

        for qi in queries:
            mask = np.arange(len(clean)) != qi
            s, r = sims[qi, mask], rel[qi, mask]
            pool_texts = [t for j, t in enumerate(texts) if j != qi]
            selections = {
                "static": static_select(s, config.STATIC_THRESHOLD),
                "dynamic": dynamic_select(s, config.DYNAMIC_K),
                "grid": static_select(s, cv_t[qi]),
                "top-k": topk_select(s, config.TOP_K),
                "keyword": keyword_select(clean["keywords"].iloc[qi], pool_texts),
            }
            for method, chosen in selections.items():
                p, rc, f1 = prf(chosen, r)
                rows.append({"query_id": clean["response_id"].iloc[qi], "label": label,
                             "method": method, "precision": p, "recall": rc, "f1": f1,
                             "n_retrieved": int(chosen.sum()), "n_relevant": int(r.sum())})
    return pd.DataFrame(rows), info


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    """Mean scores per label and method (the results table)."""
    g = results.groupby(["label", "method"], sort=False)
    table = g[["precision", "recall", "f1", "n_retrieved"]].mean()
    table["no_results"] = g["n_retrieved"].apply(lambda x: int((x == 0).sum()))
    table["queries"] = g.size()
    return table.reset_index()


def paired_test(results: pd.DataFrame, label: str, method: str) -> str:
    """Wilcoxon signed-rank test of per-query F1: method vs keyword baseline."""
    try:
        from scipy.stats import wilcoxon
    except ImportError:
        return "scipy not installed, significance test skipped"
    sub = results[results["label"] == label].pivot(index="query_id", columns="method", values="f1")
    diff = sub[method] - sub["keyword"]
    if (diff == 0).all():
        return "no difference in any query"
    stat, p = wilcoxon(sub[method], sub["keyword"])
    p_text = "p < .001" if p < 0.001 else f"p = {p:.3f}"   # APA style
    return f"W = {stat:.1f}, {p_text} (n = {len(sub)})"


def setting_text(method: str, cv_thresholds: list[float]) -> str:
    if method == "static":
        return f"T = {config.STATIC_THRESHOLD:.2f}"
    if method == "dynamic":
        return f"T = mean + {config.DYNAMIC_K:g}*std"
    if method == "grid":
        if len(cv_thresholds) == 1:
            return f"T = {cv_thresholds[0]:.2f} (all folds)"
        return f"T = {min(cv_thresholds):.2f}-{max(cv_thresholds):.2f} (by fold)"
    if method == "top-k":
        return f"k = {config.TOP_K}"
    return f"match {config.KEYWORD_MATCH_MODE} word"


def build_report(table: pd.DataFrame, info: dict, results: pd.DataFrame,
                 best_threshold: dict, chosen_method: str, synthetic: bool,
                 n_responses: int, model_name: str) -> str:
    lines = []
    add = lines.append
    add("RETRIEVAL REPORT")
    add("=" * 78)
    if synthetic:
        add("WARNING: this input is SYNTHETIC test data. Do not report these")
        add("numbers as results; rerun on the real questionnaire export.")
    add(f"Embedding model:   {model_name}")
    add(f"Responses:         {n_responses}")
    add("Queries:           each respondent's own keywords, own response left out")
    add("")
    names = {"area": "Relevant = same issue area",
             "issue": "Relevant = at least one shared issue (issues_selected)"}
    for label in LABELS:
        sub = table[table["label"] == label]
        add(f"{names[label]}  (queries skipped: {info['skipped'][label]})")
        add("-" * 78)
        add(f"{'Method':<9}{'Setting':<30}{'P':>7}{'R':>7}{'F1':>7}{'Mean ret.':>11}{'None ret.':>10}")
        for _, r in sub.iterrows():
            add(f"{r['method']:<9}{setting_text(r['method'], info['cv_thresholds'][label]):<30.30}"
                f"{r['precision']:>7.3f}{r['recall']:>7.3f}{r['f1']:>7.3f}"
                f"{r['n_retrieved']:>11.1f}{r['no_results']:>10d}")
        best = sub[sub["method"].isin(SEMANTIC_METHODS)].sort_values("f1", ascending=False).iloc[0]
        base = sub[sub["method"] == "keyword"].iloc[0]
        verdict = "beats" if best["f1"] > base["f1"] else "does NOT beat"
        add(f"Best semantic method: {best['method']} (F1 {best['f1']:.3f}) {verdict} "
            f"the keyword baseline (F1 {base['f1']:.3f}).")
        add(f"Wilcoxon test, {best['method']} vs keyword (per-query F1): "
            f"{paired_test(results, label, best['method'])}")
        add(f"Grid threshold tuned on all queries (for the app): {best_threshold[label]:.2f}")
        add("")
    add(f"Method saved for the web app: {chosen_method} (best F1, issue-area label)")
    add("Grid search curve: grid_search_curve.csv; per-query scores: retrieval_results.csv")
    return "\n".join(lines)


def main() -> None:
    clean = pd.read_csv(config.PROCESSED_DIR / "clean_responses.csv", encoding="utf-8-sig",
                        dtype=str, keep_default_na=False)
    store = load_embeddings()
    # Only responses that have a vector take part (translated ones are skipped in Stage 2)
    clean = clean.set_index("response_id").loc[store["response_ids"]].reset_index()
    synthetic = bool(store.get("synthetic"))

    print(f"Loading {store['model_name']} to embed the query keywords ...")
    model = load_model(store["model_name"])
    queries = np.vstack([embed_query(model, k) if k.strip() else np.zeros(store["embeddings"].shape[1])
                         for k in clean["keywords"]])
    sims = queries @ store["embeddings"].T

    results, info = evaluate(clean, sims)
    table = summarize(results)

    # Curve over all queries (for a figure) and the overall best threshold for the app
    curves, best_threshold = [], {}
    for label in LABELS:
        rel = relevance_matrix(clean, label)
        np.fill_diagonal(rel, False)
        qs = [i for i, k in enumerate(clean["keywords"]) if k.strip() and rel[i].any()]
        curve = grid_f1_table(sims, rel, qs)
        best_threshold[label] = float(thresholds()[curve[:, 2].argmax()])
        curves.append(pd.DataFrame({"label": label, "threshold": thresholds(),
                                    "precision": curve[:, 0], "recall": curve[:, 1], "f1": curve[:, 2]}))

    area = table[(table["label"] == "area") & table["method"].isin(SEMANTIC_METHODS)]
    chosen_method = area.sort_values("f1", ascending=False).iloc[0]["method"]
    settings = {"method": chosen_method, "static_threshold": config.STATIC_THRESHOLD,
                "dynamic_k": config.DYNAMIC_K, "grid_threshold": best_threshold["area"],
                "top_k": config.TOP_K, "model_name": store["model_name"], "synthetic": synthetic}

    out = config.PROCESSED_DIR
    results.to_csv(out / "retrieval_results.csv", index=False, encoding="utf-8-sig")
    pd.concat(curves).to_csv(out / "grid_search_curve.csv", index=False, encoding="utf-8-sig")
    config.RETRIEVAL_SETTINGS_FILE.write_text(json.dumps(settings, indent=2), encoding="utf-8")
    report = build_report(table, info, results, best_threshold, chosen_method, synthetic,
                          len(clean), store["model_name"])
    (out / "retrieval_report.txt").write_text(report, encoding="utf-8")
    print(report)
    print(f"\nSaved outputs to {out}")


if __name__ == "__main__":
    main()
