"""Tests for Stage 3. Run offline with:  pytest -q"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from retrieve import (dynamic_threshold, evaluate, keyword_select, prf,  # noqa: E402
                      relevance_matrix, retrieve, split_issues, summarize, topk_select)
from test_embed import FakeModel  # noqa: E402
from embed import embed_query, unit_length  # noqa: E402


def test_prf():
    retrieved = np.array([True, True, False, False])
    relevant = np.array([True, False, True, False])
    assert prf(retrieved, relevant) == (0.5, 0.5, 0.5)
    assert prf(np.zeros(4, bool), relevant) == (0.0, 0.0, 0.0)


def test_threshold_methods():
    scores = np.array([0.9, 0.5, 0.1, 0.3])
    assert list(topk_select(scores, 2)) == [True, True, False, False]
    assert np.isclose(dynamic_threshold(scores, 1.0), scores.mean() + scores.std())


def test_keyword_baseline_matches_whole_words():
    texts = ["The bus was late.", "Buses are crowded.", "Hospital and bus"]
    assert list(keyword_select("bus delay", texts, "any")) == [True, False, True]
    assert list(keyword_select("bus late", texts, "all")) == [True, False, False]
    assert not keyword_select("the and", texts).any(), "stopwords alone match nothing"


def test_split_issues_keeps_options_with_commas():
    text = "Difficulty getting appointments, tests or referrals, Shortage of medicines; Power cuts"
    assert split_issues(text) == {"Difficulty getting appointments, tests or referrals",
                                  "Shortage of medicines", "Power cuts"}


def _clean():
    return pd.DataFrame({
        "response_id": ["A", "B", "C", "D", "E", "F"],
        "issue_area": ["Health", "Health", "Health", "Transport", "Transport", "Transport"],
        "issues_selected": ["Long waits", "Long waits", "Medicines", "Buses", "Roads", "Buses"],
        "keywords": ["hospital queue", "clinic queue", "medicine shortage",
                     "bus late", "road potholes", "bus crowded"],
        "response_text": ["Long hospital queue at the clinic.", "The clinic queue took hours at the hospital.",
                          "No medicine in the hospital pharmacy.", "The bus is always late.",
                          "Road has potholes and the bus shakes.", "The bus is crowded and late."],
    })


def test_evaluate_runs_all_methods_and_leaves_query_out():
    clean = _clean()
    model = FakeModel()
    matrix = unit_length(model.encode(clean["response_text"].tolist(), normalize_embeddings=True))
    queries = np.vstack([embed_query(model, k) for k in clean["keywords"]])
    sims = queries @ matrix.T

    results, info = evaluate(clean, sims)
    assert set(results["method"]) == {"static", "dynamic", "grid", "top-k", "keyword"}
    # each query has 5 candidates; the 2 other same-area responses are relevant
    area = results[results["label"] == "area"]
    assert (area["n_relevant"] == 2).all()
    assert (area["n_retrieved"] <= 5).all()
    # "issue" label: C and E share no issue with anyone, so they are skipped
    assert info["skipped"]["issue"] == 2

    table = summarize(results)
    assert table["f1"].between(0, 1).all()


def test_relevance_matrix_area():
    rel = relevance_matrix(_clean(), "area")
    assert rel[0, 1] and not rel[0, 3]


def test_retrieve_uses_saved_settings():
    store = {"response_ids": ["A", "B", "C"],
             "embeddings": unit_length(np.array([[1, 0], [0.8, 0.6], [0, 1]], dtype=float))}
    q = np.array([1.0, 0.0])
    base = {"static_threshold": 0.5, "dynamic_k": 1.0, "grid_threshold": 0.7, "top_k": 2}
    assert [r for r, _ in retrieve(q, store, {**base, "method": "static"})] == ["A", "B"]
    assert [r for r, _ in retrieve(q, store, {**base, "method": "grid"})] == ["A", "B"]
    assert [r for r, _ in retrieve(q, store, {**base, "method": "top-k", "top_k": 1})] == ["A"]
