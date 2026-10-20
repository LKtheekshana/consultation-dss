"""Tests for Stage 7. BERTScore is replaced by a stand-in, so they run
offline without downloading roberta-large. Run with:  pytest -q"""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import config  # noqa: E402
import evaluate as ev  # noqa: E402


def fake_bertscore(cands, refs):
    """Stand-in: share of candidate words found in the best reference."""
    f = [max(len(set(c.lower().split()) & set(r.lower().split())) / max(len(c.split()), 1) for r in rs)
         for c, rs in zip(cands, refs)]
    return f, f, f


def test_sus_score():
    assert ev.sus_score([5, 1] * 5) == 100.0          # best possible answers
    assert ev.sus_score([1, 5] * 5) == 0.0            # worst possible answers
    assert ev.sus_score([3] * 10) == 50.0
    with pytest.raises(ValueError):
        ev.sus_score([3] * 9)


def test_control_pairs_never_pair_a_respondent_with_themselves():
    ids = [f"R{i}" for i in range(10)]
    pairs = ev.control_pairs(ids)
    assert set(pairs) == set(ids) and all(a != b for a, b in pairs.items())
    assert sorted(pairs.values()) == sorted(ids)


def test_score_texts_identical_text_scores_highest():
    s = ev.score_texts(["the bus is late every day", "rice prices are high"],
                       [["the bus is late every day"], ["the bus is late every day"]], fake_bertscore)
    assert s.loc[0, "rouge1"] == pytest.approx(1.0) and s.loc[0, "bleu"] == pytest.approx(100.0)
    assert s.loc[1, "rouge1"] < s.loc[0, "rouge1"]


def _generations():
    rows = []
    for q, real in [("R1", "The bus is late every morning. Add more buses."),
                    ("R2", "Rice and gas prices keep rising. Control prices."),
                    ("R3", "The hospital has no medicines. Supply medicines.")]:
        for ptype, text in [("zero-shot", real.split(".")[0]), ("few-shot", real)]:
            rows.append({"query_id": q, "issue_area": "x", "keywords": "k", "prompt_type": ptype,
                         "generated_text": text, "real_text": real, "seconds": "1.0"})
    return pd.DataFrame(rows)


def test_evaluate_generation_has_control_rows():
    scores = ev.evaluate_generation(_generations(), fake_bertscore)
    assert list(scores["comparison"].unique()) == ["zero-shot", "few-shot", "control (unrelated submission)"]
    few = scores[scores["comparison"] == "few-shot"]["rouge1"].mean()
    control = scores[scores["comparison"].str.startswith("control")]["rouge1"].mean()
    assert few > control
    assert any("Wilcoxon" in line for line in ev.generation_table(scores))


def test_rating_sheet_is_blinded_and_not_overwritten(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "RATING_SHEET_GENERATIONS", 2)
    clean = pd.DataFrame({"problem_description": ["Bus late."], "suggested_improvement": ["More buses."]},
                         index=["R1"])
    systems = {"FLAN-T5": pd.DataFrame({"query": ["q1"], "retrieved_ids": ["R1"], "summary": ["S1"]}),
               "Llama": pd.DataFrame({"query": ["q1"], "retrieved_ids": ["R1"], "summary": ["S2"]})}
    assert ev.make_rating_sheet(_generations(), systems, clean, tmp_path)
    sheet = pd.read_csv(tmp_path / "rating_sheet_TEMPLATE.csv", encoding="utf-8-sig", keep_default_na=False)
    key = pd.read_csv(tmp_path / "rating_key.csv", encoding="utf-8-sig")
    assert len(sheet) == 2 + 2 * 2 and "system" not in sheet.columns
    assert set(key["system"]) == {"FLAN-T5", "Llama", "Llama zero-shot", "Llama few-shot"}
    assert not ev.make_rating_sheet(_generations(), systems, clean, tmp_path), "must not overwrite"


def test_ratings_are_read_and_summarized(tmp_path):
    key = pd.DataFrame({"item_id": ["I01", "I02"], "kind": ["summary", "summary"],
                        "system": ["FLAN-T5", "Llama"], "about": ["q1", "q1"]})
    for rater, scores in [("A", (2, 4)), ("B", (3, 5))]:
        pd.DataFrame({"item_id": ["I01", "I02"], "readability": scores, "informativeness": scores,
                      "flow": scores}).to_csv(tmp_path / f"rating_{rater}.csv", index=False)
    ratings = ev.load_ratings(tmp_path)
    assert len(ratings) == 2 * 2 * 3
    lines, means = ev.ratings_table(ratings, key)
    assert means[("summary", "FLAN-T5")] == pytest.approx(2.5)
    assert means[("summary", "Llama")] == pytest.approx(4.5)
    assert any("Krippendorff" in line for line in lines)


def test_criteria_pending_and_met(monkeypatch):
    monkeypatch.setattr(ev, "retrieval_criterion", lambda: ("dynamic F1 0.5 vs keyword F1 0.3", "Met"))
    lines = ev.criteria_lines(None, None)
    assert sum("Pending" in line for line in lines) == 2 and any("Met" in line for line in lines)
    means = pd.Series({("summary", "Llama"): 4.0, ("summary", "FLAN-T5"): 2.0})
    sus = pd.DataFrame({"sus": [70.0, 80.0]})
    text = "\n".join(ev.criteria_lines(means, sus))
    assert "Not met" in text and text.count("Met") >= 3 and "Pending" not in text
