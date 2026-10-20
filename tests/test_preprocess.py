"""Tests for Stage 1. Run from the project folder with:  pytest -q"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import config  # noqa: E402
from preprocess import clean_text, detect_language, load_raw, preprocess  # noqa: E402


def test_clean_text_removes_noise_and_personal_details():
    text = "Call 077 123 4567 or mail me@test.com. NIC 200012345678. See https://x.lk/a!!! 😡"
    out = clean_text(text)
    assert "077" not in out and "[phone removed]" in out
    assert "me@test.com" not in out and "[email removed]" in out
    assert "200012345678" not in out and "[ID removed]" in out
    assert "https" not in out and "😡" not in out and "!!!" not in out


def test_clean_text_keeps_case_and_punctuation():
    assert clean_text("The Bus was LATE. Again?") == "The Bus was LATE. Again?"


def test_sinhala_and_tamil_responses_are_not_treated_as_duplicates():
    """Different Sinhala/Tamil responses must stay; only real repeats are removed."""
    raw = pd.DataFrame({
        "response_id": ["R1", "R2", "R3", "R4"],
        "issue_area": ["Healthcare"] * 4,
        "issues_selected": [""] * 4,
        "problem_description": ["மருத்துவமனையில் வரிசை மிகவும் நீளமாக உள்ளது, வயதானவர்களுக்கு இது மிகவும் கஷ்டமாக உள்ளது.",
                                "பாடசாலையில் கணித ஆசிரியர்கள் இல்லை, மாணவர்கள் தனியார் வகுப்புக்கு செல்கிறார்கள்.",
                                "රෝහලේ බෙහෙත් නැති නිසා පිටින් මිලදී ගන්න සිදුවෙනවා, එය ඉතා වියදම් සහිතයි.",
                                "රෝහලේ බෙහෙත් නැති නිසා පිටින් මිලදී ගන්න සිදුවෙනවා, එය ඉතා වියදම් සහිතයි."],
        "suggested_improvement": [""] * 4,
        "keywords": ["a", "b", "c", "c"],
    })
    clean, removed, _ = preprocess(raw)
    assert list(clean["response_id"]) == ["R1", "R2", "R3"]
    assert list(removed["response_id"]) == ["R4"] and removed.loc[0, "removal_reason"] == "duplicate"


def test_detect_language():
    assert detect_language("Hospital queue is long") == "english"
    assert detect_language("රෝහලේ පෝලිම දිගයි") == "sinhala"
    assert detect_language("மருத்துவமனையில் வரிசை நீளம்") == "tamil"


def _mock_form_export(path: Path) -> None:
    """Write a CSV shaped like a real Google Forms export."""
    q = config.ISSUES_QUESTION
    headers = (["Timestamp",
                "Do you consent to participate in this study on the terms described above?",
                "Age range", "Gender", "Province of residence",
                "Which area does your issue relate to?",
                q, q,  # two domain checklists with the same title
                "Please name the area your issue relates to",
                "Describe the problem or issue you have experienced, in as much detail as possible.",
                "What improvements or solutions would you suggest to the government regarding this issue?",
                "In 2–3 words, what keywords would you use to summarize your issue above?"])
    yes, no = "Yes, I consent and wish to continue", config.CONSENT_NO
    rows = [
        ["2026/10/10 10:00", yes, "18–24", "Female", "Western Province", "Healthcare",
         "Long waiting times at hospitals or clinics", "", "",
         "I waited five hours at the clinic with my grandmother and there were no seats.",
         "Use an appointment system.", "clinic waiting"],
        ["2026/10/10 10:05", no, "", "", "", "", "", "", "", "", "", ""],
        ["2026/10/10 10:09", yes, "25–34", "Male", "Central Province", "Education",
         "", "Shortage of teachers", "",
         "Our school has had no maths teacher for the whole term and students are falling behind.",
         "Recruit teachers quickly.", "teacher shortage"],
        ["2026/10/10 10:12", yes, "35–44", "Male", "Uva Province", "Other",
         "", "", "Tourism",
         "Tourist information offices in our town are closed most days, which affects local guides and visitors.",
         "Keep offices open on weekends.", "tourist information"],
        ["2026/10/10 10:20", yes, "45–54", "Female", "Southern Province", "Healthcare",
         "Shortage of medicines", "", "",
         "රෝහලේ ඖෂධ නොමැති නිසා අපට පිටතින් මිලදී ගැනීමට සිදුවේ. මෙය ඉතා වියදම් සහිතයි.",
         "ඖෂධ සැපයුම වැඩි දියුණු කරන්න.", "medicine shortage"],
    ]
    pd.DataFrame(rows, columns=headers).to_csv(path, index=False, encoding="utf-8-sig")


def test_form_export_pipeline(tmp_path):
    path = tmp_path / "form_export.csv"
    _mock_form_export(path)

    raw = load_raw(path)
    assert len(raw) == 4, "the non-consenting row should be removed"
    assert list(raw["response_id"]) == ["R001", "R002", "R003", "R004"]
    assert raw.loc[0, "issues_selected"] == "Long waiting times at hospitals or clinics"
    assert raw.loc[1, "issues_selected"] == "Shortage of teachers"
    assert raw.loc[2, "issue_area"] == "Other: Tourism"

    clean, removed, sentences = preprocess(raw)
    assert len(clean) == 4 and len(removed) == 0
    sinhala = clean[clean["response_id"] == "R004"].iloc[0]
    assert sinhala["language"] == "sinhala" and bool(sinhala["needs_translation"])
    assert set(sentences["response_id"]) == set(clean["response_id"])


def _rows(**extra):
    base = {"response_id": ["R1", "R2"], "timestamp": ["t1", "t2"], "issue_area": ["Healthcare"] * 2,
            "issues_selected": [""] * 2, "keywords": ["k1", "k2"],
            "problem_description": ["The hospital queue was very long and elderly people had to stand.",
                                    "රෝහලේ බෙහෙත් නැති නිසා පිටින් මිලදී ගන්න සිදුවෙනවා, එය ඉතා වියදම් සහිතයි."],
            "suggested_improvement": ["Add seats.", "බෙහෙත් ලබා දෙන්න."]}
    base.update(extra)
    return pd.DataFrame(base)


def test_manual_exclusion_and_forced_language(monkeypatch):
    monkeypatch.setattr(config, "MANUAL_EXCLUSIONS", {"t1": "off-topic"})
    monkeypatch.setattr(config, "FORCE_LANGUAGE", {})
    clean, removed, _ = preprocess(_rows())
    assert removed.loc[0, "removal_reason"] == "researcher decision: off-topic"
    monkeypatch.setattr(config, "MANUAL_EXCLUSIONS", {})
    monkeypatch.setattr(config, "FORCE_LANGUAGE", {"t1": "sinhala"})
    clean, _, _ = preprocess(_rows())
    assert clean.loc[0, "language"] == "sinhala" and bool(clean.loc[0, "needs_translation"])


def test_pasted_answers_are_trimmed_only_when_certain():
    from preprocess import trim_pasted_answer
    assert trim_pasted_answer("Problem text.\tAdd seats.\tkeywords\n", "Add seats.") == "Problem text."
    assert trim_pasted_answer("Problem text.\tsomething else", "Add seats.") is None
    assert trim_pasted_answer("No tab here.", "Add seats.") is None


def test_translations_used_only_after_check(monkeypatch):
    monkeypatch.setattr(config, "MANUAL_EXCLUSIONS", {})
    monkeypatch.setattr(config, "FORCE_LANGUAGE", {})
    tr = pd.DataFrame({"timestamp": ["t2"],
                       "problem_en": ["The hospital has no medicines, so we have to buy them outside, which is costly."],
                       "suggestion_en": ["Supply medicines."], "keywords_en": ["medicine shortage"],
                       "translated_by": ["Claude (draft)"], "checked_by": [""]})
    clean, _, _ = preprocess(_rows(), tr, require_check=True)
    r2 = clean[clean.response_id == "R2"].iloc[0]
    assert bool(r2["needs_translation"]) and not r2["translated"] and "waiting for check" in r2["notes"]

    tr.loc[0, "checked_by"] = "Theekshana"
    clean, _, sentences = preprocess(_rows(), tr, require_check=True)
    r2 = clean[clean.response_id == "R2"].iloc[0]
    assert r2["translated"] and not r2["needs_translation"] and r2["language"] == "sinhala"
    assert r2["keywords"] == "medicine shortage" and "checked by Theekshana" in r2["notes"]
    assert "Supply medicines." in r2["response_text"]


def test_duplicates_found_before_translation(monkeypatch):
    """Two identical Sinhala answers: translating the first must not make the
    second look new."""
    monkeypatch.setattr(config, "MANUAL_EXCLUSIONS", {})
    monkeypatch.setattr(config, "FORCE_LANGUAGE", {})
    rows = _rows(response_id=["R1", "R2"], timestamp=["t1", "t2"],
                 problem_description=[_rows().problem_description[1]] * 2,
                 suggested_improvement=[_rows().suggested_improvement[1]] * 2)
    tr = pd.DataFrame({"timestamp": ["t1"],
                       "problem_en": ["The hospital has no medicines, so we have to buy them outside, which is costly."],
                       "suggestion_en": ["Supply medicines."], "keywords_en": ["medicine shortage"],
                       "translated_by": ["Claude (draft)"], "checked_by": ["Checker"]})
    clean, removed, _ = preprocess(rows, tr)
    assert list(clean["response_id"]) == ["R1"] and bool(clean.loc[0, "translated"])
    assert list(removed["response_id"]) == ["R2"] and removed.loc[0, "removal_reason"] == "duplicate"
