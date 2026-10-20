"""Tests for the submission feature of the web app. Run with:  pytest -q"""
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from submissions import (build_record, check_submission, edit_ratio,  # noqa: E402
                         find_personal_details, load_submissions, receipt_text,
                         save_submission)

DRAFT = {"problem": "The bus is late every morning and I miss work.",
         "suggestion": "Add more buses at peak hours.",
         "prompt_type": "few-shot", "model": "llama", "example_ids": ["SYN-014", "SYN-015"]}


def test_find_personal_details():
    assert find_personal_details("Call 077 123 4567 or a@b.com") == ["phone number", "email address"]
    assert find_personal_details("My NIC is 200012345678") == ["NIC number"]
    assert find_personal_details("The bus is late.") == []


def test_edit_ratio():
    assert edit_ratio("the bus is late", "the bus is late") == 0.0
    assert 0 < edit_ratio("the bus is late", "the bus is very late") < 0.5
    assert edit_ratio("the bus is late", "") == 1.0


def test_record_keeps_ai_draft_and_measures_edits():
    unchanged = build_record(DRAFT["problem"], DRAFT["suggestion"], "Infrastructure / Transport",
                             "bus delay", DRAFT)
    assert unchanged["ai_assisted"] and unchanged["edited"] is False and unchanged["edit_ratio"] == 0.0
    edited = build_record("The 7am bus is always late.", DRAFT["suggestion"],
                          "Infrastructure / Transport", "bus delay", DRAFT)
    assert edited["edited"] is True and edited["edit_ratio"] > 0
    assert edited["ai_problem"] == DRAFT["problem"] and edited["example_ids"] == "SYN-014; SYN-015"
    own = build_record("I wrote this myself about the late buses.", "", "Other", "", None)
    assert not own["ai_assisted"] and own["ai_problem"] == ""


def test_check_submission():
    ok = build_record(DRAFT["problem"], DRAFT["suggestion"], "Infrastructure / Transport", "", DRAFT)
    assert check_submission(ok) == []
    assert check_submission(build_record("", "", "Healthcare", "", None)) == ["Please describe the problem."]
    assert "at least" in check_submission(build_record("Too short.", "", "Healthcare", "", None))[0]
    with_phone = build_record(DRAFT["problem"] + " Call me on 077 123 4567.", "", "Healthcare", "", None)
    assert "phone number" in check_submission(with_phone)[0]
    assert check_submission(build_record(DRAFT["problem"], "", "Not an area", "", None))


def test_save_gives_reference_numbers_per_day(tmp_path):
    path = tmp_path / "submissions" / "submissions.csv"
    record = build_record(DRAFT["problem"], DRAFT["suggestion"], "Healthcare", "bus", DRAFT)
    first = save_submission(record, path, now=datetime(2026, 10, 9, 10, 0))
    second = save_submission(record, path, now=datetime(2026, 10, 9, 11, 0))
    next_day = save_submission(record, path, now=datetime(2026, 10, 10, 9, 0))
    assert first["reference"] == "SUB-20261009-0001"
    assert second["reference"] == "SUB-20261009-0002"
    assert next_day["reference"] == "SUB-20261010-0001"

    saved = load_submissions(path)
    assert len(saved) == 3 and list(saved["reference"]) == [first["reference"], second["reference"],
                                                            next_day["reference"]]
    assert saved.loc[0, "ai_problem"] == DRAFT["problem"]


def test_receipt_text():
    record = build_record(DRAFT["problem"], DRAFT["suggestion"], "Healthcare", "bus", DRAFT)
    draft_copy = receipt_text(record)
    assert "DRAFT (not submitted)" in draft_copy and DRAFT["problem"] in draft_copy
    assert "AI assistance" in draft_copy and "without changes" in draft_copy
    saved_copy = receipt_text({**record, "reference": "SUB-20261009-0001",
                               "submitted_at": "2026-10-09T10:00:00"})
    assert "SUB-20261009-0001" in saved_copy and "2026-10-09 10:00:00" in saved_copy
    own = receipt_text(build_record("My own words about the buses here.", "", "Other", "", None))
    assert "Written by the citizen." in own
