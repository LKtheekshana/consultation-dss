"""Tests for the invented-name check. Run with:  pytest -q"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from factcheck import candidate_names, unsupported_names  # noqa: E402

SOURCES = ["At the clinic in Gampaha District General Hospital the pharmacy had no medicines.",
           "The CEB hotline is always busy and the Pradeshiya Sabha does not collect garbage."]


def test_invented_ministry_is_flagged():
    summary = ("A series of citizen complaints have been lodged with the Ministry of Health and "
               "Family Welfare (MoHFW) in Colombo.")
    assert unsupported_names(summary, SOURCES) == ["Ministry of Health and Family Welfare", "MoHFW", "Colombo"]


def test_names_from_the_submissions_are_not_flagged():
    summary = "Patients at Gampaha District General Hospital lack medicines and the CEB hotline is busy."
    assert unsupported_names(summary, SOURCES) == []


def test_sentence_starts_and_context_names_are_ignored():
    summary = "Citizens are unhappy. They want Sri Lanka's government to act. These problems are serious."
    assert candidate_names(summary) == ["Sri Lanka's"]
    assert unsupported_names(summary, SOURCES) == []
    assert unsupported_names("People in Jaffna complain.", SOURCES) == ["Jaffna"]


def test_names_do_not_run_across_sentences_and_starters_are_dropped():
    assert candidate_names("They live in Sri Lanka. They want help.") == ["Sri Lanka"]
    assert candidate_names("Residents of Colombo are angry.") == ["Colombo"]
