"""Submissions to the consultation office (prototype).

Used by the web app. A citizen reviews and edits a draft (written by them or
by Llama), then submits it. Each submission gets a reference number and is
saved to config.SUBMISSIONS_FILE, where the "Office inbox" page shows it.

This is a prototype channel, not a connection to the Information Center's
real systems. Submissions are kept apart from the study data.

For the evaluation, every record keeps the original AI draft next to the
final text, so it can be measured how much citizens changed the drafts.
"""
from __future__ import annotations

import difflib
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402
from preprocess import EMAIL_RE, NIC_RE, PHONE_RE, word_count  # noqa: E402

COLUMNS = ["reference", "submitted_at", "issue_area", "keywords", "problem", "suggestion",
           "ai_assisted", "prompt_type", "model", "example_ids",
           "ai_problem", "ai_suggestion", "edited", "edit_ratio", "data_source"]


def find_personal_details(text: str) -> list[str]:
    """Names of the kinds of personal details found in the text (same
    patterns as Stage 1). Submissions must not contain them."""
    found = []
    for name, pattern in [("phone number", PHONE_RE), ("email address", EMAIL_RE),
                          ("NIC number", NIC_RE)]:
        if pattern.search(text or ""):
            found.append(name)
    return found


def edit_ratio(original: str, final: str) -> float:
    """Share of the text that was changed, from 0 (unchanged) to 1 (all new).
    Compared word by word, so fixing a typo counts as a small change."""
    ratio = difflib.SequenceMatcher(None, original.split(), final.split()).ratio()
    return round(1 - ratio, 3)


def next_reference(existing: pd.DataFrame, now: datetime) -> str:
    """Reference numbers look like SUB-20261009-0007: date + number of the day."""
    prefix = f"SUB-{now:%Y%m%d}-"
    today = existing["reference"].astype(str).str.startswith(prefix).sum() if len(existing) else 0
    return f"{prefix}{today + 1:04d}"


def load_submissions(path: Path = config.SUBMISSIONS_FILE) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=COLUMNS)
    return pd.read_csv(path, encoding="utf-8-sig", dtype=str, keep_default_na=False)


def build_record(problem: str, suggestion: str, issue_area: str, keywords: str,
                 draft: dict | None) -> dict:
    """The data saved for one submission.

    draft: the AI draft the citizen started from ({"problem", "suggestion",
    "prompt_type", "model", "example_ids"}), or None if they wrote it themselves.
    """
    problem, suggestion = problem.strip(), suggestion.strip()
    record = {"issue_area": issue_area, "keywords": keywords.strip(),
              "problem": problem, "suggestion": suggestion,
              "ai_assisted": draft is not None,
              "prompt_type": "", "model": "", "example_ids": "",
              "ai_problem": "", "ai_suggestion": "", "edited": "", "edit_ratio": "",
              "data_source": "Web app submission (prototype, not study data)"}
    if draft is not None:
        ai_text = f"{draft['problem']} {draft['suggestion']}".strip()
        final_text = f"{problem} {suggestion}".strip()
        record.update(prompt_type=draft["prompt_type"], model=draft["model"],
                      example_ids="; ".join(draft["example_ids"]),
                      ai_problem=draft["problem"], ai_suggestion=draft["suggestion"],
                      edited=ai_text.split() != final_text.split(),
                      edit_ratio=edit_ratio(ai_text, final_text))
    return record


def check_submission(record: dict) -> list[str]:
    """Problems that stop the submission from being sent (empty list = OK)."""
    problems = []
    if not record["problem"]:
        problems.append("Please describe the problem.")
    elif word_count(f"{record['problem']} {record['suggestion']}") < config.MIN_WORDS:
        problems.append(f"Please write at least {config.MIN_WORDS} words.")
    if record["issue_area"] not in config.ISSUE_AREAS:
        problems.append("Please choose an issue area.")
    found = find_personal_details(f"{record['problem']} {record['suggestion']}")
    if found:
        problems.append("Please remove personal details before submitting: "
                        + ", ".join(found) + ".")
    return problems


def save_submission(record: dict, path: Path = config.SUBMISSIONS_FILE,
                    now: datetime | None = None) -> dict:
    """Give the record a reference number and time, add it to the file and
    return the saved record."""
    now = now or datetime.now()
    existing = load_submissions(path)
    saved = {"reference": next_reference(existing, now),
             "submitted_at": now.isoformat(timespec="seconds"), **record}
    path.parent.mkdir(parents=True, exist_ok=True)
    row = pd.DataFrame([saved], columns=COLUMNS)
    row.to_csv(path, mode="a", header=not path.exists(), index=False,
               encoding="utf-8-sig" if not path.exists() else "utf-8")
    return saved


def receipt_text(record: dict) -> str:
    """Plain-text copy for the citizen to download. Works for a saved
    submission (with a reference) and for a draft that was not sent yet."""
    reference = record.get("reference") or "DRAFT (not submitted)"
    when = record.get("submitted_at", "").replace("T", " ") or datetime.now().strftime("%Y-%m-%d %H:%M")
    if record["ai_assisted"]:
        origin = (f"Drafted with AI assistance ({record['model']}) and reviewed by the "
                  "citizen before submission.")
        if record.get("edited") is False:
            origin += " The AI draft was submitted without changes."
    else:
        origin = "Written by the citizen."
    lines = [
        "CONSULTATION SUBMISSION",
        "=" * 60,
        f"Reference:   {reference}",
        f"Date:        {when}",
        f"Issue area:  {record['issue_area']}",
        f"Keywords:    {record['keywords'] or '-'}",
        "",
        "Problem",
        "-" * 60,
        record["problem"],
        "",
        "Suggested improvement",
        "-" * 60,
        record["suggestion"] or "-",
        "",
        "-" * 60,
        origin,
        "Prototype system: this submission is stored by the research prototype and",
        "is not sent to the Information Center's official systems.",
    ]
    return "\n".join(lines) + "\n"
