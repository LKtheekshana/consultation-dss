"""Simple check for invented names in AI-written text.

Summarization models sometimes add names that are in none of the
submissions (in testing, FLAN-T5 wrote "Ministry of Health and Family
Welfare (MoHFW)", an Indian ministry). This module finds names in a summary,
such as organizations, places and acronyms, and returns the ones that do not
appear in the source submissions, so the web app can warn the reader and
Stage 7 can count how often each model does this.

It is a heuristic, not a full fact check: it only looks at capitalized
names, so a wrong number or a wrong claim in plain words is not caught.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402

# A name: capitalized words, optionally joined by "of", "and", "for" or "the"
# ("Ministry of Health and Family Welfare"), or an acronym ("MoHFW", "CEB").
NAME_RE = re.compile(
    r"\b[A-Z][\w'’-]*(?:\s+(?:(?:of|and|for|the)\s+)?[A-Z][\w'’-]*)*"
)
# Ordinary words that are capitalized only because they start a sentence.
# "Residents of Colombo" is checked as "Colombo".
SENTENCE_STARTERS = {"a", "an", "the", "these", "this", "they", "their", "there", "many", "some",
                     "most", "citizens", "residents", "people", "patients", "students", "farmers",
                     "parents", "families", "complaints", "main", "problems", "issues"}
ACRONYM_RE = re.compile(r"\b[A-Za-z]*[A-Z][A-Za-z]*[A-Z][A-Za-z]*\b")   # two or more capitals


def candidate_names(text: str) -> list[str]:
    """Names in the text, skipping ordinary words that are capitalized only
    because they start a sentence ("These", "Citizens", ...)."""
    names = []
    for m in NAME_RE.finditer(text):
        name = m.group().strip(" .-'’")
        before = text[:m.start()].rstrip()
        sentence_start = before == "" or before[-1] in ".!?:;(\"“"
        if sentence_start:
            first, _, rest = name.partition(" ")
            if first.lower() in SENTENCE_STARTERS:
                # drop the ordinary first word and a joining word after it
                name = re.sub(r"^(?:of|and|for|the)\s+", "", rest).strip()
            elif not rest and not ACRONYM_RE.fullmatch(name):
                continue                                # e.g. "Long", "Water" at sentence start
        if name:
            names.append(name)
    for m in ACRONYM_RE.finditer(text):           # acronyms inside longer phrases
        if m.group() not in names:
            names.append(m.group())
    return names


def unsupported_names(summary: str, sources: list[str]) -> list[str]:
    """Names in the summary that appear in none of the source texts
    (ignoring case), apart from general context names in config.CONTEXT_NAMES."""
    source = " ".join(sources).lower()
    found = []
    for name in candidate_names(summary):
        # "Sri Lanka's" counts as "Sri Lanka"; context names are removed, so
        # "Sri Lanka's Government" is allowed but "Sri Lanka's Ministry" is not
        rest = re.sub(r"['’]s\b", "", name.lower())
        for context in sorted(config.CONTEXT_NAMES, key=len, reverse=True):
            rest = re.sub(rf"\b{re.escape(context.lower())}\b", " ", rest)
        rest = " ".join(w for w in rest.split() if w not in {"of", "and", "for", "the"})
        if not rest or rest in source or name.lower() in source:
            continue
        if name not in found:
            found.append(name)
    return found
