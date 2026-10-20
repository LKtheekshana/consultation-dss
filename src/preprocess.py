"""Stage 1: Preprocessing.

Reads the questionnaire export (or the synthetic test file), cleans the
text and saves the data that later stages need.

Run from the project folder:
    python src/preprocess.py

Outputs (in data/processed/):
    clean_responses.csv    one row per usable response
    sentences.csv          one row per sentence (used for embeddings)
    removed_responses.csv  rows that were dropped, with the reason
    preprocessing_report.txt  counts and statistics
"""
from __future__ import annotations

import re
import sys
import unicodedata
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402

# ---------------------------------------------------------------------------
# Text patterns
# ---------------------------------------------------------------------------
URL_RE = re.compile(r"(https?://[^\s)\]]+|www\.[^\s)\]]+)", re.IGNORECASE)
EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
# Sri Lankan phone numbers: 07X XXX XXXX, 0XX XXX XXXX, +94 7X XXX XXXX
PHONE_RE = re.compile(r"(\+94|0094|\b0)\s?\d{2}[\s-]?\d{3}[\s-]?\d{4}\b")
# NIC numbers: old format 9 digits + V/X, new format 12 digits
NIC_RE = re.compile(r"\b(\d{9}[VvXx]|\d{12})\b")
EMOJI_RE = re.compile(
    "["
    "\U0001F300-\U0001FAFF"  # symbols, pictographs, emoticons
    "\U00002700-\U000027BF"  # dingbats
    "\U00002600-\U000026FF"  # misc symbols
    "\U0000FE0F"             # variation selector
    "]+"
)
REPEATED_PUNCT_RE = re.compile(r"([!?.]){2,}")
SINHALA_RE = re.compile(r"[඀-෿]")
TAMIL_RE = re.compile(r"[஀-௿]")


def clean_text(text: str) -> str:
    """Remove noise and personal details while keeping case and punctuation.

    Case and punctuation are kept on purpose: the sentence transformer and
    T5 were trained on natural text, and punctuation marks sentence ends.
    """
    if not isinstance(text, str):
        return ""
    text = URL_RE.sub(" ", text)
    text = EMAIL_RE.sub("[email removed]", text)
    text = NIC_RE.sub("[ID removed]", text)
    text = PHONE_RE.sub("[phone removed]", text)
    text = EMOJI_RE.sub(" ", text)
    text = REPEATED_PUNCT_RE.sub(r"\1", text)      # "!!!" -> "!"
    text = re.sub(r"\(\s*see\s*\)", " ", text)      # leftover "(see )"
    text = re.sub(r"\(\s*\)", " ", text)            # empty brackets
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s+([,.!?;:])", r"\1", text)    # no space before punctuation
    return text


def detect_language(text: str) -> str:
    """Return 'sinhala', 'tamil' or 'english' based on the script used."""
    if not isinstance(text, str):
        return "english"
    if SINHALA_RE.search(text):
        return "sinhala"
    if TAMIL_RE.search(text):
        return "tamil"
    return "english"


def end_sentence(text: str) -> str:
    """Add a full stop if the text does not already end with . ! or ?"""
    text = text.strip() if isinstance(text, str) else ""
    if text and text[-1] not in ".!?":
        text += "."
    return text


def normalize_for_duplicates(text: str) -> str:
    """Lower-case text with punctuation and spacing removed, used to find
    duplicate responses.

    Written character by character on purpose: pandas' regex ``\\W`` treats
    every Sinhala and Tamil character as a non-letter, which made all
    Sinhala/Tamil responses look identical. Letters (L), vowel signs (M) and
    digits (N) in any script are kept here."""
    kept = [c if unicodedata.category(c)[0] in "LMN" else " " for c in str(text).lower()]
    return " ".join("".join(kept).split())


def word_count(text: str) -> int:
    return len(text.split()) if isinstance(text, str) else 0


def _regex_sentences(text: str) -> list[str]:
    """Simple sentence splitter used if NLTK's data is unavailable."""
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])", text)
    return [p.strip() for p in parts if p.strip()]


def split_sentences(text: str) -> list[str]:
    """Split text into sentences, using NLTK when it is available."""
    if not text:
        return []
    try:
        from nltk.tokenize import sent_tokenize
        return [s.strip() for s in sent_tokenize(text) if s.strip()]
    except (ImportError, LookupError):
        return _regex_sentences(text)


def ensure_nltk_data() -> str:
    """Download NLTK's sentence model if needed. Returns which splitter is used."""
    try:
        import nltk
        try:
            nltk.data.find("tokenizers/punkt_tab")
        except LookupError:
            nltk.download("punkt_tab", quiet=True)
            nltk.data.find("tokenizers/punkt_tab")
        return "NLTK punkt"
    except Exception:
        return "regex fallback (NLTK data unavailable)"


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_raw(path: Path) -> pd.DataFrame:
    """Load either the synthetic test file or a Google Forms export and
    return a dataframe with standard column names."""
    df = pd.read_csv(path, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    df.columns = [c.strip() for c in df.columns]

    is_form_export = "Timestamp" in df.columns
    if is_form_export:
        # Merge the per-domain checklist columns into one column.
        # pandas renames duplicate titles to "title.1", "title.2", ...
        issue_cols = [c for c in df.columns if c.startswith(config.ISSUES_QUESTION)]
        if issue_cols:
            df["issues_selected"] = df[issue_cols].apply(
                lambda r: "; ".join(v for v in r if v), axis=1)
            df = df.drop(columns=issue_cols)
        df = df.rename(columns=config.FORM_COLUMN_MAP)

        # Remove people who did not consent
        if "consent" in df.columns:
            df = df[df["consent"] != config.CONSENT_NO]

        # If "Other" was chosen, use the area they typed in
        if "other_area" in df.columns:
            df["other_area"] = df["other_area"].str.strip()
            df["issue_area"] = df.apply(
                lambda r: f"Other: {r['other_area']}" if r["issue_area"] == "Other" and r["other_area"] else r["issue_area"],
                axis=1)

        df.insert(0, "response_id", [f"R{i:03d}" for i in range(1, len(df) + 1)])
        df["data_source"] = "Questionnaire"

    required = ["response_id", "issue_area", "problem_description",
                "suggested_improvement", "keywords"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing columns: {missing}. If this is a form export, check that "
            "the question titles in config.FORM_COLUMN_MAP match the CSV headers.")
    for col in ["issues_selected", "data_source", "timestamp"]:
        if col not in df.columns:
            df[col] = ""
    return df.reset_index(drop=True)


def load_translations(path: Path = config.TRANSLATIONS_FILE) -> pd.DataFrame | None:
    """The translations file (see README), or None if there is none yet."""
    if not path.exists():
        return None
    return pd.read_csv(path, encoding="utf-8-sig", dtype=str, keep_default_na=False)


def trim_pasted_answer(problem: str, suggestion: str) -> str | None:
    """Some respondents pasted all their answers into the problem box,
    separated by tabs. If the text after the first tab starts with their own
    suggestion, return the problem without the pasted copy; otherwise None
    (nothing is changed unless the copy is certain)."""
    if "\t" not in problem or not suggestion.strip():
        return None
    first, rest = problem.split("\t", 1)
    if normalize_for_duplicates(rest).startswith(normalize_for_duplicates(suggestion)):
        return first.strip()
    return None


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def add_note(df: pd.DataFrame, i, note: str) -> None:
    """Add a note to a response (several notes are joined with "; ")."""
    df.at[i, "notes"] = f"{df.at[i, 'notes']}; {note}" if df.at[i, "notes"] else note


def preprocess(df: pd.DataFrame, translations: pd.DataFrame | None = None,
               require_check: bool = config.TRANSLATIONS_REQUIRE_CHECK
               ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Clean responses. Returns (clean, removed, sentences).

    translations: the translations file (columns timestamp, problem_en,
    suggestion_en, keywords_en, translated_by, checked_by), or None.
    """
    df = df.copy()
    if "timestamp" not in df.columns:
        df["timestamp"] = ""
    df["notes"] = ""

    # Answers pasted twice into the problem box (see trim_pasted_answer)
    for i in df.index:
        trimmed = trim_pasted_answer(df.at[i, "problem_description"], df.at[i, "suggested_improvement"])
        if trimmed is not None:
            df.at[i, "problem_description"] = trimmed
            add_note(df, i, "pasted copy of other answers removed from problem text")

    # Duplicates are found on the respondent's own words, before translation:
    # once one copy is translated, it no longer matches the untranslated copy
    df["original_text"] = df["problem_description"] + " " + df["suggested_improvement"]
    df["original_words"] = (df["problem_description"].map(word_count)
                            + df["suggested_improvement"].map(word_count))

    # Language is detected on the original text so nothing is lost by cleaning.
    # Sinhala/Tamil written in English letters is set by hand in the config.
    df["language"] = (df["problem_description"] + " " + df["suggested_improvement"]).map(detect_language)
    for i in df.index:
        if df.at[i, "timestamp"] in config.FORCE_LANGUAGE:
            df.at[i, "language"] = config.FORCE_LANGUAGE[df.at[i, "timestamp"]]

    # Use checked English translations in place of the Sinhala/Tamil text
    df["translated"] = False
    if translations is not None:
        by_time = translations.set_index("timestamp")
        for i in df.index[df["language"] != "english"]:
            ts = df.at[i, "timestamp"]
            if ts not in by_time.index or not by_time.at[ts, "problem_en"].strip():
                continue
            t = by_time.loc[ts]
            checked = bool(t["checked_by"].strip())
            if require_check and not checked:
                add_note(df, i, "translation drafted, waiting for check")
                continue
            df.at[i, "problem_description"] = t["problem_en"]
            df.at[i, "suggested_improvement"] = t["suggestion_en"]
            df.at[i, "keywords"] = t["keywords_en"]
            df.at[i, "translated"] = True
            add_note(df, i, f"translated from {df.at[i, 'language']} by {t['translated_by']}; "
                                 + (f"checked by {t['checked_by']}" if checked else "NOT CHECKED"))

    for col in ["problem_description", "suggested_improvement", "keywords"]:
        df[col] = df[col].map(clean_text)

    # The text the system will search and summarize: problem + suggestion
    df["response_text"] = df.apply(
        lambda r: " ".join(p for p in [end_sentence(r["problem_description"]),
                                       end_sentence(f"Suggested improvement: {r['suggested_improvement']}")
                                       if r["suggested_improvement"] else ""] if p),
        axis=1)
    df["word_count"] = df["response_text"].map(word_count)

    # ---- decide which rows to remove, recording the reason ----
    reasons = pd.Series("", index=df.index)
    for i in df.index:
        reason = config.MANUAL_EXCLUSIONS.get(df.at[i, "timestamp"])
        if reason:
            reasons[i] = f"researcher decision: {reason}"
    reasons[(df["problem_description"] == "") & (reasons == "")] = "empty description"

    normalized = df["original_text"].map(clean_text).map(normalize_for_duplicates)
    dup = normalized.duplicated(keep="first") & (reasons == "")
    reasons[dup] = "duplicate"

    short = (df["word_count"] < config.MIN_WORDS) & (reasons == "")
    reasons[short] = f"too short (< {config.MIN_WORDS} words)"

    removed = df[reasons != ""].assign(removal_reason=reasons[reasons != ""])
    clean = df[reasons == ""].copy()

    # Sinhala/Tamil responses without a checked translation are kept but
    # flagged; Stage 2 skips them (see README). They are not removed.
    clean["needs_translation"] = (clean["language"] != "english") & ~clean["translated"]
    clean["approx_tokens"] = (clean["word_count"] / 0.75).round().astype(int)

    # ---- sentence table for embeddings ----
    rows = []
    for _, r in clean.iterrows():
        for i, s in enumerate(split_sentences(r["response_text"]), start=1):
            rows.append({"response_id": r["response_id"], "sentence_no": i, "sentence": s})
    sentences = pd.DataFrame(rows, columns=["response_id", "sentence_no", "sentence"])
    sent_counts = sentences.groupby("response_id").size()
    clean["sentence_count"] = clean["response_id"].map(sent_counts).fillna(0).astype(int)

    return clean.reset_index(drop=True), removed.reset_index(drop=True), sentences


def build_report(raw: pd.DataFrame, clean: pd.DataFrame, removed: pd.DataFrame,
                 sentences: pd.DataFrame, splitter: str, source: Path) -> str:
    lines = []
    add = lines.append
    add("PREPROCESSING REPORT")
    add("=" * 60)
    add(f"Input file:          {source.name}")
    if (raw.get("data_source", pd.Series(dtype=str)).astype(str).str.contains("SYNTHETIC")).any():
        add("WARNING: this input is SYNTHETIC test data. Do not report these")
        add("numbers as results; rerun on the real questionnaire export.")
    add(f"Sentence splitter:   {splitter}")
    add("")
    add(f"Responses loaded:    {len(raw)}")
    add(f"Responses removed:   {len(removed)}")
    for reason, n in removed["removal_reason"].value_counts().items():
        add(f"    - {reason}: {n}")
    add(f"Responses kept:      {len(clean)}")
    add(f"Written in Sinhala:  {int((clean['language'] == 'sinhala').sum())}, "
        f"Tamil: {int((clean['language'] == 'tamil').sum())}")
    add(f"    translated and checked:     {int(clean['translated'].sum())}")
    add(f"    still need translation:     {int(clean['needs_translation'].sum())}"
        "  (skipped by Stage 2 until translated)")
    unchecked = clean["notes"].str.contains("NOT CHECKED").sum()
    if unchecked:
        add(f"    WARNING: {unchecked} translation(s) used WITHOUT a check "
            "(trial run only; do not report these results)")
    add(f"Total sentences:     {len(sentences)}")
    pasted = clean[clean["notes"].str.contains("pasted copy")]
    if len(pasted):
        add(f"Pasted copies of other answers removed from the problem text: "
            + ", ".join(pasted["response_id"]))
    add("")
    add("Responses per issue area")
    add("-" * 60)
    for area, n in clean["issue_area"].value_counts().items():
        add(f"    {area:<52}{n:>4}")
    add("")
    add("Word count of response text (description + suggestion)")
    add("-" * 60)
    stats = clean["word_count"].describe()
    for k in ["count", "mean", "std", "min", "25%", "50%", "75%", "max"]:
        add(f"    {k:<10}{stats[k]:>10.2f}")
    over = clean[clean["approx_tokens"] > config.T5_TOKEN_LIMIT]
    add("")
    add(f"Responses above the T5 limit (~{config.T5_TOKEN_LIMIT} tokens): {len(over)}")
    if len(over):
        add("    " + ", ".join(over["response_id"]))
    return "\n".join(lines)


def main() -> None:
    source = config.INPUT_FILE
    if not source.exists():
        print(f"Input file not found: {source}\n"
              "Export the Google Form responses as CSV (Responses tab > three-dot "
              "menu > Download responses), unzip it, and save it there under that name.")
        sys.exit(1)
    print(f"Reading {source} ...")
    raw = load_raw(source)
    splitter = ensure_nltk_data()

    clean, removed, sentences = preprocess(raw, load_translations())

    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    keep_cols = ["response_id", "timestamp", "data_source", "issue_area", "issues_selected",
                 "problem_description", "suggested_improvement", "keywords",
                 "response_text", "language", "translated", "needs_translation", "notes",
                 "word_count", "approx_tokens", "sentence_count"]
    # Demographic columns are kept (if present) for the sample description
    extra = [c for c in ["age_range", "gender", "province", "prior_use", "satisfaction"] if c in clean.columns]
    clean[keep_cols + extra].to_csv(config.PROCESSED_DIR / "clean_responses.csv", index=False, encoding="utf-8-sig")
    sentences.to_csv(config.PROCESSED_DIR / "sentences.csv", index=False, encoding="utf-8-sig")
    removed[["response_id", "timestamp", "removal_reason", "problem_description"]].to_csv(
        config.PROCESSED_DIR / "removed_responses.csv", index=False, encoding="utf-8-sig")

    report = build_report(raw, clean, removed, sentences, splitter, source)
    (config.PROCESSED_DIR / "preprocessing_report.txt").write_text(report, encoding="utf-8")
    print(report)
    print(f"\nSaved outputs to {config.PROCESSED_DIR}")


if __name__ == "__main__":
    main()
