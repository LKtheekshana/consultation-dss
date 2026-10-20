"""Convert the Google Forms response exports into the files evaluate.py reads.

    data/raw/google_forms/Evaluation 1_ Reference Summaries.csv   -> data/raw/reference_summaries.csv
    data/raw/google_forms/Evaluation 2_ Rating Computer-Generated Texts.csv
                                                                  -> data/raw/human_ratings/rating_<rater>.csv
    data/raw/google_forms/Evaluation 3_ Usability Questionnaire (SUS).csv
                                                                  -> data/raw/sus_responses.csv

Answers are copied as given; nothing is filled in or corrected. Run:
    python src/import_google_forms.py
then python src/evaluate.py.
"""
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402

FORMS_DIR = config.RAW_DIR / "google_forms"
EVAL1 = FORMS_DIR / "Evaluation 1_ Reference Summaries.csv"
EVAL2 = FORMS_DIR / "Evaluation 2_ Rating Computer-Generated Texts.csv"
EVAL3 = FORMS_DIR / "Evaluation 3_ Usability Questionnaire (SUS).csv"
TOPICS = {chr(ord("A") + i): q for i, q in enumerate(config.EVAL_QUERIES)}


def read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig", dtype=str, keep_default_na=False)


def rater_id(name: str) -> str:
    """Initials for file names, e.g. 'A.B.C Perera' -> 'ABCP'."""
    parts = re.findall(r"[A-Za-z]+", name)
    return "".join(p[0] if len(p) > 1 else p for p in parts).upper() if len(parts) > 1 else name.strip().upper()


def import_references() -> list[str]:
    """Fill reference_summaries.csv: each topic's summaries go into its writer slots in form order."""
    form = read(EVAL1)
    refs = read(config.REFERENCE_SUMMARIES_FILE)
    refs["reference_summary"] = ""
    refs["written_by"] = ""
    notes = []
    for letter, query in TOPICS.items():
        col = f"Your summary of Topic {letter} (40 to 80 words)"
        answers = [(r["Your name or initials"].strip(), r[col].strip()) for _, r in form.iterrows() if r[col].strip()]
        slots = refs.index[refs["query"] == query].tolist()
        if len(answers) > len(slots):
            raise ValueError(f"Topic {letter}: {len(answers)} summaries but only {len(slots)} writer slots")
        for i, (writer, text) in zip(slots, answers):
            refs.loc[i, ["reference_summary", "written_by"]] = [text, writer]
        notes.append(f"Topic {letter} ({query}): {len(answers)} reference summar{'y' if len(answers) == 1 else 'ies'}"
                     + "".join(f"; {w}: {len(t.split())} words" for w, t in answers))
    refs.to_csv(config.REFERENCE_SUMMARIES_FILE, index=False, encoding="utf-8-sig")
    return notes


def import_ratings() -> list[str]:
    """One rating_<rater>.csv per response row, in the format of rating_sheet_TEMPLATE.csv."""
    form = read(EVAL2)
    template = read(config.HUMAN_RATINGS_DIR / "rating_sheet_TEMPLATE.csv")
    pattern = re.compile(r"^Item (I\d+): .*\[(Readability|Informativeness|Flow)\]$")
    notes = []
    for _, row in form.iterrows():
        sheet = template.copy()
        n = 0
        for col, value in row.items():
            m = pattern.match(col)
            if m and value.strip():
                sheet.loc[sheet["item_id"] == m.group(1), m.group(2).lower()] = value.strip()
                n += 1
        comments = [v for c, v in row.items() if c.startswith("Comments on this page") and v.strip()]
        name = row["Your name or initials"].strip()
        out = config.HUMAN_RATINGS_DIR / f"rating_{rater_id(name)}.csv"
        sheet.to_csv(out, index=False, encoding="utf-8-sig")
        notes.append(f"{name} -> {out.name}: {n} scores ({n // 3} items), {len(comments)} page comment(s)")
    return notes


def import_sus() -> list[str]:
    form = read(EVAL3)
    consented = form[form["Consent"] == "I agree to take part"]
    items = [c for c in form.columns if re.match(r"^\d+\. ", c)]
    out = pd.DataFrame({"participant": consented.iloc[:, 1].str.strip()})
    for i, c in enumerate(items, start=1):
        out[f"q{i}"] = consented[c].str.strip()
    out.to_csv(config.SUS_FILE, index=False, encoding="utf-8-sig")
    return [f"{len(form)} responses, {len(consented)} with consent -> {config.SUS_FILE.name}"]


def main() -> None:
    for title, fn, path in [("Reference summaries", import_references, EVAL1),
                            ("Human ratings", import_ratings, EVAL2),
                            ("SUS", import_sus, EVAL3)]:
        print(title)
        if not path.exists():
            print(f"    not found: {path.relative_to(config.ROOT)}")
            continue
        for line in fn():
            print("    " + line)


if __name__ == "__main__":
    main()
