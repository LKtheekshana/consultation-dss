"""Functional test run of the web application on the real data (evidence for
the evaluation).

Drives the Streamlit app without a browser (Streamlit AppTest), using the
real models and the Llama API, and writes the result of every test case to
data/processed/functional_test_report.txt.

Run from the project folder (needs the .env key and internet for Llama):
    python functional_test_run.py

Submissions made during the run are removed afterwards, so the office inbox
is left exactly as it was.
"""
from __future__ import annotations

import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
import config  # noqa: E402
import llm  # noqa: E402

APP = str(ROOT / "app.py")
results = []


def record(case, requirement, action, expected, passed, observed):
    results.append((case, requirement, action, expected, "Pass" if passed else "Fail", observed))
    print(f"{case} {'PASS' if passed else 'FAIL'}: {observed}")


def button(at, label):
    return next(b for b in at.button if b.label == label)


def main() -> None:
    sub_file = config.SUBMISSIONS_FILE
    backup = sub_file.with_suffix(".backup")
    had_file = sub_file.exists()
    if had_file:
        shutil.copy(sub_file, backup)
    try:
        run_cases()
    finally:
        if had_file:
            shutil.move(backup, sub_file)
        elif sub_file.exists():
            sub_file.unlink()

    lines = ["FUNCTIONAL TEST REPORT", "=" * 100,
             f"Run: {datetime.now():%Y-%m-%d %H:%M}; data: {config.INPUT_FILE.name} (real questionnaire data)",
             f"Passed: {sum(r[4] == 'Pass' for r in results)} of {len(results)}", ""]
    for case, req, action, expected, status, observed in results:
        lines += [f"{case} [{req}] {status}", f"    Action:   {action}", f"    Expected: {expected}",
                  f"    Observed: {observed}", ""]
    out = config.PROCESSED_DIR / "functional_test_report.txt"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nSaved {out}")


def run_cases() -> None:
    at = AppTest.from_file(APP, default_timeout=900).run()
    assert not at.exception, at.exception

    # TC01 search
    at.text_input[0].input("hospital waiting time")
    at.button[0].click().run()
    count = next((m.value for m in at.markdown if "related submission(s)" in m.value), "no count")
    hits = [m.value for m in at.markdown if "similarity **" in m.value]
    # the Next button at the end of step 1 should open step 2
    button(at, "Next: read a summary").click().run()
    step = at.session_state["step"]
    record("TC01", "FR1, FR2", "Search with the keywords 'hospital waiting time', then press Next",
           "Related submissions listed with issue area and similarity score; Next opens step 2",
           bool(hits) and step.startswith("2.") and not at.exception,
           f"{count.split(' from')[0].strip('*')}; first: {hits[0] if hits else '-'}; after Next: '{step}'")

    # TC02 weak-match warning for vague keywords
    at.text_input[0].input("travel price")
    at.button[0].click().run()
    warned = [w.value for w in at.warning if "No closely matching" in w.value]
    record("TC02", "FR2", "Search with the vague keywords 'travel price'",
           "Closest submissions listed with a warning that none match closely",
           bool(warned) and not at.exception, warned[0][:110] + "..." if warned else "no warning")
    at.text_input[0].input("hospital waiting time")
    at.button[0].click().run()

    # TC03 summary with the offline backup (Llama made unavailable by hiding the key)
    real_key = llm.load_api_key
    llm.load_api_key = lambda: None
    try:
        start = time.perf_counter()
        button(at, "Summarize").click().run()
        secs = time.perf_counter() - start
    finally:
        llm.load_api_key = real_key
    backup = any("offline backup" in w.value for w in at.warning)
    summary = [i.value for i in at.info if i.value]
    record("TC03", "FR4, FR5", "Summarize when Llama is unavailable (no API key)",
           "Summary written by the offline backup model, with a notice",
           backup and bool(summary) and not at.exception, f"{secs:.1f} s; '{summary[0][:90]}...'" if summary else "-")

    # TC04 summary with Llama
    start = time.perf_counter()
    button(at, "Summarize").click().run()
    secs = time.perf_counter() - start
    summary = [i.value for i in at.info if i.value]
    by_llama = any("AI-generated summary" in m.value and config.LLAMA_MODEL in m.value for m in at.markdown)
    record("TC04", "FR4, FR5", "Summarize the related submissions",
           "AI-labelled summary by Llama with the list of source submissions",
           bool(summary) and by_llama and not at.exception, f"{secs:.1f} s; '{summary[0][:90]}...'")

    # TC05 few-shot draft
    button(at, "Generate AI draft").click().run()
    problem = at.text_area(key="draft_problem").value
    suggestion = at.text_area(key="draft_suggestion").value
    record("TC05", "FR6", "Generate an AI draft",
           "Problem and suggested improvement filled in; draft labelled AI-generated",
           bool(problem.strip()) and bool(suggestion.strip()) and not at.exception,
           f"problem: '{problem[:70]}...'")

    # TC06 confirmation required
    at.text_area(key="draft_problem").input(at.text_area(key="draft_problem").value + " This happened to my father last month.")
    at.checkbox(key="draft_confirm").uncheck().run()
    button(at, "Submit to consultation office").click().run()
    errors = [e.value for e in at.error]
    record("TC06", "FR7, FR8", "Edit the draft and submit without ticking the confirmation box",
           "Submission blocked with a message", any("tick the box" in e for e in errors), "; ".join(errors))

    # TC07 personal details blocked
    good = at.text_area(key="draft_problem").value
    at.text_area(key="draft_problem").input(good + " Call me on 077 123 4567.")
    at.checkbox(key="draft_confirm").check().run()
    button(at, "Submit to consultation office").click().run()
    errors = [e.value for e in at.error]
    record("TC07", "FR8", "Submit a text containing a phone number",
           "Submission blocked until the number is removed", any("phone number" in e for e in errors), "; ".join(errors))

    # TC08 too short
    at.text_area(key="draft_problem").input("Long wait.")
    at.text_area(key="draft_suggestion").input("")
    at.checkbox(key="draft_confirm").check().run()
    button(at, "Submit to consultation office").click().run()
    errors = [e.value for e in at.error]
    record("TC08", "FR8", "Submit a text shorter than eight words",
           "Submission blocked", any("at least" in e for e in errors), "; ".join(errors))

    # TC09 valid submission
    at.text_area(key="draft_problem").input(good)
    at.text_area(key="draft_suggestion").input("Introduce a token system so patients know their time.")
    at.checkbox(key="draft_confirm").check().run()
    button(at, "Submit to consultation office").click().run()
    success = [s.value for s in at.success]
    cleared = at.text_area(key="draft_problem").value == ""
    has_receipt = any(getattr(b, "label", "") == "Download your copy" for b in at.get("download_button"))
    record("TC09", "FR9, FR10", "Submit a valid, confirmed submission",
           "Reference number shown, form cleared, copy can be downloaded",
           bool(success) and cleared and has_receipt, f"{success[0] if success else '-'}; form cleared: {cleared}")

    # TC10 office inbox
    at.switch_page("app_pages/office.py").run()
    rows = len(at.dataframe[0].value) if at.dataframe else 0
    metrics = {m.label: m.value for m in at.metric}
    record("TC10", "FR11", "Open the office inbox",
           "The new submission is listed with summary figures and CSV export",
           rows >= 1 and not at.exception, f"{rows} submission(s); {metrics}")


if __name__ == "__main__":
    main()
