"""Office inbox: submissions received through the citizen portal (prototype)."""
from __future__ import annotations

import pandas as pd
import streamlit as st

import config
from submissions import load_submissions, receipt_text

st.title("Office inbox")
st.caption("Submissions sent through the citizen portal of this prototype. In a real "
           "deployment this page would require a staff login, and submissions would go to "
           "the Information Center's own systems.")

subs = load_submissions()
if subs.empty:
    st.info("No submissions yet. They appear here after a citizen presses "
            "**Submit to consultation office**.", icon=":material/inbox:")
    st.stop()

# Values are read back from the CSV as text
subs["ai_assisted"] = subs["ai_assisted"] == "True"
subs["edited"] = subs["edited"] == "True"
subs["edit_ratio"] = pd.to_numeric(subs["edit_ratio"], errors="coerce")
subs["submitted_at"] = pd.to_datetime(subs["submitted_at"])
subs = subs.sort_values("submitted_at", ascending=False).reset_index(drop=True)

# ---- Overview ----
ai = subs[subs["ai_assisted"]]
with st.container(horizontal=True):
    st.metric("Submissions", len(subs), border=True)
    st.metric("AI-assisted", f"{len(ai) / len(subs):.0%}", border=True,
              help="Started from an AI draft")
    st.metric("AI drafts sent unchanged", f"{(~ai['edited']).mean():.0%}" if len(ai) else "-",
              border=True, help="Share of AI-assisted submissions the citizen did not edit")
    st.metric("Mean share of AI draft changed", f"{ai['edit_ratio'].mean():.0%}" if len(ai) else "-",
              border=True, help="0% = sent as drafted, 100% = completely rewritten (word level)")

# ---- Filter and list ----
areas = [a for a in config.ISSUE_AREAS if a in set(subs["issue_area"])]
chosen = st.pills("Issue area", areas, selection_mode="multi", default=areas)
shown = subs[subs["issue_area"].isin(chosen)].reset_index(drop=True)

event = st.dataframe(
    shown,
    column_order=["reference", "submitted_at", "issue_area", "keywords", "problem",
                  "ai_assisted", "edited", "edit_ratio"],
    column_config={
        "reference": "Reference",
        "submitted_at": st.column_config.DatetimeColumn("Submitted", format="YYYY-MM-DD HH:mm"),
        "issue_area": "Issue area",
        "keywords": "Keywords",
        "problem": st.column_config.TextColumn("Problem", width="large"),
        "ai_assisted": st.column_config.CheckboxColumn("AI-assisted"),
        "edited": st.column_config.CheckboxColumn("Edited"),
        "edit_ratio": st.column_config.ProgressColumn("Changed", format="percent",
                                                      min_value=0, max_value=1),
    },
    hide_index=True, on_select="rerun", selection_mode="single-row", key="inbox_table",
    alt="Table of submissions received by the consultation office",
)

st.download_button("Download as CSV", shown.to_csv(index=False).encode("utf-8-sig"),
                   file_name="submissions.csv", mime="text/csv", on_click="ignore",
                   icon=":material/download:")

# ---- One submission in full ----
selected = event.selection.rows
if not selected:
    st.caption("Select a row to read the full submission.")
else:
    row = shown.iloc[selected[0]].to_dict()
    with st.container(border=True):
        st.subheader(row["reference"])
        st.caption(f"{row['submitted_at']:%Y-%m-%d %H:%M} · {row['issue_area']} · "
                   f"keywords: {row['keywords'] or '-'}")
        st.markdown(f"**Problem:** {row['problem']}")
        st.markdown(f"**Suggested improvement:** {row['suggestion'] or '-'}")
        if row["ai_assisted"]:
            with st.expander("Original AI draft"):
                st.caption(f"{row['model']} · {row['prompt_type']}"
                           + (f" · examples: {row['example_ids']}" if row["example_ids"] else ""))
                st.markdown(f"**Problem:** {row['ai_problem']}")
                st.markdown(f"**Suggested improvement:** {row['ai_suggestion'] or '-'}")
        receipt = receipt_text({**row, "submitted_at": f"{row['submitted_at']:%Y-%m-%d %H:%M}"})
        st.download_button("Download this submission", receipt, file_name=f"{row['reference']}.txt",
                           mime="text/plain", on_click="ignore", icon=":material/download:")
