"""Stage 6: Web application (Streamlit).

Two pages:
    Citizen portal  (app_pages/citizen.py)
        1. retrieve related citizen submissions for some keywords (Stage 3),
        2. summarize them with FLAN-T5 or Llama (Stages 4-5),
        3. draft a submission (by hand or with Llama), edit it, download a
           copy and submit it to the consultation office.
    Office inbox    (app_pages/office.py)
        the submissions received, for Information Center staff.

Run from the project folder (after Stages 1-3):
    streamlit run app.py
It opens in the browser at http://localhost:8501
"""
import sys
from pathlib import Path

import streamlit as st

# Let the pages import config.py and the modules in src/
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

st.set_page_config(page_title="Consultation DSS", page_icon=":material/forum:", layout="wide")

page = st.navigation(
    [
        st.Page("app_pages/citizen.py", title="Citizen portal", icon=":material/edit_note:",
                default=True),
        st.Page("app_pages/office.py", title="Office inbox", icon=":material/inbox:"),
    ],
    position="top",
)
page.run()
