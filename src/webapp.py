"""Shared helpers for the web app pages (app_pages/).

Models and data are loaded once and cached, so they are not reloaded on
every click or for every user.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402
from embed import load_embeddings, load_model  # noqa: E402
from llm import chat, load_api_key  # noqa: E402
from summarize import load_summarizer  # noqa: E402


@st.cache_data
def load_data() -> tuple[pd.DataFrame, dict, dict]:
    """Cleaned responses, their embeddings and the Stage 3 retrieval settings."""
    clean = pd.read_csv(config.PROCESSED_DIR / "clean_responses.csv", encoding="utf-8-sig",
                        dtype=str, keep_default_na=False).set_index("response_id")
    store = load_embeddings()
    settings = json.loads(config.RETRIEVAL_SETTINGS_FILE.read_text(encoding="utf-8"))
    return clean, store, settings


@st.cache_resource(show_spinner="Loading the embedding model ...")
def get_embedder(name: str):
    return load_model(name)


@st.cache_resource(show_spinner="Loading FLAN-T5 (first time only, may take a minute) ...")
def get_summarizer():
    return load_summarizer()


def llama_chat(messages, temperature, max_tokens):
    return chat(messages, temperature, max_tokens, api_key=load_api_key())
