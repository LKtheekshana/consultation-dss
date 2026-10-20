"""Tests for Stage 5 and the API client. They use fake API responses, so
they run offline without an API key. Run with:  pytest -q"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import config  # noqa: E402
import llm  # noqa: E402
from generate import (build_messages, choose_examples, generate_submission,  # noqa: E402
                      parse_submission, run_holdout, summarize_with_llama)


class FakeChat:
    """Stand-in for the Llama API: records the messages and returns a fixed answer."""

    def __init__(self, answer="Problem: The bus is late every day.\nSuggested improvement: Add more buses."):
        self.answer, self.calls = answer, []

    def __call__(self, messages, temperature, max_tokens):
        self.calls.append(messages)
        return {"text": self.answer, "prompt_tokens": 100, "completion_tokens": 20, "seconds": 0.1}


def _clean():
    return pd.DataFrame({
        "response_id": ["A", "B", "C"],
        "issue_area": ["Transport", "Transport", "Health"],
        "keywords": ["bus delay", "train crowded", "hospital queue"],
        "problem_description": ["The bus is late.", "The train is crowded.", "Long hospital queue."],
        "suggested_improvement": ["More buses.", "More carriages.", ""],
    }).set_index("response_id")


def _store():
    vectors = np.array([[1.0, 0.0], [0.9, 0.436], [0.0, 1.0]])
    return {"response_ids": ["A", "B", "C"],
            "embeddings": vectors / np.linalg.norm(vectors, axis=1, keepdims=True)}


def test_parse_submission():
    assert parse_submission("Problem: Late bus.\nSuggested improvement: More buses.") == \
        ("Late bus.", "More buses.", True)
    problem, suggestion, ok = parse_submission("Just some text without the format.")
    assert problem == "Just some text without the format." and suggestion == "" and not ok


def test_parse_submission_removes_personal_details():
    problem, _, _ = parse_submission("Problem: Call me on 077 123 4567.\nSuggested improvement: Fix it.")
    assert "077" not in problem and "[phone removed]" in problem


def test_zero_shot_and_few_shot_prompts():
    clean = _clean()
    zero = build_messages("bus delay", [])
    few = build_messages("bus delay", [clean.loc["B"], clean.loc["C"]])
    assert "Example" not in zero[1]["content"] and "Keywords: bus delay" in zero[1]["content"]
    assert "Example 2:" in few[1]["content"] and "The train is crowded." in few[1]["content"]
    # an example without a suggestion has no empty "Suggested improvement:" line
    assert "Long hospital queue.\n\n" in few[1]["content"]


def test_generate_submission():
    out = generate_submission("bus delay", [], FakeChat())
    assert out["parsed_ok"] and out["generated_text"] == "The bus is late every day. Add more buses."


def test_choose_examples_leaves_own_response_out():
    store = _store()
    assert choose_examples(store["embeddings"][0], store, 2, exclude_id="A") == ["B", "C"]
    assert choose_examples(store["embeddings"][0], store, 1) == ["A"]


def test_holdout_run_and_resume():
    clean, store = _clean(), _store()
    vectors = {rid: store["embeddings"][i] for i, rid in enumerate(store["response_ids"])}
    saved = []
    chat = FakeChat()
    gen = run_holdout(clean, store, vectors, chat, pd.DataFrame(), saved.append)
    assert len(gen) == 6 and set(gen["prompt_type"]) == {"zero-shot", "few-shot"}
    few_a = gen[(gen["query_id"] == "A") & (gen["prompt_type"] == "few-shot")].iloc[0]
    assert "A" not in few_a["example_ids"].split("; ")
    assert gen["ai_generated"].all() and len(saved) == 3

    # Running again with the finished rows makes no new API calls
    chat2 = FakeChat()
    again = run_holdout(clean, store, vectors, chat2, gen, saved.append)
    assert len(again) == 6 and chat2.calls == []


def test_llama_summary_uses_same_cap_as_flan():
    chat = FakeChat("A summary.")
    texts = [f"Response {i}." for i in range(config.SUMMARY_MAX_RESPONSES + 3)]
    summary, info = summarize_with_llama(texts, chat)
    assert summary == "A summary." and info["n_used"] == config.SUMMARY_MAX_RESPONSES
    assert f"Response {config.SUMMARY_MAX_RESPONSES}." not in chat.calls[0][1]["content"]


# ---------------------------------------------------------------------------
# API client
# ---------------------------------------------------------------------------
class FakeResponse:
    def __init__(self, status, data=None, headers=None):
        self.status_code, self._data, self.headers, self.text = status, data, headers or {}, ""

    def json(self):
        return self._data


OK = {"choices": [{"message": {"content": " Hello "}}],
      "usage": {"prompt_tokens": 5, "completion_tokens": 1}}


@pytest.fixture(autouse=True)
def fake_account_id(monkeypatch):
    """Cloudflare's API address needs an account ID; the tests use a fake one."""
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "test-account")


def test_api_base_url_fills_in_account_id(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "LLAMA_API_BASE_URL", "https://x/accounts/{account_id}/ai/v1")
    assert llm.api_base_url() == "https://x/accounts/test-account/ai/v1"
    monkeypatch.delenv("CLOUDFLARE_ACCOUNT_ID")
    with pytest.raises(llm.LlamaError, match="account ID"):
        llm.api_base_url(tmp_path / "missing.env")


def test_daily_limit_stops_without_retrying(monkeypatch):
    monkeypatch.setattr(llm.time, "sleep", lambda s: None)
    reply = FakeResponse(429)
    reply.text = '{"errors":[{"code":4006,"message":"you have used up your daily free allocation of 10,000 neurons"}]}'
    calls = []

    def fake_post(*a, **k):
        calls.append(1)
        return reply

    with pytest.raises(llm.LlamaError, match="daily allowance"):
        llm.chat([], 0.2, 10, api_key="k", post=fake_post)
    assert len(calls) == 1


def test_chat_retries_after_rate_limit(monkeypatch):
    monkeypatch.setattr(llm.time, "sleep", lambda s: None)
    replies = [FakeResponse(429, headers={"retry-after": "1"}), FakeResponse(200, OK)]
    sent = []

    def fake_post(url, json, headers, timeout):
        sent.append(json)
        return replies.pop(0)

    out = llm.chat([{"role": "user", "content": "hi"}], 0.2, 10, api_key="k", post=fake_post)
    assert out["text"] == "Hello" and out["prompt_tokens"] == 5 and len(sent) == 2
    assert sent[0]["model"] == config.LLAMA_MODEL


def test_chat_bad_key_and_missing_key(monkeypatch, tmp_path):
    with pytest.raises(llm.LlamaError, match="rejected"):
        llm.chat([], 0.2, 10, api_key="bad", post=lambda *a, **k: FakeResponse(401))
    monkeypatch.delenv("LLAMA_API_KEY", raising=False)
    monkeypatch.setattr(llm, "load_api_key", lambda: None)
    with pytest.raises(llm.LlamaError, match="No API key"):
        llm.chat([], 0.2, 10)


def test_load_api_key_from_env_file(monkeypatch, tmp_path):
    monkeypatch.delenv("LLAMA_API_KEY", raising=False)
    env = tmp_path / ".env"
    env.write_text("# comment\nLLAMA_API_KEY=your-key-here\n", encoding="utf-8")
    assert llm.load_api_key(env) is None, "the placeholder from .env.example is not a key"
    env.write_text('LLAMA_API_KEY="gsk_abc"\n', encoding="utf-8")
    assert llm.load_api_key(env) == "gsk_abc"
