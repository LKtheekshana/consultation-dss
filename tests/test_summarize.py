"""Tests for Stage 4. They use stand-ins for the tokenizer and the model,
so they run offline without downloading FLAN-T5. Run with:  pytest -q"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import config  # noqa: E402
from summarize import format_responses, pack_groups, summarize_texts, summary_input  # noqa: E402


def count_words(text):
    """Stand-in tokenizer: one token per word."""
    return len(text.split())


class FakeGenerator:
    """Stand-in model: remembers every input and returns a short 'summary'."""

    def __init__(self):
        self.inputs = []

    def __call__(self, prompt):
        self.inputs.append(prompt)
        return f"summary {len(self.inputs)}"


def test_short_input_is_summarized_in_one_step():
    gen = FakeGenerator()
    summary, info = summarize_texts(["Bus is late.", "Train is crowded."], gen, count_words)
    assert summary == "summary 1"
    assert info["stages"] == 1 and info["groups"] == 1 and not info["truncated"]
    assert gen.inputs[0].startswith(config.SUMMARY_PROMPT)
    assert "- Bus is late.\n- Train is crowded." in gen.inputs[0]


def test_no_input_gives_empty_summary():
    summary, info = summarize_texts([], FakeGenerator(), count_words)
    assert summary == "" and info["n_used"] == 0


def test_number_of_responses_is_capped():
    texts = [f"Response {i} about buses." for i in range(config.SUMMARY_MAX_RESPONSES + 5)]
    _, info = summarize_texts(texts, FakeGenerator(), count_words)
    assert info["n_retrieved"] == config.SUMMARY_MAX_RESPONSES + 5
    assert info["n_used"] == config.SUMMARY_MAX_RESPONSES


def test_long_input_is_summarized_in_stages_and_every_input_fits():
    # 10 responses of 150 words each cannot fit in 512 tokens together
    texts = [" ".join(["word"] * 150) for _ in range(10)]
    gen = FakeGenerator()
    summary, info = summarize_texts(texts, gen, count_words)
    assert info["groups"] > 1 and info["stages"] == 2
    assert all(count_words(p) <= config.T5_TOKEN_LIMIT for p in gen.inputs)
    assert gen.inputs[-1].startswith(config.SUMMARY_COMBINE_PROMPT)
    assert summary == f"summary {len(gen.inputs)}"


def test_pack_groups_keeps_order():
    groups = pack_groups(["a b", "c d", "e f"], count_words, budget=5)
    assert [t for g in groups for t in g] == ["a b", "c d", "e f"]
    assert all(count_words(format_responses(g)) <= 5 for g in groups)


def test_single_overlong_response_is_flagged():
    _, info = summarize_texts([" ".join(["word"] * 700)], FakeGenerator(), count_words)
    assert info["truncated"]


def test_summary_input_leaves_out_label():
    row = {"problem_description": "The bus is late.", "suggested_improvement": "Add more buses."}
    assert summary_input(row) == "The bus is late. Add more buses."
    assert summary_input({"problem_description": "The bus is late.", "suggested_improvement": ""}) == "The bus is late."
