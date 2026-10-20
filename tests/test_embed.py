"""Tests for Stage 2. They use a tiny stand-in model so they run offline,
without downloading all-mpnet-base-v2. Run with:  pytest -q"""
import re
import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from embed import (build_check, embed_query, embed_responses,  # noqa: E402
                   load_embeddings, save_embeddings)


class FakeModel:
    """Stand-in for SentenceTransformer: a bag-of-words vector where each word
    is hashed into one of `dim` slots. Texts sharing words get similar vectors."""

    def __init__(self, dim: int = 64):
        self.dim = dim

    def encode(self, texts, batch_size=32, normalize_embeddings=False,
               convert_to_numpy=True, show_progress_bar=False):
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            for w in re.findall(r"[a-z]+", t.lower()):
                out[i, zlib.crc32(w.encode()) % self.dim] += 1
        if normalize_embeddings:
            norms = np.linalg.norm(out, axis=1, keepdims=True)
            out = out / np.where(norms == 0, 1, norms)
        return out


def _data():
    clean = pd.DataFrame({
        "response_id": ["A", "B", "C", "D"],
        "issue_area": ["Health", "Health", "Transport", "Health"],
        "needs_translation": ["False", "False", "False", "True"],
    })
    sentences = pd.DataFrame({
        "response_id": ["A", "A", "B", "C", "D"],
        "sentence_no": [1, 2, 1, 1, 1],
        "sentence": ["Hospital queue was long.", "Doctors were busy.",
                     "Long hospital queue again.", "Buses are always late.",
                     "රෝහලේ පෝලිම දිගයි"],
    })
    return clean, sentences


def test_vectors_are_mean_pooled_and_unit_length():
    clean, sentences = _data()
    ids, matrix, skipped = embed_responses(FakeModel(), clean, sentences)
    assert ids == ["A", "B", "C"]
    assert skipped == ["D"], "responses needing translation are skipped"
    assert np.allclose(np.linalg.norm(matrix, axis=1), 1.0)

    # A has two sentences: its vector is the re-normalized mean of both
    s = FakeModel().encode(["Hospital queue was long.", "Doctors were busy."], normalize_embeddings=True)
    expected = s.mean(axis=0) / np.linalg.norm(s.mean(axis=0))
    assert np.allclose(matrix[0], expected, atol=1e-6)


def test_similar_texts_score_higher():
    clean, sentences = _data()
    ids, matrix, _ = embed_responses(FakeModel(), clean, sentences)
    q = embed_query(FakeModel(), "hospital queue")
    scores = matrix @ q
    assert scores[ids.index("B")] > scores[ids.index("C")]


def test_save_load_and_check(tmp_path):
    clean, sentences = _data()
    ids, matrix, skipped = embed_responses(FakeModel(), clean, sentences)
    path = tmp_path / "emb.pkl"
    save_embeddings(path, ids, matrix, "fake", synthetic=True)
    store = load_embeddings(path)
    assert store["response_ids"] == ids and np.allclose(store["embeddings"], matrix)

    report = build_check(ids, matrix, skipped, clean, "fake", synthetic=True)
    assert "SYNTHETIC" in report and "Responses embedded:      3" in report
