"""Stage 2: Embeddings.

Turns every cleaned response into one vector with the sentence transformer
all-mpnet-base-v2. Each sentence is embedded separately, then the sentence
vectors of a response are averaged (mean pooling) and scaled back to unit
length. Unit-length vectors mean cosine similarity is a simple dot product
in Stage 3.

Run from the project folder (after Stage 1):
    python src/embed.py

Inputs (from data/processed/):
    clean_responses.csv, sentences.csv
Outputs (in data/processed/):
    embeddings.pkl         response ids + one vector per response
    embedding_check.txt    sanity checks
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402


def load_model(name: str = config.EMBEDDING_MODEL):
    """Load the sentence transformer. Imported here so the rest of the file
    (and the tests) work without PyTorch installed."""
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(name)


def encode(model, texts: list[str]) -> np.ndarray:
    """Embed a list of texts. normalize_embeddings=True gives unit vectors."""
    vectors = model.encode(texts, batch_size=config.EMBED_BATCH_SIZE,
                           normalize_embeddings=True, convert_to_numpy=True,
                           show_progress_bar=len(texts) > 100)
    return np.asarray(vectors, dtype=np.float32)


def unit_length(matrix: np.ndarray) -> np.ndarray:
    """Scale each row to length 1 (rows of zeros are left as they are)."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def embed_responses(model, clean: pd.DataFrame, sentences: pd.DataFrame
                    ) -> tuple[list[str], np.ndarray, list[str]]:
    """Return (response_ids, matrix, skipped_ids).

    Responses flagged needs_translation are skipped: an English model would
    give them meaningless vectors. Translate them and rerun Stage 1 first.
    """
    needs_tr = clean["needs_translation"].astype(str).str.lower() == "true"
    skipped = clean.loc[needs_tr, "response_id"].tolist()
    keep_ids = clean.loc[~needs_tr, "response_id"].tolist()

    sents = sentences[sentences["response_id"].isin(keep_ids)].reset_index(drop=True)
    sent_vectors = encode(model, sents["sentence"].tolist())

    # Mean pooling: average the sentence vectors of each response
    ids, rows = [], []
    for rid in keep_ids:
        idx = sents.index[sents["response_id"] == rid]
        if len(idx) == 0:
            skipped.append(rid)
            continue
        ids.append(rid)
        rows.append(sent_vectors[idx].mean(axis=0))
    matrix = unit_length(np.vstack(rows)) if rows else np.zeros((0, 0), dtype=np.float32)
    return ids, matrix.astype(np.float32), skipped


def embed_query(model, text: str) -> np.ndarray:
    """Embed a user's keywords as one unit vector (used by Stage 3 and the app)."""
    return unit_length(encode(model, [text]))[0]


def save_embeddings(path: Path, ids: list[str], matrix: np.ndarray,
                    model_name: str, synthetic: bool) -> None:
    with open(path, "wb") as f:
        pickle.dump({"response_ids": ids, "embeddings": matrix,
                     "model_name": model_name, "synthetic": synthetic}, f)


def load_embeddings(path: Path = config.EMBEDDINGS_FILE) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def build_check(ids: list[str], matrix: np.ndarray, skipped: list[str],
                clean: pd.DataFrame, model_name: str, synthetic: bool) -> str:
    """Simple checks that the vectors make sense. If the model works, responses
    from the same issue area should be more similar to each other than to
    responses from other areas."""
    lines = []
    add = lines.append
    add("EMBEDDING CHECK")
    add("=" * 60)
    if synthetic:
        add("WARNING: this input is SYNTHETIC test data. Do not report these")
        add("numbers as results; rerun on the real questionnaire export.")
    add(f"Model:                   {model_name}")
    add(f"Responses embedded:      {len(ids)}")
    add(f"Skipped (need translation or no sentences): {len(skipped)}")
    if skipped:
        add("    " + ", ".join(skipped))
    if len(ids) == 0:
        return "\n".join(lines)
    norms = np.linalg.norm(matrix, axis=1)
    add(f"Vector dimension:        {matrix.shape[1]}")
    add(f"Vector length min/max:   {norms.min():.4f} / {norms.max():.4f} (should be 1.0000)")

    sims = matrix @ matrix.T
    areas = clean.set_index("response_id").loc[ids, "issue_area"].to_numpy()
    same = areas[:, None] == areas[None, :]
    off_diag = ~np.eye(len(ids), dtype=bool)
    within = sims[same & off_diag]
    between = sims[~same]
    add("")
    add("Cosine similarity between responses")
    add("-" * 60)
    if within.size:
        add(f"    Same issue area:       mean {within.mean():.3f}  (sd {within.std():.3f})")
    if between.size:
        add(f"    Different issue area:  mean {between.mean():.3f}  (sd {between.std():.3f})")
    add("    The first value should be clearly higher than the second.")

    # Nearest neighbour: how often is the most similar other response
    # from the same issue area?
    nn_sims = np.where(off_diag, sims, -np.inf)
    nn = nn_sims.argmax(axis=1)
    hit = (areas[nn] == areas).mean() if len(ids) > 1 else float("nan")
    add(f"    Nearest neighbour in same issue area: {hit:.1%} of responses")
    return "\n".join(lines)


def main() -> None:
    clean = pd.read_csv(config.PROCESSED_DIR / "clean_responses.csv", encoding="utf-8-sig",
                        dtype=str, keep_default_na=False)
    sentences = pd.read_csv(config.PROCESSED_DIR / "sentences.csv", encoding="utf-8-sig",
                            dtype=str, keep_default_na=False)
    synthetic = clean["data_source"].str.contains("SYNTHETIC").any()

    print(f"Loading {config.EMBEDDING_MODEL} (downloads on first run) ...")
    model = load_model()
    ids, matrix, skipped = embed_responses(model, clean, sentences)

    save_embeddings(config.EMBEDDINGS_FILE, ids, matrix, config.EMBEDDING_MODEL, synthetic)
    report = build_check(ids, matrix, skipped, clean, config.EMBEDDING_MODEL, synthetic)
    (config.PROCESSED_DIR / "embedding_check.txt").write_text(report, encoding="utf-8")
    print(report)
    print(f"\nSaved {config.EMBEDDINGS_FILE}")


if __name__ == "__main__":
    main()
