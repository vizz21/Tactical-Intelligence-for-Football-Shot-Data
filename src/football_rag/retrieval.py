"""Bi-encoder retrieval over the shot corpus: exact top-k by matrix multiply (10k vectors, no FAISS)."""
import json
from dataclasses import dataclass

import numpy as np
from sentence_transformers import SentenceTransformer

from .config import CORPUS, ENCODER_DIR, INDEX_DIR, MAX_SEQ_LENGTH

ATTRS = ("team", "body", "side", "area", "outcome")


def load_corpus() -> list[dict]:
    """Corpus rows in file order; row_idx must equal position (same row order as tabular.csv)."""
    with open(CORPUS, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    assert all(r["row_idx"] == i for i, r in enumerate(rows)), "corpus row_idx is not 0..N-1 in order"
    return rows


def load_encoder(device: str = "cpu") -> SentenceTransformer:
    enc = SentenceTransformer(str(ENCODER_DIR), device=device)
    enc.max_seq_length = MAX_SEQ_LENGTH
    return enc


def embed(enc: SentenceTransformer, texts: list[str], batch_size: int = 64) -> np.ndarray:
    return enc.encode(texts, batch_size=batch_size, normalize_embeddings=True, convert_to_numpy=True,
                      show_progress_bar=False).astype("float32")


def is_relevant(query_attrs: dict, doc_attrs: dict) -> bool:
    """Phase 3 definition: a document is relevant if it satisfies EVERY attribute the query asks for."""
    return all(doc_attrs.get(k) == v for k, v in query_attrs.items())


@dataclass
class Index:
    embeddings: np.ndarray      # (N, 384) float32, L2-normalised, row i = corpus row i
    shot_ids: list[str]
    row_idx: np.ndarray

    def similarities(self, q_emb: np.ndarray) -> np.ndarray:
        """(n_queries, N) cosine similarities (both sides are normalised, so a dot product)."""
        return np.atleast_2d(q_emb) @ self.embeddings.T

    def topk(self, q_emb: np.ndarray, k: int):
        """(indices, scores), each (n_queries, k), best first."""
        sims = self.similarities(q_emb)
        k = min(k, sims.shape[1])
        part = np.argpartition(-sims, k - 1, axis=1)[:, :k]
        order = np.take_along_axis(sims, part, 1).argsort(axis=1)[:, ::-1]
        idx = np.take_along_axis(part, order, 1)
        return idx, np.take_along_axis(sims, idx, 1)


def load_index() -> Index:
    emb = np.load(INDEX_DIR / "embeddings.npy")
    ids = json.loads((INDEX_DIR / "ids.json").read_text(encoding="utf-8"))
    return Index(emb, [i["shot_id"] for i in ids], np.array([i["row_idx"] for i in ids]))
