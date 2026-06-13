"""Semantic embedding utilities backed by sentence-transformers.

Lazy-loads the model on first use so smoke.py (zero GPU/ML deps) is unaffected.
On AMD MI300X with ROCm torch installed, device is picked up automatically via
torch.cuda.is_available().

Configuration:
  EMBED_MODEL — HuggingFace model id (default: BAAI/bge-small-en-v1.5)
                33M params, <200 MB, fast on CPU and ROCm GPU alike.
"""
from __future__ import annotations

import os

import numpy as np

_MODEL_NAME = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
_model = None  # lazy-loaded


def _get_model():
    global _model
    if _model is None:
        import torch
        from sentence_transformers import SentenceTransformer
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _model = SentenceTransformer(_MODEL_NAME, device=device)
    return _model


def embed(texts: list[str]) -> np.ndarray:
    """Return L2-normalised embeddings; shape (N, D), dtype float32."""
    return _get_model().encode(texts, normalize_embeddings=True)


def top_k_similar(query: str, candidates: list[str], k: int = 5) -> list[int]:
    """Indices of top-k candidates ranked by cosine similarity to `query`."""
    if not candidates:
        return []
    vecs = embed([query] + candidates)
    scores = vecs[1:] @ vecs[0]
    k = min(k, len(candidates))
    return list(np.argsort(scores)[::-1][:k])


def cosine_scores(query: str, candidates: list[str]) -> np.ndarray:
    """Cosine similarity of every candidate against `query`; shape (N,)."""
    if not candidates:
        return np.array([], dtype=np.float32)
    vecs = embed([query] + candidates)
    return (vecs[1:] @ vecs[0]).astype(np.float32)
