"""Turn text into unit-length vectors.

Three interchangeable backends:
  * SentenceTransformerEmbedder - local MiniLM model (default, free, no API key)
  * GeminiEmbedder              - Gemini embedding API (optional)
  * HashEmbedder                - tiny offline embedder used for tests/CI. It is
                                  NOT a semantic model: it only knows about
                                  shared words, so use MiniLM for real numbers.

All embedders return float32 arrays of shape (n, dim), L2-normalised, so the
dot product of two vectors is their cosine similarity.
"""
from __future__ import annotations

import logging
import re
import zlib
from typing import Protocol

import numpy as np

log = logging.getLogger("semcache.embed")


class Embedder(Protocol):
    name: str
    dim: int

    def embed(self, texts: list[str]) -> np.ndarray: ...


def _normalise(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    return (mat / np.maximum(norms, 1e-9)).astype(np.float32)


# --------------------------------------------------------------------------
class HashEmbedder:
    """Deterministic hashing-trick embedder (words + bigrams + char trigrams)."""

    _WORD = re.compile(r"[a-z0-9]+")
    _STOP = {
        "a", "an", "the", "is", "are", "was", "were", "of", "in", "on", "to", "for",
        "and", "or", "do", "does", "did", "i", "me", "my", "you", "can", "how", "what",
        "which", "tell", "please", "it", "its", "s", "be", "with", "this", "that",
    }

    def __init__(self, dim: int = 512):
        self.dim = dim
        self.name = f"hash-{dim}"

    def _add(self, v: np.ndarray, key: str, weight: float) -> None:
        h = zlib.crc32(key.encode("utf-8"))
        v[h % self.dim] += weight if (h >> 16) & 1 else -weight

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            words = self._WORD.findall(text.lower())
            for w in words:
                wt = 0.3 if w in self._STOP else 1.0
                self._add(out[row], "w:" + w, wt)
                padded = f"^{w}$"
                for i in range(len(padded) - 2):
                    self._add(out[row], "c:" + padded[i : i + 3], 0.25 * wt)
            for a, b in zip(words, words[1:]):
                both_stop = a in self._STOP and b in self._STOP
                self._add(out[row], f"b:{a}_{b}", 0.15 if both_stop else 0.5)
        return _normalise(out)


# --------------------------------------------------------------------------
class SentenceTransformerEmbedder:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer  # lazy: heavy import

        self.model = SentenceTransformer(model_name)
        # Probe with one sentence: works on every sentence-transformers version
        # (the "get dimension" method was renamed in v5/v6).
        self.dim = int(self.model.encode(["probe"], convert_to_numpy=True,
                                         show_progress_bar=False).shape[1])
        self.name = model_name

    def embed(self, texts: list[str]) -> np.ndarray:
        vecs = self.model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
            batch_size=64,
        )
        return np.asarray(vecs, dtype=np.float32)


# --------------------------------------------------------------------------
class GeminiEmbedder:
    """Gemini embedding API via plain REST (no SDK, so no version churn)."""

    def __init__(self, api_key: str, model: str = "gemini-embedding-001",
                 base_url: str = "https://generativelanguage.googleapis.com/v1beta",
                 output_dim: int = 768):
        import httpx

        if not api_key:
            raise ValueError("GEMINI_API_KEY is required for EMBEDDER=gemini")
        self.model, self.dim, self.name = model, output_dim, f"gemini:{model}:{output_dim}"
        self._url = f"{base_url}/models/{model}:batchEmbedContents"
        self._client = httpx.Client(timeout=30.0, headers={"x-goog-api-key": api_key})

    def embed(self, texts: list[str]) -> np.ndarray:
        out: list[list[float]] = []
        for i in range(0, len(texts), 50):
            chunk = texts[i : i + 50]
            body = {"requests": [{
                "model": f"models/{self.model}",
                "content": {"parts": [{"text": t}]},
                "taskType": "SEMANTIC_SIMILARITY",
                "outputDimensionality": self.dim,
            } for t in chunk]}
            r = self._client.post(self._url, json=body)
            r.raise_for_status()
            out.extend(e["values"] for e in r.json()["embeddings"])
        return _normalise(np.asarray(out, dtype=np.float32))


# --------------------------------------------------------------------------
def make_embedder(kind: str, model: str = "", api_key: str = "",
                  gemini_model: str = "gemini-embedding-001") -> Embedder:
    kind = kind.lower()
    if kind == "hash":
        return HashEmbedder()
    if kind == "gemini":
        return GeminiEmbedder(api_key, gemini_model)
    if kind in ("minilm", "sentence-transformers", "st"):
        try:
            return SentenceTransformerEmbedder(model or "sentence-transformers/all-MiniLM-L6-v2")
        except Exception as exc:  # not installed / model download failed
            raise RuntimeError(
                "Could not load the sentence-transformers model "
                f"({exc}). Run `pip install -r requirements.txt` and make sure you are "
                "online for the first run (the model downloads once, ~90 MB). "
                "To run without it, set EMBEDDER=hash in .env (demo only, not semantic)."
            ) from exc
    raise ValueError(f"Unknown embedder '{kind}' (use minilm | gemini | hash)")
