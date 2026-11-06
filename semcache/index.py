"""Nearest-neighbour index over unit vectors (inner product == cosine).

Uses FAISS (IndexIDMap2 over IndexFlatIP) when it is installed, and falls back
to a plain NumPy matrix otherwise. Both support add / remove / search, which is
what the cache needs for TTL expiry and LRU eviction.
"""
from __future__ import annotations

import numpy as np


class VectorIndex:
    def __init__(self, dim: int, backend: str = "auto"):
        self.dim = dim
        self._faiss = None
        if backend in ("auto", "faiss"):
            try:
                import faiss

                self._faiss = faiss
                self._index = faiss.IndexIDMap2(faiss.IndexFlatIP(dim))
            except ImportError:
                if backend == "faiss":
                    raise
        if self._faiss is None:
            self._ids: list[int] = []
            self._mat = np.zeros((0, dim), dtype=np.float32)
        self.backend = "faiss" if self._faiss is not None else "numpy"

    def __len__(self) -> int:
        return int(self._index.ntotal) if self._faiss is not None else len(self._ids)

    def add(self, ids: list[int], vectors: np.ndarray) -> None:
        vectors = np.ascontiguousarray(vectors, dtype=np.float32).reshape(len(ids), self.dim)
        if self._faiss is not None:
            self._index.add_with_ids(vectors, np.asarray(ids, dtype=np.int64))
        else:
            self._ids.extend(int(i) for i in ids)
            self._mat = np.vstack([self._mat, vectors])

    def remove(self, ids: list[int]) -> None:
        if not ids:
            return
        if self._faiss is not None:
            self._index.remove_ids(np.asarray(ids, dtype=np.int64))
        else:
            drop = set(int(i) for i in ids)
            keep = [k for k, i in enumerate(self._ids) if i not in drop]
            self._ids = [self._ids[k] for k in keep]
            self._mat = self._mat[keep] if keep else np.zeros((0, self.dim), dtype=np.float32)

    def search(self, vector: np.ndarray, k: int) -> list[tuple[int, float]]:
        n = len(self)
        if n == 0 or k <= 0:
            return []
        k = min(k, n)
        q = np.ascontiguousarray(vector, dtype=np.float32).reshape(1, self.dim)
        if self._faiss is not None:
            scores, ids = self._index.search(q, k)
            return [(int(i), float(s)) for i, s in zip(ids[0], scores[0]) if i != -1]
        sims = self._mat @ q[0]
        top = np.argsort(-sims)[:k]
        return [(self._ids[int(t)], float(sims[t])) for t in top]

    def clear(self) -> None:
        if self._faiss is not None:
            self._index.reset()
        else:
            self._ids, self._mat = [], np.zeros((0, self.dim), dtype=np.float32)
