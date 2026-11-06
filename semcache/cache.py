"""The semantic cache itself.

Lookup pipeline (modes: exact | semantic | semantic+verify):

    normalise -> exact-match dict            (free, ~microseconds)
              -> embed + nearest neighbours   (~10-20 ms with MiniLM)
              -> similarity >= threshold ?
              -> verifier says "same question"?  -> HIT
              otherwise                            -> MISS (caller asks the LLM)

Also handles TTL expiry, LRU eviction, per-namespace isolation (so answers for
different models / system prompts never mix) and persistence to disk.
"""
from __future__ import annotations

import json
import logging
import re
import threading
import time
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .embedders import Embedder
from .index import VectorIndex
from .verifier import NoVerifier

log = logging.getLogger("semcache.cache")

MODES = ("exact", "semantic", "semantic+verify")


def normalize(query: str) -> str:
    q = unicodedata.normalize("NFKC", query).strip().lower()
    q = re.sub(r"\s+", " ", q)
    return q.rstrip("?!.,; ")


@dataclass
class CacheEntry:
    id: int
    namespace: str
    query: str
    norm: str
    answer: str
    created_at: float
    last_access: float
    hits: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    meta: dict[str, Any] = field(default_factory=dict)
    embedding: np.ndarray | None = None


@dataclass
class LookupResult:
    hit: bool
    kind: str                         # exact | semantic | miss
    similarity: float | None = None   # similarity of the served entry
    entry: CacheEntry | None = None
    nearest_similarity: float | None = None  # best candidate even on a miss
    nearest_query: str | None = None
    verifier_rejections: int = 0
    reason: str = ""
    embedding: np.ndarray | None = None
    embed_ms: float = 0.0
    search_ms: float = 0.0
    verify_ms: float = 0.0


class SemanticCache:
    def __init__(self, embedder: Embedder, verifier: NoVerifier, *, threshold: float = 0.85,
                 mode: str = "semantic+verify", ttl_seconds: int = 86400,
                 max_entries: int = 10000, top_k: int = 5, index_backend: str = "auto"):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        self.embedder, self.verifier = embedder, verifier
        self.threshold, self.mode = threshold, mode
        self.ttl, self.max_entries, self.top_k = ttl_seconds, max_entries, top_k
        self._use_vectors = mode != "exact"
        self.index = VectorIndex(embedder.dim, index_backend)
        self._entries: dict[int, CacheEntry] = {}
        self._exact: dict[tuple[str, str], int] = {}
        self._next_id = 1
        self._lock = threading.RLock()
        self._unsaved = 0

    # ---------------------------------------------------------------- helpers
    def __len__(self) -> int:
        return len(self._entries)

    def _expired(self, e: CacheEntry, now: float) -> bool:
        return self.ttl > 0 and now - e.created_at > self.ttl

    def _remove_locked(self, ids: list[int]) -> None:
        gone = []
        for i in ids:
            e = self._entries.pop(i, None)
            if e is not None:
                self._exact.pop((e.namespace, e.norm), None)
                gone.append(i)
        if self._use_vectors:
            self.index.remove(gone)

    def _touch(self, e: CacheEntry, now: float) -> None:
        e.hits += 1
        e.last_access = now

    # ----------------------------------------------------------------- lookup
    def lookup(self, query: str, namespace: str = "default",
               embedding: np.ndarray | None = None) -> LookupResult:
        now = time.time()
        norm = normalize(query)

        with self._lock:
            eid = self._exact.get((namespace, norm))
            if eid is not None:
                e = self._entries[eid]
                if self._expired(e, now):
                    self._remove_locked([eid])
                else:
                    self._touch(e, now)
                    return LookupResult(True, "exact", 1.0, e, 1.0, e.query,
                                        reason="exact match")
        if not self._use_vectors:
            return LookupResult(False, "miss", reason="no exact match")

        t0 = time.perf_counter()
        if embedding is None:
            embedding = self.embedder.embed([query])[0]
        embed_ms = (time.perf_counter() - t0) * 1000

        t1 = time.perf_counter()
        with self._lock:
            hits = self.index.search(embedding, self.top_k * 3)
            cands: list[tuple[CacheEntry, float]] = []
            expired: list[int] = []
            for cid, sim in hits:
                e = self._entries.get(cid)
                if e is None:
                    continue
                if self._expired(e, now):
                    expired.append(cid)
                elif e.namespace == namespace:
                    cands.append((e, sim))
            if expired:
                self._remove_locked(expired)
        search_ms = (time.perf_counter() - t1) * 1000

        res = LookupResult(False, "miss", embedding=embedding, embed_ms=embed_ms,
                           search_ms=search_ms, reason="no candidate above threshold")
        if cands:
            res.nearest_similarity, res.nearest_query = cands[0][1], cands[0][0].query

        t2 = time.perf_counter()
        for e, sim in cands[: self.top_k]:
            if sim < self.threshold:
                break
            if self.mode == "semantic+verify":
                v = self.verifier.verify(query, e.query)
                if not v.ok:
                    res.verifier_rejections += 1
                    res.reason = f"rejected by verifier: {v.reason}"
                    continue
            with self._lock:
                if e.id not in self._entries:
                    continue
                self._touch(e, now)
            res.hit, res.kind, res.similarity, res.entry = True, "semantic", sim, e
            res.reason = "semantic match"
            break
        res.verify_ms = (time.perf_counter() - t2) * 1000
        return res

    # -------------------------------------------------------------------- add
    def add(self, query: str, answer: str, namespace: str = "default",
            embedding: np.ndarray | None = None, prompt_tokens: int = 0,
            completion_tokens: int = 0, meta: dict[str, Any] | None = None) -> CacheEntry:
        norm = normalize(query)
        if self._use_vectors and embedding is None:
            embedding = self.embedder.embed([query])[0]
        now = time.time()
        with self._lock:
            old = self._exact.get((namespace, norm))
            if old is not None:                      # refresh an existing entry
                e = self._entries[old]
                e.answer, e.created_at, e.last_access = answer, now, now
                e.prompt_tokens, e.completion_tokens = prompt_tokens, completion_tokens
                return e
            e = CacheEntry(self._next_id, namespace, query, norm, answer, now, now, 0,
                           prompt_tokens, completion_tokens, meta or {}, embedding)
            self._next_id += 1
            self._entries[e.id] = e
            self._exact[(namespace, norm)] = e.id
            if self._use_vectors:
                self.index.add([e.id], embedding.reshape(1, -1))
            while len(self._entries) > self.max_entries:      # LRU eviction
                lru = min(self._entries.values(), key=lambda x: x.last_access)
                self._remove_locked([lru.id])
            self._unsaved += 1
            return e

    # ------------------------------------------------------------ housekeeping
    def clear(self) -> int:
        with self._lock:
            n = len(self._entries)
            self._entries.clear()
            self._exact.clear()
            self.index.clear()
            self._unsaved = 0
            return n

    def entries(self, limit: int = 50) -> list[CacheEntry]:
        with self._lock:
            return sorted(self._entries.values(), key=lambda e: -e.last_access)[:limit]

    @property
    def unsaved(self) -> int:
        return self._unsaved

    # ------------------------------------------------------------ persistence
    def save(self, directory: str | Path) -> int:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        with self._lock:
            ents = list(self._entries.values())
            meta = {
                "embedder": self.embedder.name, "dim": self.embedder.dim,
                "entries": [
                    {"id": e.id, "namespace": e.namespace, "query": e.query, "answer": e.answer,
                     "created_at": e.created_at, "last_access": e.last_access, "hits": e.hits,
                     "prompt_tokens": e.prompt_tokens, "completion_tokens": e.completion_tokens,
                     "meta": e.meta} for e in ents],
            }
            vecs = (np.stack([e.embedding for e in ents]) if ents and self._use_vectors
                    else np.zeros((0, self.embedder.dim), dtype=np.float32))
            self._unsaved = 0
        (d / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
        np.save(d / "vectors.npy", vecs)
        return len(ents)

    def load(self, directory: str | Path) -> int:
        d = Path(directory)
        if not (d / "meta.json").exists():
            return 0
        meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        if meta.get("embedder") != self.embedder.name or meta.get("dim") != self.embedder.dim:
            log.warning("Saved cache was built with embedder %r; current is %r. "
                        "Ignoring it (vectors are not comparable).",
                        meta.get("embedder"), self.embedder.name)
            return 0
        vecs = np.load(d / "vectors.npy") if self._use_vectors else None
        loaded = 0
        with self._lock:
            for k, m in enumerate(meta["entries"]):
                emb = vecs[k] if vecs is not None and k < len(vecs) else None
                if self._use_vectors and emb is None:
                    continue
                e = CacheEntry(m["id"], m["namespace"], m["query"], normalize(m["query"]),
                               m["answer"], m["created_at"], m["last_access"], m["hits"],
                               m["prompt_tokens"], m["completion_tokens"], m.get("meta", {}), emb)
                if self._expired(e, time.time()):
                    continue
                self._entries[e.id] = e
                self._exact[(e.namespace, e.norm)] = e.id
                if self._use_vectors:
                    self.index.add([e.id], emb.reshape(1, -1))
                self._next_id = max(self._next_id, e.id + 1)
                loaded += 1
        return loaded
