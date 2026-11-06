"""Thread-safe counters and latency percentiles for the /v1/metrics endpoint."""
from __future__ import annotations

import threading
import time
from collections import deque

import numpy as np


def _pct(values, q: float) -> float | None:
    return round(float(np.percentile(list(values), q)), 2) if values else None


class Metrics:
    def __init__(self, price_in_per_m: float, price_out_per_m: float, window: int = 5000):
        self._lock = threading.Lock()
        self.started = time.time()
        self.price_in, self.price_out = price_in_per_m, price_out_per_m
        self.requests = self.exact_hits = self.semantic_hits = self.coalesced = 0
        self.misses = self.verifier_rejections = self.bypasses = self.errors = 0
        self.tokens_saved_in = self.tokens_saved_out = 0
        self.tokens_spent_in = self.tokens_spent_out = 0
        self.hit_lat: deque[float] = deque(maxlen=window)
        self.miss_lat: deque[float] = deque(maxlen=window)

    def record_hit(self, kind: str, latency_ms: float, p_tok: int, c_tok: int, rejections: int = 0):
        with self._lock:
            self.requests += 1
            if kind == "exact":
                self.exact_hits += 1
            elif kind == "coalesced":
                self.coalesced += 1
            else:
                self.semantic_hits += 1
            self.verifier_rejections += rejections
            self.tokens_saved_in += p_tok
            self.tokens_saved_out += c_tok
            self.hit_lat.append(latency_ms)

    def record_miss(self, latency_ms: float, p_tok: int, c_tok: int,
                    rejections: int = 0, bypass: bool = False):
        with self._lock:
            self.requests += 1
            self.misses += 1
            self.bypasses += int(bypass)
            self.verifier_rejections += rejections
            self.tokens_spent_in += p_tok
            self.tokens_spent_out += c_tok
            self.miss_lat.append(latency_ms)

    def record_error(self):
        with self._lock:
            self.errors += 1

    def reset(self):
        with self._lock:
            self.__init__(self.price_in, self.price_out)  # type: ignore[misc]

    def snapshot(self) -> dict:
        with self._lock:
            hits = self.exact_hits + self.semantic_hits + self.coalesced
            total = hits + self.misses
            saved_usd = (self.tokens_saved_in * self.price_in
                         + self.tokens_saved_out * self.price_out) / 1e6
            spent_usd = (self.tokens_spent_in * self.price_in
                         + self.tokens_spent_out * self.price_out) / 1e6
            return {
                "uptime_s": round(time.time() - self.started, 1),
                "requests": total,
                "hits": hits,
                "exact_hits": self.exact_hits,
                "semantic_hits": self.semantic_hits,
                "coalesced": self.coalesced,
                "misses": self.misses,
                "bypasses": self.bypasses,
                "errors": self.errors,
                "hit_rate": round(hits / total, 4) if total else 0.0,
                "llm_calls_saved": hits,
                "verifier_rejections": self.verifier_rejections,
                "latency_ms": {
                    "hit_p50": _pct(self.hit_lat, 50), "hit_p95": _pct(self.hit_lat, 95),
                    "miss_p50": _pct(self.miss_lat, 50), "miss_p95": _pct(self.miss_lat, 95),
                },
                "tokens": {"saved_in": self.tokens_saved_in, "saved_out": self.tokens_saved_out,
                           "spent_in": self.tokens_spent_in, "spent_out": self.tokens_spent_out},
                "notional_cost_usd": {"saved": round(saved_usd, 6), "spent": round(spent_usd, 6),
                                      "note": "uses PRICE_IN_PER_M / PRICE_OUT_PER_M from .env"},
            }
