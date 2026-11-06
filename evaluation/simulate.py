"""Workload simulation: replay N paraphrased requests through each cache setup.

    python -m evaluation.simulate --embedder minilm --threshold 0.85 --requests 1000

No LLM is called. A cache miss is charged a SIMULATED LLM latency (default
1200 ms; set --llm-latency-ms to what you measured on Gemini). Everything on
the cache side (embedding, search, verification) is measured for real.
Every request knows which intent it belongs to, so we can count answers
served for the WRONG question exactly.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from semcache.cache import SemanticCache
from semcache.config import Settings, load_dotenv
from semcache.embedders import make_embedder
from semcache.verifier import make_verifier

from .dataset import build_workload


def run(mode: str, embedder, verifier, threshold: float, workload, llm_ms: float,
        answer_tokens: int, price_in: float, price_out: float) -> dict:
    cache = SemanticCache(embedder, verifier, threshold=threshold, mode=mode, ttl_seconds=0,
                          max_entries=100000, index_backend="auto")
    lat, hit_lat, miss_lat = [], [], []
    hits = wrong = calls = 0
    tok_in = tok_out = 0
    for query, intent in workload:
        t0 = time.perf_counter()
        r = cache.lookup(query)
        real_ms = (time.perf_counter() - t0) * 1000
        p_tok = max(1, int(len(query.split()) * 1.3))
        if r.hit:
            hits += 1
            wrong += int(r.entry.meta.get("intent") != intent)
            lat.append(real_ms)
            hit_lat.append(real_ms)
        else:
            calls += 1
            tok_in += p_tok
            tok_out += answer_tokens
            cache.add(query, f"answer for {intent}", embedding=r.embedding, prompt_tokens=p_tok,
                      completion_tokens=answer_tokens, meta={"intent": intent})
            lat.append(real_ms + llm_ms)
            miss_lat.append(real_ms + llm_ms)
    n = len(workload)
    baseline_cost = sum(max(1, int(len(q.split()) * 1.3)) for q, _ in workload) * price_in / 1e6 \
        + n * answer_tokens * price_out / 1e6
    cost = (tok_in * price_in + tok_out * price_out) / 1e6
    pct = lambda v, q: round(float(np.percentile(v, q)), 2) if v else None
    return {
        "mode": mode, "requests": n, "llm_calls": calls,
        "llm_calls_saved_pct": round(100 * (n - calls) / n, 1),
        "hit_rate_pct": round(100 * hits / n, 1),
        "wrong_answers_served": wrong,
        "false_hit_rate_pct_of_hits": round(100 * wrong / hits, 2) if hits else 0.0,
        "latency_ms_overall_p50": pct(lat, 50), "latency_ms_overall_p95": pct(lat, 95),
        "latency_ms_hit_p50": pct(hit_lat, 50), "latency_ms_hit_p95": pct(hit_lat, 95),
        "latency_ms_miss_p50_simulated": pct(miss_lat, 50),
        "notional_cost_usd": round(cost, 5), "notional_cost_without_cache_usd": round(baseline_cost, 5),
        "notional_cost_reduction_pct": round(100 * (1 - cost / baseline_cost), 1) if baseline_cost else 0.0,
    }


def main() -> None:
    load_dotenv()
    s = Settings.from_env()
    ap = argparse.ArgumentParser()
    ap.add_argument("--embedder", default=s.embedder)
    ap.add_argument("--embedder-model", default=s.embedder_model)
    ap.add_argument("--verifier", default="rules")
    ap.add_argument("--threshold", type=float, default=s.similarity_threshold)
    ap.add_argument("--requests", type=int, default=1000)
    ap.add_argument("--llm-latency-ms", type=float, default=1200.0)
    ap.add_argument("--answer-tokens", type=int, default=250)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    embedder = make_embedder(args.embedder, args.embedder_model, s.gemini_api_key, s.gemini_embed_model)
    verifier = make_verifier(args.verifier, s.cross_encoder_model, s.cross_encoder_threshold)
    workload = build_workload(args.requests)
    print(f"Workload: {len(workload)} requests, {len({i for _, i in workload})} distinct intents, "
          f"embedder={embedder.name}, verifier={verifier.name}, threshold={args.threshold}, "
          f"simulated LLM latency={args.llm_latency_ms:.0f} ms")

    results = [run(m, embedder, verifier, args.threshold, workload, args.llm_latency_ms,
                   args.answer_tokens, s.price_in_per_m, s.price_out_per_m)
               for m in ("exact", "semantic", "semantic+verify")]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "workload.json").write_text(json.dumps(
        {"embedder": embedder.name, "verifier": verifier.name, "threshold": args.threshold,
         "simulated_llm_latency_ms": args.llm_latency_ms, "results": results}, indent=2))

    md = ["| Setup | LLM calls saved | Wrong answers served | p50 latency (all) | p95 latency (all) | "
          "p50 on hits | Notional cost cut |", "|---|---|---|---|---|---|---|"]
    for r in results:
        md.append(f"| {r['mode']} | {r['llm_calls_saved_pct']}% | {r['wrong_answers_served']} "
                  f"({r['false_hit_rate_pct_of_hits']}% of hits) | {r['latency_ms_overall_p50']} ms | "
                  f"{r['latency_ms_overall_p95']} ms | {r['latency_ms_hit_p50']} ms | "
                  f"{r['notional_cost_reduction_pct']}% |")
    (out / "workload.md").write_text("\n".join(md) + "\n")
    print("\n" + "\n".join(md))
    print("\nNote: miss latency is SIMULATED; hit latency (embed+search+verify) is measured.")


if __name__ == "__main__":
    main()
