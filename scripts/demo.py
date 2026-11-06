from __future__ import annotations

import argparse

import httpx

SEQUENCE = [
    ("original question",            "What is the capital of France?"),
    ("paraphrase",                   "Which city is the capital of France?"),
    ("paraphrase",                   "Can you tell me the capital city of France?"),
    ("different casing/punctuation", "what is the capital of france"),
    ("DANGEROUS near-miss",          "What is the capital of Germany?"),
    ("new question",                 "How do I enable dark mode in Chrome?"),
    ("DANGEROUS near-miss (antonym)", "How do I disable dark mode in Chrome?"),
    ("paraphrase",                   "How can I enable dark mode on Chrome?"),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--clear", action="store_true", help="empty the cache first")
    args = ap.parse_args()
    with httpx.Client(base_url=args.url, timeout=120) as c:
        if args.clear:
            c.delete("/v1/cache")
        print(f"{'case':30} {'result':10} {'ms':>8}  {'sim':>6}  note")
        print("-" * 100)
        for label, q in SEQUENCE:
            r = c.post("/v1/chat", json={"query": q})
            if r.status_code != 200:
                print(f"{label:30} ERROR {r.status_code}: {r.text[:200]}")
                continue
            d = r.json()
            ci = d["cache"]
            note = q
            if ci["verifier_rejections"]:
                note += f"   [verifier blocked a wrong match; {ci['reason']}]"
            sim = f"{ci['similarity']:.3f}" if ci["similarity"] is not None else "-"
            print(f"{label:30} {ci['kind'].upper():10} {d['latency_ms']['total']:8.1f}  {sim:>6}  {note}")
        m = c.get("/v1/metrics").json()
        print("-" * 100)
        print(f"requests={m['requests']}  hit_rate={m['hit_rate']:.0%}  llm_calls_saved={m['llm_calls_saved']}  "
              f"verifier_rejections={m['verifier_rejections']}  "
              f"hit p50={m['latency_ms']['hit_p50']} ms  miss p50={m['latency_ms']['miss_p50']} ms")


if __name__ == "__main__":
    main()
