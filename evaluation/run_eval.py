"""Threshold sweep + comparison of exact / semantic / semantic+verifier caches.

    python -m evaluation.run_eval --embedder minilm --verifier rules
    python -m evaluation.run_eval --embedder minilm --verifier rules+cross --dataset qqp

Definitions (each pair = "cached question" + "new question", with a human label):
  hit rate        = share of TRUE duplicates that the cache answered  (recall)
  false-hit rate  = share of DIFFERENT questions that the cache wrongly answered
  precision       = of all cache hits, share that were correct
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from semcache.cache import normalize
from semcache.config import Settings, load_dotenv
from semcache.embedders import make_embedder
from semcache.verifier import make_verifier

from .dataset import build_synthetic_pairs, load_qqp


def metrics(pred: np.ndarray, label: np.ndarray) -> dict:
    tp = int((pred & label).sum())
    fp = int((pred & ~label).sum())
    fn = int((~pred & label).sum())
    tn = int((~pred & ~label).sum())
    prec = tp / (tp + fp) if tp + fp else float("nan")
    rec = tp / (tp + fn) if tp + fn else float("nan")
    f1 = 2 * prec * rec / (prec + rec) if tp and prec == prec and rec == rec else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": prec, "hit_rate": rec,
            "false_hit_rate": fp / (fp + tn) if fp + tn else float("nan"), "f1": f1}


def fmt(x, pct=True):
    if x != x:
        return "n/a"
    return f"{x * 100:.1f}%" if pct else f"{x:.3f}"


def main() -> None:
    load_dotenv()
    s = Settings.from_env()
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["synthetic", "qqp"], default="synthetic")
    ap.add_argument("--embedder", default=s.embedder)
    ap.add_argument("--embedder-model", default=s.embedder_model)
    ap.add_argument("--verifier", default="rules", help="rules | cross | rules+cross")
    ap.add_argument("--max-pairs", type=int, default=4000, help="QQP subsample size")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    pairs = build_synthetic_pairs() if args.dataset == "synthetic" else load_qqp(args.max_pairs)
    label = np.array([p.label for p in pairs], dtype=bool)
    print(f"Dataset: {args.dataset} | {len(pairs)} pairs "
          f"({int(label.sum())} duplicates, {int((~label).sum())} different)")

    embedder = make_embedder(args.embedder, args.embedder_model, s.gemini_api_key, s.gemini_embed_model)
    texts = sorted({t for p in pairs for t in (p.a, p.b)})
    vecs = embedder.embed(texts)
    pos = {t: i for i, t in enumerate(texts)}
    sims = np.array([float(vecs[pos[p.a]] @ vecs[pos[p.b]]) for p in pairs])

    verifier = make_verifier(args.verifier, s.cross_encoder_model, s.cross_encoder_threshold)
    vok = np.array([r.ok for r in verifier.verify_many([(p.b, p.a) for p in pairs])], dtype=bool)
    exact = np.array([normalize(p.a) == normalize(p.b) for p in pairs], dtype=bool)

    # ---- sweep ----------------------------------------------------------
    thresholds = np.round(np.arange(0.50, 1.0, 0.01), 2)
    rows = []
    for t in thresholds:
        sem = sims >= t
        m1, m2 = metrics(sem, label), metrics(sem & vok, label)
        rows.append({"threshold": float(t),
                     **{f"semantic_{k}": v for k, v in m1.items()},
                     **{f"verified_{k}": v for k, v in m2.items()}})
    with open(out / "threshold_sweep.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    def best(prefix: str, key=lambda r, p: r[f"{p}_f1"]):
        return max(rows, key=lambda r: (key(r, prefix), r["threshold"]))

    b_sem, b_ver = best("semantic"), best("verified")
    low_fhr = [r for r in rows if r["verified_false_hit_rate"] == r["verified_false_hit_rate"]
               and r["verified_false_hit_rate"] <= 0.01]
    safe = max(low_fhr, key=lambda r: r["verified_hit_rate"]) if low_fhr else None

    em = metrics(exact, label)
    summary = {
        "dataset": args.dataset, "pairs": len(pairs), "embedder": embedder.name,
        "verifier": verifier.name,
        "exact_match": em,
        "semantic_best_f1": {"threshold": b_sem["threshold"],
                             **{k[9:]: v for k, v in b_sem.items() if k.startswith("semantic_")}},
        "semantic_verified_best_f1": {"threshold": b_ver["threshold"],
                                      **{k[9:]: v for k, v in b_ver.items() if k.startswith("verified_")}},
        "recommended_threshold_for_false_hit_le_1pct": safe["threshold"] if safe else None,
    }

    # ---- false-hit breakdown by failure type at the chosen thresholds ------
    kinds = sorted({p.kind for p in pairs if p.label == 0})
    breakdown = {}
    for k in kinds:
        idx = np.array([p.kind == k for p in pairs])
        n = int(idx.sum())
        a = int(((sims >= b_sem["threshold"]) & idx).sum())
        v = int(((sims >= b_ver["threshold"]) & vok & idx).sum())
        breakdown[k] = {"pairs": n, "semantic_false_hits": a, "verified_false_hits": v}
    summary["false_hit_breakdown"] = breakdown
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float))

    # ---- markdown table ----------------------------------------------------
    sem, ver = summary["semantic_best_f1"], summary["semantic_verified_best_f1"]
    md = [f"### Results: `{args.dataset}` ({len(pairs)} pairs) | embedder `{embedder.name}` | "
          f"verifier `{verifier.name}`", "",
          "| Setup | Threshold | Hit rate | False-hit rate | Precision | F1 |",
          "|---|---|---|---|---|---|",
          f"| Exact match | - | {fmt(em['hit_rate'])} | {fmt(em['false_hit_rate'])} | "
          f"{fmt(em['precision'])} | {fmt(em['f1'], False)} |",
          f"| Semantic | {sem['threshold']:.2f} | {fmt(sem['hit_rate'])} | {fmt(sem['false_hit_rate'])} | "
          f"{fmt(sem['precision'])} | {fmt(sem['f1'], False)} |",
          f"| Semantic + verifier | {ver['threshold']:.2f} | {fmt(ver['hit_rate'])} | "
          f"{fmt(ver['false_hit_rate'])} | {fmt(ver['precision'])} | {fmt(ver['f1'], False)} |",
          "", "False hits by failure type (each at its own best-F1 threshold):", "",
          "| Type | Different-question pairs | Semantic false hits | With verifier |", "|---|---|---|---|"]
    md += [f"| {k} | {v['pairs']} | {v['semantic_false_hits']} | {v['verified_false_hits']} |"
           for k, v in breakdown.items()]
    (out / f"comparison_{args.dataset}.md").write_text("\n".join(md) + "\n")
    print("\n" + "\n".join(md))
    if safe:
        print(f"\nHighest hit rate with false-hit rate <= 1%: threshold {safe['threshold']:.2f} "
              f"(hit rate {fmt(safe['verified_hit_rate'])})")

    # ---- plot (optional) ---------------------------------------------------
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        t = [r["threshold"] for r in rows]
        ax[0].plot(t, [r["semantic_hit_rate"] for r in rows], label="hit rate (semantic)")
        ax[0].plot(t, [r["verified_hit_rate"] for r in rows], label="hit rate (+verifier)")
        ax[0].plot(t, [r["semantic_false_hit_rate"] for r in rows], "--", label="false-hit rate (semantic)")
        ax[0].plot(t, [r["verified_false_hit_rate"] for r in rows], "--", label="false-hit rate (+verifier)")
        ax[0].set_xlabel("similarity threshold"); ax[0].set_ylabel("rate"); ax[0].legend(fontsize=8)
        ax[0].set_title("Hit rate vs false-hit rate")
        ax[1].plot([r["semantic_hit_rate"] for r in rows], [r["semantic_false_hit_rate"] for r in rows],
                   label="semantic")
        ax[1].plot([r["verified_hit_rate"] for r in rows], [r["verified_false_hit_rate"] for r in rows],
                   label="semantic + verifier")
        ax[1].set_xlabel("hit rate (good)"); ax[1].set_ylabel("false-hit rate (bad)")
        ax[1].set_title("Trade-off curve (lower-right is better)"); ax[1].legend()
        fig.tight_layout()
        fig.savefig(out / f"threshold_sweep_{args.dataset}.png", dpi=150)
        print(f"Saved plot to {out / f'threshold_sweep_{args.dataset}.png'}")
    except ImportError:
        print("(matplotlib not installed - skipped the plot; `pip install matplotlib`)")


if __name__ == "__main__":
    main()
