"""Labelled question data for evaluation.

Two sources:
  * synthetic - built here from templates. Every "intent" (e.g. "capital of
    France") has several paraphrases. Paraphrases of one intent are TRUE
    duplicates; different intents from the same template family are HARD
    NEGATIVES (capital of France vs capital of Germany, enable vs disable...).
    This is a controlled stress test of exactly the failures semantic caches
    have. Because the verifier rules were designed with these failure types in
    mind, treat synthetic numbers as a stress test, not as general accuracy.
  * qqp - the Quora Question Pairs validation set (real human-written pairs).
    This is the credible benchmark: use it for the numbers on your resume.
"""
from __future__ import annotations

import itertools
import random
from dataclasses import dataclass


@dataclass(frozen=True)
class Pair:
    a: str
    b: str
    label: int          # 1 = same question (cache hit is correct), 0 = different
    kind: str           # positive | entity | number | antonym | topic | order | easy | qqp


@dataclass(frozen=True)
class Intent:
    id: str
    family: str
    neg_type: str
    queries: tuple[str, ...]


_ON = [("enable", "turn on"), ("disable", "turn off")]
_APPS = ["VS Code", "Chrome", "Slack"]

# family -> (neg_type, templates, slot dicts)
_FAMILIES: dict[str, tuple[str, list[str], list[dict]]] = {
    "capital": ("entity", [
        "What is the capital of {x}?", "capital of {x}", "Which city is the capital of {x}?",
        "Can you tell me the capital city of {x}?", "what's {x}'s capital city"],
        [{"x": c} for c in ["France", "Germany", "Japan", "Brazil", "Canada", "Italy", "Egypt", "Kenya"]]),
    "install": ("entity", [
        "How do I install {x} on Windows?", "What are the steps to install {x} on Windows?",
        "installing {x} on a windows pc - how do I do it?", "Guide me through setting up {x} on Windows",
        "how to get {x} installed on windows"],
        [{"x": c} for c in ["Python", "Node.js", "Docker", "Git", "Java", "MongoDB"]]),
    "miles": ("number", [
        "What is {n} miles in kilometers?", "Convert {n} miles to kilometers",
        "{n} miles equals how many kilometers?", "how many kilometres is {n} miles", "{n} mi in km?"],
        [{"n": n} for n in ["5", "10", "26", "100", "250"]]),
    "darkmode": ("antonym", [
        "How do I {v} dark mode in {app}?", "How can I {v} dark mode on {app}?",
        "Steps to {v} dark mode in {app}", "What's the way to {v} the dark theme in {app}?",
        "How do I {vo} dark mode in {app}?"],
        [{"v": v, "vo": vo, "app": app} for (v, vo) in _ON for app in _APPS]),
    "concept": ("topic", [
        "What is {x}?", "Explain {x} in simple terms", "Can you give me a simple explanation of {x}?",
        "Define {x}", "{x} - what does it mean?"],
        [{"x": c} for c in ["overfitting", "gradient descent", "a hash table", "recursion",
                            "backpropagation", "a REST API", "a mutex"]]),
    "complexity": ("topic", [
        "What is the time complexity of {x}?", "time complexity of {x}",
        "How fast is {x} in big O terms?", "Tell me the big O time complexity for {x}",
        "{x} time complexity?"],
        [{"x": c} for c in ["quicksort", "merge sort", "binary search", "bubble sort", "heap sort"]]),
    "difference": ("order", [
        "What is the difference between {a} and {b}?", "How is {a} different from {b}?",
        "{a} vs {b}: what's the difference?", "Difference between {b} and {a}",
        "Can you explain how {b} and {a} differ?"],
        [{"a": a, "b": b} for a, b in [("TCP", "UDP"), ("list", "tuple"), ("SQL", "NoSQL"),
                                      ("process", "thread"), ("stack", "heap")]]),
    "reverse": ("entity", [
        "How do I reverse a string in {x}?", "reverse a string in {x}",
        "What's the way to reverse a string using {x}?", "{x} code to reverse a string",
        "Show me how to reverse a string with {x}"],
        [{"x": c} for c in ["Python", "Java", "JavaScript", "C++", "Go", "Rust"]]),
    "polarity": ("antonym", [
        "How can I get {x} working {w} {y}?", "{x} {w} {y} - how do I set it up?",
        "Is it possible to run {x} {w} {y}?", "What's the way to use {x} {w} {y}?"],
        [{"x": x, "y": y, "w": w} for (x, y) in [("Docker", "sudo"), ("Python", "pip"), ("Excel", "macros")]
         for w in ("with", "without")]),
}


def build_intents() -> list[Intent]:
    intents: list[Intent] = []
    for fam, (neg_type, templates, slots) in _FAMILIES.items():
        for k, slot in enumerate(slots):
            intents.append(Intent(f"{fam}:{k}", fam, neg_type,
                                  tuple(t.format(**slot) for t in templates)))
    return intents


def build_synthetic_pairs(seed: int = 7, pos_per_intent: int = 4, neg_per_intent_pair: int = 2,
                          easy_negatives: int = 80) -> list[Pair]:
    rng = random.Random(seed)
    intents = build_intents()
    pairs: list[Pair] = []

    for it in intents:                                            # true duplicates
        combos = list(itertools.combinations(it.queries, 2))
        for a, b in rng.sample(combos, min(pos_per_intent, len(combos))):
            pairs.append(Pair(a, b, 1, "positive"))

    by_fam: dict[str, list[Intent]] = {}
    for it in intents:
        by_fam.setdefault(it.family, []).append(it)
    for fam, group in by_fam.items():                             # hard negatives
        for i1, i2 in itertools.combinations(group, 2):
            for _ in range(neg_per_intent_pair):
                pairs.append(Pair(rng.choice(i1.queries), rng.choice(i2.queries), 0, i1.neg_type))

    for _ in range(easy_negatives):                               # unrelated questions
        i1, i2 = rng.sample(intents, 2)
        if i1.family != i2.family:
            pairs.append(Pair(rng.choice(i1.queries), rng.choice(i2.queries), 0, "easy"))
    return pairs


def build_workload(n: int = 1000, seed: int = 11, zipf_s: float = 1.1) -> list[tuple[str, str]]:
    """n (query, intent_id) requests; popular intents repeat, each time paraphrased."""
    rng = random.Random(seed)
    intents = build_intents()
    rng.shuffle(intents)
    weights = [1.0 / (rank + 1) ** zipf_s for rank in range(len(intents))]
    picks = rng.choices(intents, weights=weights, k=n)
    # Real users rarely repeat a question verbatim: add casual noise so that
    # exact-match caching is not unrealistically strong.
    prefixes = ["", "", "", "please ", "hey, ", "quick question: ", "i was wondering, ", "ok so "]
    suffixes = ["", "", "", " thanks", " pls", " (short answer)", " asap"]
    out = []
    for it in picks:
        q = rng.choice(it.queries)
        q = rng.choice(prefixes) + q.rstrip("?") + ("?" if q.endswith("?") else "") + rng.choice(suffixes)
        out.append((q, it.id))
    return out


def load_qqp(max_pairs: int = 4000, seed: int = 7) -> list[Pair]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit("QQP needs the 'datasets' package: pip install datasets") from exc
    try:
        ds = load_dataset("nyu-mll/glue", "qqp", split="validation")
    except Exception:
        ds = load_dataset("SetFit/qqp", split="validation")   # backup copy of the same data
    rows = [(r["question1"], r["question2"], int(r["label"])) for r in ds]
    random.Random(seed).shuffle(rows)
    return [Pair(a, b, lab, "positive" if lab else "qqp") for a, b, lab in rows[:max_pairs]]
