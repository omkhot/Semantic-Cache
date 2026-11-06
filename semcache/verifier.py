"""Second-stage check that decides whether a *similar* cached question is really
the *same* question.

Why it exists: embedding models rate "capital of France?" and "capital of
Germany?" as very similar, but the answers differ. Raising the threshold
fixes that only by throwing away real paraphrases too. The verifier keeps
the threshold lower (more hits) and rejects the dangerous near-misses:

  * rules      - fast, explainable, no model: entities, numbers, negation,
                 antonyms (enable/disable), question type (who/when/where/why)
  * cross      - a cross-encoder trained on Quora duplicate questions
  * rules+cross- rules first (cheap), cross-encoder only if rules pass

Rules are deliberately conservative: a false rejection only costs one extra
LLM call, while a false acceptance serves a wrong answer.
"""
from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class VerifyResult:
    ok: bool
    reason: str = ""


# Capitalised words that are NOT entities just because they start a question.
_NOT_ENTITY = set("""
what whats how why when where who whom whose which can could would should will do does did is
are was were am tell explain define describe show give list convert guide help please steps step
the a an i my me we you it to in on of for and or if let write make find get create install setup
set use using best way ways any some there this that these those need want has have hi hello hey
compare difference differences code
""".split())

_NEGATIONS = {
    "not", "no", "never", "cannot", "cant", "dont", "doesnt", "isnt", "wont", "didnt",
    "wasnt", "arent", "shouldnt", "couldnt", "wouldnt", "none", "nothing", "neither", "nor",
}

_WH = {"who", "whom", "when", "where", "why"}

_NUMBER_WORDS = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6",
    "seven": "7", "eight": "8", "nine": "9", "ten": "10", "eleven": "11", "twelve": "12",
}

_ANTONYMS = [
    ("enable", "disable"), ("on", "off"), ("add", "remove"), ("install", "uninstall"),
    ("increase", "decrease"), ("open", "close"), ("start", "stop"), ("buy", "sell"),
    ("allow", "block"), ("before", "after"), ("minimum", "maximum"), ("min", "max"),
    ("first", "last"), ("encrypt", "decrypt"), ("compress", "decompress"),
    ("push", "pull"), ("public", "private"), ("synchronous", "asynchronous"),
    ("mutable", "immutable"), ("with", "without"), ("up", "down"), ("lock", "unlock"),
    ("login", "logout"), ("import", "export"), ("upload", "download"), ("create", "delete"),
    ("accept", "reject"), ("show", "hide"), ("true", "false"), ("fast", "slow"),
    ("high", "low"), ("old", "new"), ("true", "false"), ("win", "lose"), ("hot", "cold"),
]

_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9.+#_'-]*")


def _stem(w: str) -> str:
    for suf in ("ing", "ed", "es", "s", "e"):
        if w.endswith(suf) and len(w) - len(suf) >= 3:
            return w[: -len(suf)]
    return w


_ANT_STEMS = [(_stem(a), _stem(b)) for a, b in _ANTONYMS]


def _clean(tok: str) -> str:
    tok = tok.strip(".-_'")
    if tok.lower().endswith("'s"):
        tok = tok[:-2]
    return tok


def _tokens(text: str) -> list[str]:
    return [t for t in (_clean(t) for t in _TOKEN.findall(text)) if t]


def _is_entity(tok: str) -> bool:
    low = tok.lower()
    if re.fullmatch(r"\d[\d.,]*", tok):
        return False
    if len(tok) >= 2 and tok.isupper():                       # TCP, SQL
        return True
    if any(c.isupper() for c in tok[1:]):                     # JavaScript, NoSQL
        return True
    if re.search(r"[a-z]\.[a-z]|\+\+|#", tok):                # Node.js, C++, C#
        return True
    if tok[0].isupper() and low not in _NOT_ENTITY:           # France, Python
        return True
    return False


def _numbers(text: str, toks: list[str]) -> set[str]:
    nums = {m.replace(",", "") for m in re.findall(r"\d+(?:[.,]\d+)*", text)}
    nums |= {_NUMBER_WORDS[t.lower()] for t in toks if t.lower() in _NUMBER_WORDS}
    return nums


def rule_check(new_query: str, cached_query: str) -> VerifyResult:
    a_toks, b_toks = _tokens(new_query), _tokens(cached_query)
    a_low, b_low = {t.lower() for t in a_toks}, {t.lower() for t in b_toks}
    a_st, b_st = {_stem(t) for t in a_low}, {_stem(t) for t in b_low}

    # 1. Entities: every proper noun in one query must appear in the other
    #    (case-insensitive and stem-insensitive, so "windows" matches "Windows"
    #    and a sentence-initial "Installing" matches "install").
    for toks, other in ((a_toks, b_st), (b_toks, a_st)):
        for t in toks:
            if _is_entity(t) and _stem(t.lower()) not in other:
                return VerifyResult(False, f"entity mismatch: '{t}'")

    # 2. Numbers must match exactly.
    na, nb = _numbers(new_query, a_toks), _numbers(cached_query, b_toks)
    if na != nb:
        return VerifyResult(False, f"number mismatch: {sorted(na)} vs {sorted(nb)}")

    # 3. Negation parity.
    a_neg = bool(a_low & _NEGATIONS) or bool(re.search(r"n't\b", new_query.lower()))
    b_neg = bool(b_low & _NEGATIONS) or bool(re.search(r"n't\b", cached_query.lower()))
    if a_neg != b_neg:
        return VerifyResult(False, "negation mismatch")

    # 4. Antonyms (enable vs disable, with vs without, ...).
    for x, y in _ANT_STEMS:
        if (x in a_st and y in b_st and x not in b_st and y not in a_st) or (
            y in a_st and x in b_st and y not in b_st and x not in a_st
        ):
            return VerifyResult(False, f"antonym mismatch: {x}/{y}")

    # 5. Question type (who / when / where / why) must agree if both state one.
    wa, wb = a_low & _WH, b_low & _WH
    if wa and wb and wa != wb:
        return VerifyResult(False, "question-type mismatch")

    return VerifyResult(True, "rules passed")


# --------------------------------------------------------------------------
class NoVerifier:
    name = "none"

    def verify(self, new_query: str, cached_query: str) -> VerifyResult:
        return VerifyResult(True, "no verifier")

    def verify_many(self, pairs: list[tuple[str, str]]) -> list[VerifyResult]:
        return [self.verify(a, b) for a, b in pairs]


class RuleVerifier(NoVerifier):
    name = "rules"

    def verify(self, new_query: str, cached_query: str) -> VerifyResult:
        return rule_check(new_query, cached_query)


class CrossEncoderVerifier(NoVerifier):
    name = "cross"

    def __init__(self, model_name: str = "cross-encoder/quora-distilroberta-base",
                 threshold: float = 0.5):
        from sentence_transformers import CrossEncoder  # lazy: heavy import

        self.model = CrossEncoder(model_name)
        self.threshold = threshold

    def _score(self, pairs: list[tuple[str, str]]) -> list[float]:
        return [float(s) for s in self.model.predict(pairs, show_progress_bar=False)]

    def verify(self, new_query: str, cached_query: str) -> VerifyResult:
        return self.verify_many([(new_query, cached_query)])[0]

    def verify_many(self, pairs: list[tuple[str, str]]) -> list[VerifyResult]:
        if not pairs:
            return []
        return [VerifyResult(s >= self.threshold, f"cross-encoder score {s:.2f}")
                for s in self._score(pairs)]


class CompositeVerifier(NoVerifier):
    """Rules first (free), cross-encoder only for pairs the rules accept."""

    name = "rules+cross"

    def __init__(self, cross: CrossEncoderVerifier):
        self.cross = cross

    def verify(self, new_query: str, cached_query: str) -> VerifyResult:
        r = rule_check(new_query, cached_query)
        return r if not r.ok else self.cross.verify(new_query, cached_query)

    def verify_many(self, pairs: list[tuple[str, str]]) -> list[VerifyResult]:
        rules = [rule_check(a, b) for a, b in pairs]
        todo = [i for i, r in enumerate(rules) if r.ok]
        for i, res in zip(todo, self.cross.verify_many([pairs[i] for i in todo])):
            rules[i] = res
        return rules


def make_verifier(kind: str, cross_model: str = "cross-encoder/quora-distilroberta-base",
                  cross_threshold: float = 0.5) -> NoVerifier:
    kind = kind.lower()
    if kind == "none":
        return NoVerifier()
    if kind == "rules":
        return RuleVerifier()
    if kind in ("cross", "rules+cross"):
        try:
            cross = CrossEncoderVerifier(cross_model, cross_threshold)
        except Exception as exc:
            raise RuntimeError(
                f"Could not load the cross-encoder '{cross_model}' ({exc}). "
                "Install requirements and stay online for the first download, "
                "or set VERIFIER=rules in .env."
            ) from exc
        if kind == "cross":
            cross.name = "cross"
            return cross
        return CompositeVerifier(cross)
    raise ValueError(f"Unknown verifier '{kind}' (use none | rules | cross | rules+cross)")
