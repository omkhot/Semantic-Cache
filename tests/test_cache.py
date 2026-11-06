import time

import pytest

from semcache.cache import SemanticCache, normalize
from semcache.embedders import HashEmbedder
from semcache.verifier import NoVerifier, RuleVerifier


def make(mode="semantic+verify", threshold=0.5, **kw):
    ver = RuleVerifier() if mode == "semantic+verify" else NoVerifier()
    return SemanticCache(HashEmbedder(), ver, threshold=threshold, mode=mode,
                         index_backend="numpy", **kw)


def test_normalize():
    assert normalize("  What   is X?? ") == "what is x"


def test_exact_hit():
    c = make("exact")
    c.add("What is the capital of France?", "Paris")
    r = c.lookup("what is the capital of france")
    assert r.hit and r.kind == "exact" and r.entry.answer == "Paris"
    assert not c.lookup("capital of France").hit      # exact mode ignores paraphrases


def test_semantic_hit_on_paraphrase():
    c = make("semantic")
    c.add("What is the capital of France?", "Paris")
    r = c.lookup("Which city is the capital of France?")
    assert r.hit and r.kind == "semantic" and r.similarity >= 0.5


def test_verifier_blocks_false_hit_that_plain_semantic_serves():
    plain, verified = make("semantic"), make("semantic+verify")
    for c in (plain, verified):
        c.add("What is the capital of France?", "Paris")
    q = "What is the capital of Germany?"
    assert plain.lookup(q).hit                       # the bug semantic caching has
    r = verified.lookup(q)
    assert not r.hit and r.verifier_rejections == 1  # the verifier fixes it


def test_namespace_isolation():
    c = make("semantic")
    c.add("What is the capital of France?", "Paris", namespace="a")
    assert not c.lookup("What is the capital of France?", namespace="b").hit
    assert c.lookup("What is the capital of France?", namespace="a").hit


def test_ttl_expiry():
    c = make("semantic", ttl_seconds=1)
    c.add("What is the capital of France?", "Paris")
    assert c.lookup("What is the capital of France?").hit
    time.sleep(1.2)
    assert not c.lookup("What is the capital of France?").hit
    assert len(c) == 0


def test_lru_eviction():
    c = make("semantic", max_entries=2)
    c.add("alpha question one", "1")
    time.sleep(0.01)
    c.add("beta question two", "2")
    time.sleep(0.01)
    c.lookup("alpha question one")                   # touch alpha -> beta is now LRU
    time.sleep(0.01)
    c.add("gamma question three", "3")
    assert len(c) == 2
    assert c.lookup("alpha question one").hit and not c.lookup("beta question two").hit


def test_persistence_roundtrip(tmp_path):
    c = make("semantic")
    c.add("What is the capital of France?", "Paris", meta={"intent": "x"})
    assert c.save(tmp_path) == 1
    c2 = make("semantic")
    assert c2.load(tmp_path) == 1
    r = c2.lookup("Which city is the capital of France?")
    assert r.hit and r.entry.answer == "Paris" and r.entry.meta == {"intent": "x"}


def test_persistence_ignored_when_embedder_changes(tmp_path):
    c = make("semantic")
    c.add("hello there", "hi")
    c.save(tmp_path)
    other = SemanticCache(HashEmbedder(dim=256), NoVerifier(), mode="semantic", index_backend="numpy")
    assert other.load(tmp_path) == 0


def test_invalid_mode():
    with pytest.raises(ValueError):
        make("bogus")
