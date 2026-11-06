import pytest

from semcache.verifier import rule_check


@pytest.mark.parametrize("a,b", [
    ("What is the capital of France?", "Which city is the capital of France?"),
    ("How do I install Python on Windows?", "installing python on a windows pc - how?"),
    ("What is the difference between TCP and UDP?", "Difference between UDP and TCP"),
    ("What is 10 miles in kilometers?", "Convert 10 miles to kilometers"),
    ("How do I enable dark mode in Chrome?", "How can I enable dark mode on Chrome?"),
    ("Installing Python on Windows", "How do I install Python on Windows?"),
])
def test_paraphrases_pass(a, b):
    assert rule_check(a, b).ok, rule_check(a, b).reason


@pytest.mark.parametrize("a,b,why", [
    ("What is the capital of France?", "What is the capital of Germany?", "entity"),
    ("How do I reverse a string in Python?", "How do I reverse a string in Java?", "entity"),
    ("What is 5 miles in kilometers?", "What is 50 miles in kilometers?", "number"),
    ("How do I enable dark mode in Chrome?", "How do I disable dark mode in Chrome?", "antonym"),
    ("Run Docker with sudo", "Run Docker without sudo", "antonym"),
    ("Why is my build failing?", "Why is my build not failing?", "negation"),
    ("Who invented the telephone?", "When was the telephone invented?", "question-type"),
])
def test_dangerous_near_misses_rejected(a, b, why):
    r = rule_check(a, b)
    assert not r.ok
    assert why in r.reason
