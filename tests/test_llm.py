import asyncio

import httpx
import pytest

from semcache.llm import GeminiClient, LLMError


def client(handler, **kw):
    return GeminiClient("KEY", "gemini-test", "https://example.test/v1beta",
                        backoff_base=0, transport=httpx.MockTransport(handler), **kw)


OK = {"candidates": [{"content": {"parts": [{"text": "Paris"}]}}],
      "usageMetadata": {"promptTokenCount": 7, "candidatesTokenCount": 2}}


def test_success_and_request_shape():
    seen = {}

    def handler(req: httpx.Request):
        seen["url"], seen["key"], seen["body"] = str(req.url), req.headers["x-goog-api-key"], req.read()
        return httpx.Response(200, json=OK)

    r = asyncio.run(client(handler).generate("capital of France?", system="Be brief", temperature=0.2))
    assert r.text == "Paris" and (r.prompt_tokens, r.completion_tokens) == (7, 2)
    assert seen["url"].endswith("/models/gemini-test:generateContent") and seen["key"] == "KEY"
    assert b"systemInstruction" in seen["body"] and b"maxOutputTokens" in seen["body"]


def test_retries_on_429_then_succeeds():
    calls = {"n": 0}

    def handler(req):
        calls["n"] += 1
        return httpx.Response(429, json={"error": {"message": "quota"}}) if calls["n"] < 3 \
            else httpx.Response(200, json=OK)

    assert asyncio.run(client(handler).generate("q")).text == "Paris" and calls["n"] == 3


def test_gives_up_with_helpful_error():
    def handler(req):
        return httpx.Response(429, json={"error": {"message": "quota exceeded"}})

    with pytest.raises(LLMError, match="quota exceeded"):
        asyncio.run(client(handler, max_retries=1).generate("q"))


def test_blocked_response():
    def handler(req):
        return httpx.Response(200, json={"promptFeedback": {"blockReason": "SAFETY"}})

    with pytest.raises(LLMError, match="SAFETY"):
        asyncio.run(client(handler).generate("q"))


def test_missing_key():
    with pytest.raises(LLMError, match="GEMINI_API_KEY"):
        GeminiClient("", "m", "https://x")
