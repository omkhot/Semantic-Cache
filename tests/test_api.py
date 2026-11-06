import asyncio

import pytest
from fastapi.testclient import TestClient

from semcache.app import create_app
from semcache.config import Settings
from semcache.embedders import HashEmbedder
from semcache.llm import LLMError, MockLLM


def settings(**kw):
    base = dict(llm_provider="mock", embedder="hash", persist_dir="", similarity_threshold=0.5,
                index_backend="numpy", verifier="rules")
    base.update(kw)
    return Settings(**base)


@pytest.fixture
def api():
    llm = MockLLM(latency_s=0.0)
    with TestClient(create_app(settings(), embedder=HashEmbedder(), llm=llm)) as c:
        c.llm = llm
        yield c


def ask(c, q, **kw):
    r = c.post("/v1/chat", json={"query": q, **kw})
    assert r.status_code == 200, r.text
    return r.json()


def test_miss_then_exact_then_semantic(api):
    a = ask(api, "What is the capital of France?")
    assert not a["cached"] and a["cache"]["kind"] == "miss"
    b = ask(api, "what is the capital of france")
    assert b["cached"] and b["cache"]["kind"] == "exact" and b["answer"] == a["answer"]
    c = ask(api, "Which city is the capital of France?")
    assert c["cached"] and c["cache"]["kind"] == "semantic"
    assert api.llm.calls == 1


def test_verifier_prevents_wrong_answer(api):
    ask(api, "What is the capital of France?")
    d = ask(api, "What is the capital of Germany?")
    assert not d["cached"] and d["cache"]["verifier_rejections"] == 1
    assert api.llm.calls == 2


def test_bypass_and_namespace_and_system(api):
    ask(api, "What is the capital of France?")
    assert not ask(api, "What is the capital of France?", bypass_cache=True)["cached"]
    assert not ask(api, "What is the capital of France?", namespace="other")["cached"]
    assert not ask(api, "What is the capital of France?", system="Answer in French")["cached"]


def test_metrics_and_admin_endpoints(api):
    ask(api, "What is the capital of France?")
    ask(api, "What is the capital of France?")
    m = api.get("/v1/metrics").json()
    assert m["requests"] == 2 and m["hits"] == 1 and m["hit_rate"] == 0.5
    assert api.get("/health").json()["status"] == "ok"
    assert len(api.get("/v1/cache/entries").json()) == 1
    assert "gemini_api_key" in api.get("/v1/config").json()
    assert api.delete("/v1/cache").json()["cleared"] == 1
    assert api.get("/").status_code == 200


def test_validation(api):
    assert api.post("/v1/chat", json={"query": ""}).status_code == 422


def test_llm_failure_returns_502_and_is_not_cached():
    class Failing(MockLLM):
        async def generate(self, *a, **k):
            raise LLMError("boom")

    with TestClient(create_app(settings(), embedder=HashEmbedder(), llm=Failing())) as c:
        r = c.post("/v1/chat", json={"query": "hello world"})
        assert r.status_code == 502 and "boom" in r.json()["detail"]
        assert c.get("/health").json()["entries"] == 0


def test_concurrent_identical_requests_share_one_llm_call():
    import httpx

    llm = MockLLM(latency_s=0.3)
    app = create_app(settings(), embedder=HashEmbedder(), llm=llm)

    async def run():
        async with app.router.lifespan_context(app):
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://t") as c:
                rs = await asyncio.gather(*[
                    c.post("/v1/chat", json={"query": "What is the capital of France?"})
                    for _ in range(5)])
                return [r.json() for r in rs]

    out = asyncio.run(run())
    assert llm.calls == 1
    assert sum(1 for o in out if o["cache"]["kind"] == "coalesced") == 4
