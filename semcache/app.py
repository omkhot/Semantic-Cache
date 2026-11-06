"""FastAPI service: a semantic-caching proxy in front of an LLM.

Start it with:   uvicorn semcache.app:app --port 8000
Open the demo:   http://localhost:8000/
API docs:        http://localhost:8000/docs
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from .cache import SemanticCache, normalize
from .config import Settings
from .embedders import make_embedder
from .llm import LLMError, make_llm
from .metrics import Metrics
from .verifier import make_verifier

log = logging.getLogger("semcache.api")
STATIC = Path(__file__).parent / "static"


# ------------------------------------------------------------------ schemas
class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=8000, description="The user's question.")
    namespace: str = Field("default", max_length=64,
                           description="Cache partition (e.g. per tenant or per model).")
    system: str | None = Field(None, max_length=4000,
                               description="Optional system prompt. Part of the cache key.")
    temperature: float | None = Field(None, ge=0, le=2)
    bypass_cache: bool = Field(False, description="Skip the cache read and call the LLM.")


class CacheInfo(BaseModel):
    kind: str                       # exact | semantic | coalesced | miss | bypass
    similarity: float | None = None
    matched_query: str | None = None
    nearest_similarity: float | None = None
    nearest_query: str | None = None
    verifier_rejections: int = 0
    reason: str = ""


class ChatResponse(BaseModel):
    answer: str
    cached: bool
    cache: CacheInfo
    latency_ms: dict[str, float]
    tokens: dict[str, int]
    model: str


def _namespace(req: ChatRequest) -> str:
    """Different system prompts must never share cached answers."""
    if req.system:
        return f"{req.namespace}#{hashlib.sha1(req.system.encode()).hexdigest()[:8]}"
    return req.namespace


# ------------------------------------------------------------------ factory
def create_app(settings: Settings | None = None, *, embedder=None, llm=None) -> FastAPI:
    """Build the app. Tests pass their own embedder/llm; production uses env settings."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        s = settings or Settings.from_env()
        logging.basicConfig(level=logging.INFO,
                            format="%(asctime)s %(levelname)s %(name)s: %(message)s")
        emb = embedder or make_embedder(s.embedder, s.embedder_model, s.gemini_api_key,
                                        s.gemini_embed_model)
        ver = make_verifier(s.verifier, s.cross_encoder_model, s.cross_encoder_threshold)
        cache = SemanticCache(emb, ver, threshold=s.similarity_threshold, mode=s.cache_mode,
                              ttl_seconds=s.ttl_seconds, max_entries=s.max_entries,
                              top_k=s.top_k, index_backend=s.index_backend)
        loaded = cache.load(s.persist_dir) if s.persist_dir else 0
        backend = llm or make_llm(s)
        app.state.sc = SimpleNamespace(
            settings=s, cache=cache, llm=backend, inflight={},
            metrics=Metrics(s.price_in_per_m, s.price_out_per_m))
        log.info("Ready | llm=%s embedder=%s index=%s mode=%s verifier=%s threshold=%.2f "
                 "| restored %d cached entries", s.llm_provider, emb.name, cache.index.backend,
                 s.cache_mode, ver.name, s.similarity_threshold, loaded)
        yield
        if s.persist_dir:
            n = cache.save(s.persist_dir)
            log.info("Saved %d cache entries to %s", n, s.persist_dir)
        await backend.aclose()

    app = FastAPI(title="SemanticCache", version="1.0.0", lifespan=lifespan,
                  description="Semantic caching proxy for LLM calls "
                              "(embeddings + vector search + verification).")

    # ------------------------------------------------------------ endpoints
    @app.get("/", include_in_schema=False)
    async def dashboard():
        return FileResponse(STATIC / "index.html")

    @app.get("/health")
    async def health():
        st = app.state.sc
        return {"status": "ok", "entries": len(st.cache), "embedder": st.cache.embedder.name,
                "index_backend": st.cache.index.backend, "llm": st.settings.llm_provider}

    @app.get("/v1/config")
    async def config():
        return app.state.sc.settings.public_dict()

    @app.get("/v1/metrics")
    async def metrics():
        st = app.state.sc
        snap = st.metrics.snapshot()
        snap["cache_entries"] = len(st.cache)
        return snap

    @app.post("/v1/metrics/reset")
    async def reset_metrics():
        app.state.sc.metrics.reset()
        return {"reset": True}

    @app.get("/v1/cache/entries")
    async def entries(limit: int = Query(50, ge=1, le=500)):
        return [{"id": e.id, "namespace": e.namespace, "query": e.query, "hits": e.hits,
                 "age_s": round(time.time() - e.created_at, 1), "answer_preview": e.answer[:120]}
                for e in app.state.sc.cache.entries(limit)]

    @app.delete("/v1/cache")
    async def clear_cache():
        return {"cleared": app.state.sc.cache.clear()}

    @app.post("/v1/cache/save")
    async def save_cache():
        st = app.state.sc
        return {"saved": st.cache.save(st.settings.persist_dir)}

    @app.post("/v1/chat", response_model=ChatResponse)
    async def chat(req: ChatRequest):
        st = app.state.sc
        t0 = time.perf_counter()
        ns = _namespace(req)
        lookup = None

        # 1) Try the cache.
        if not req.bypass_cache:
            lookup = await asyncio.to_thread(st.cache.lookup, req.query, ns)
            if lookup.hit:
                e = lookup.entry
                total = (time.perf_counter() - t0) * 1000
                st.metrics.record_hit(lookup.kind, total, e.prompt_tokens, e.completion_tokens,
                                      lookup.verifier_rejections)
                return ChatResponse(
                    answer=e.answer, cached=True,
                    cache=CacheInfo(kind=lookup.kind, similarity=round(lookup.similarity, 4),
                                    matched_query=e.query,
                                    nearest_similarity=lookup.nearest_similarity,
                                    nearest_query=lookup.nearest_query,
                                    verifier_rejections=lookup.verifier_rejections,
                                    reason=lookup.reason),
                    latency_ms={"total": round(total, 2), "embed": round(lookup.embed_ms, 2),
                                "search": round(lookup.search_ms, 2),
                                "verify": round(lookup.verify_ms, 2), "llm": 0.0},
                    tokens={"prompt": 0, "completion": 0}, model="cache")

        # 2) Miss: call the LLM. Identical concurrent misses share ONE call.
        key = (ns, normalize(req.query))
        task = st.inflight.get(key)
        coalesced = task is not None
        if task is None:
            embedding = lookup.embedding if lookup else None

            async def call_and_store():
                r = await st.llm.generate(req.query, req.system, req.temperature)
                await asyncio.to_thread(st.cache.add, req.query, r.text, ns, embedding,
                                        r.prompt_tokens, r.completion_tokens)
                if st.settings.persist_dir and st.cache.unsaved >= st.settings.autosave_every:
                    await asyncio.to_thread(st.cache.save, st.settings.persist_dir)
                return r

            task = asyncio.create_task(call_and_store())
            st.inflight[key] = task
            task.add_done_callback(lambda _t, k=key: st.inflight.pop(k, None))
        try:
            r = await asyncio.shield(task)
        except LLMError as exc:
            st.metrics.record_error()
            raise HTTPException(status_code=502, detail=str(exc))

        total = (time.perf_counter() - t0) * 1000
        rej = lookup.verifier_rejections if lookup else 0
        if coalesced:
            st.metrics.record_hit("coalesced", total, r.prompt_tokens, r.completion_tokens)
            kind, reason = "coalesced", "shared an in-flight LLM call for the same question"
        else:
            st.metrics.record_miss(total, r.prompt_tokens, r.completion_tokens, rej,
                                   bypass=req.bypass_cache)
            kind = "bypass" if req.bypass_cache else "miss"
            reason = "cache bypassed" if req.bypass_cache else (lookup.reason if lookup else "")
        return ChatResponse(
            answer=r.text, cached=False,
            cache=CacheInfo(kind=kind,
                            nearest_similarity=lookup.nearest_similarity if lookup else None,
                            nearest_query=lookup.nearest_query if lookup else None,
                            verifier_rejections=rej, reason=reason),
            latency_ms={"total": round(total, 2),
                        "embed": round(lookup.embed_ms, 2) if lookup else 0.0,
                        "search": round(lookup.search_ms, 2) if lookup else 0.0,
                        "verify": round(lookup.verify_ms, 2) if lookup else 0.0,
                        "llm": round(r.latency_ms, 2)},
            tokens={"prompt": r.prompt_tokens, "completion": r.completion_tokens},
            model=r.model)

    return app


app = create_app()
