"""LLM backends: Gemini (REST) and a Mock for tests / offline demos."""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass

import httpx

from .config import Settings


class LLMError(RuntimeError):
    """Raised when the LLM provider fails; the API turns it into HTTP 502."""


@dataclass
class LLMResponse:
    text: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float
    model: str


class GeminiClient:
    RETRY_STATUS = {429, 500, 502, 503, 504}

    def __init__(self, api_key: str, model: str, base_url: str, max_output_tokens: int = 1024,
                 timeout_s: float = 60.0, max_retries: int = 3, backoff_base: float = 1.5,
                 transport: httpx.AsyncBaseTransport | None = None):
        if not api_key:
            raise LLMError("GEMINI_API_KEY is missing. Put it in the .env file "
                           "(or set LLM_PROVIDER=mock to run without a key).")
        self.model, self.max_output_tokens = model, max_output_tokens
        self.max_retries, self.backoff_base = max_retries, backoff_base
        self._url = f"{base_url}/models/{model}:generateContent"
        self._client = httpx.AsyncClient(
            timeout=timeout_s, transport=transport,
            headers={"x-goog-api-key": api_key, "Content-Type": "application/json"})

    async def generate(self, prompt: str, system: str | None = None,
                       temperature: float | None = None) -> LLMResponse:
        body: dict = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"maxOutputTokens": self.max_output_tokens},
        }
        if temperature is not None:
            body["generationConfig"]["temperature"] = temperature
        if system:
            body["systemInstruction"] = {"parts": [{"text": system}]}

        t0 = time.perf_counter()
        resp: httpx.Response | None = None
        for attempt in range(self.max_retries + 1):
            try:
                resp = await self._client.post(self._url, json=body)
            except httpx.HTTPError as exc:
                if attempt == self.max_retries:
                    raise LLMError(f"Network error talking to Gemini: {exc}") from exc
            else:
                if resp.status_code not in self.RETRY_STATUS:
                    break
                if attempt == self.max_retries:
                    break
            wait = self.backoff_base * (2 ** attempt)
            if resp is not None and resp.headers.get("retry-after", "").isdigit():
                wait = max(wait, float(resp.headers["retry-after"]))
            await asyncio.sleep(wait if self.backoff_base > 0 else 0)

        assert resp is not None
        if resp.status_code != 200:
            try:
                msg = resp.json()["error"]["message"]
            except Exception:
                msg = resp.text[:300]
            hint = (" (rate limit: the free tier allows few requests per minute; wait and retry)"
                    if resp.status_code == 429 else
                    " (model name wrong? run `python -m scripts.list_models`)"
                    if resp.status_code == 404 else "")
            raise LLMError(f"Gemini returned HTTP {resp.status_code}: {msg}{hint}")

        data = resp.json()
        try:
            parts = data["candidates"][0]["content"]["parts"]
            text = "".join(p.get("text", "") for p in parts).strip()
        except (KeyError, IndexError, TypeError):
            reason = data.get("promptFeedback", {}).get("blockReason") or \
                (data.get("candidates") or [{}])[0].get("finishReason", "unknown")
            raise LLMError(f"Gemini returned no text (reason: {reason}).")
        if not text:
            raise LLMError("Gemini returned an empty answer (try raising MAX_OUTPUT_TOKENS).")
        usage = data.get("usageMetadata", {})
        return LLMResponse(text, int(usage.get("promptTokenCount", 0)),
                           int(usage.get("candidatesTokenCount", 0)),
                           (time.perf_counter() - t0) * 1000, self.model)

    async def aclose(self) -> None:
        await self._client.aclose()


class MockLLM:
    """Deterministic fake LLM: lets you demo and test with no API key."""

    def __init__(self, latency_s: float = 0.8):
        self.latency_s, self.model, self.calls = latency_s, "mock-llm", 0

    async def generate(self, prompt: str, system: str | None = None,
                       temperature: float | None = None) -> LLMResponse:
        t0 = time.perf_counter()
        self.calls += 1
        await asyncio.sleep(self.latency_s)
        text = f"[mock answer] You asked: {prompt}"
        return LLMResponse(text, max(1, int(len(prompt.split()) * 1.3)),
                           max(1, int(len(text.split()) * 1.3)),
                           (time.perf_counter() - t0) * 1000, self.model)

    async def aclose(self) -> None:
        return None


def make_llm(s: Settings):
    if s.llm_provider == "mock":
        return MockLLM()
    if s.llm_provider == "gemini":
        return GeminiClient(s.gemini_api_key, s.gemini_model, s.gemini_base_url,
                            s.max_output_tokens, s.llm_timeout_s)
    raise ValueError(f"Unknown LLM_PROVIDER '{s.llm_provider}' (use gemini | mock)")
