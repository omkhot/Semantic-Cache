from __future__ import annotations

import asyncio
import sys

import httpx

from semcache.config import Settings
from semcache.llm import GeminiClient, LLMError


def main() -> int:
    s = Settings.from_env()
    if not s.gemini_api_key:
        print("GEMINI_API_KEY is empty"
              "(free key: https://aistudio.google.com/apikey).")
        return 1
    try:
        r = httpx.get(f"{s.gemini_base_url}/models", params={"pageSize": 200},
                      headers={"x-goog-api-key": s.gemini_api_key}, timeout=30)
        r.raise_for_status()
        names = sorted(m["name"].removeprefix("models/") for m in r.json().get("models", [])
                       if "generateContent" in m.get("supportedGenerationMethods", []))
        print(f"Models your key can use for text generation ({len(names)}):")
        for n in names:
            print("  ", n, "  <-- configured" if n == s.gemini_model else "")
    except Exception as exc:
        print(f"Could not list models: {exc}")
        return 1

    async def ping():
        c = GeminiClient(s.gemini_api_key, s.gemini_model, s.gemini_base_url, 256, 30)
        try:
            return await c.generate("Reply with the single word: ready")
        finally:
            await c.aclose()

    try:
        out = asyncio.run(ping())
    except LLMError as exc:
        print(f"\nTest call to '{s.gemini_model}' FAILED: {exc}")
        return 1
    print(f"\nTest call to '{s.gemini_model}' OK -> {out.text!r} ({out.latency_ms:.0f} ms)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
