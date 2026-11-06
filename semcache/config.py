"""Central configuration. Every setting can be overridden with an environment
variable (or a line in the .env file), so you never need to edit code."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def load_dotenv(path: str | Path = ".env") -> None:
    """Tiny .env loader (KEY=VALUE per line). Real environment variables win."""
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _s(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _i(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


def _f(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


@dataclass(frozen=True)
class Settings:
    # ---- LLM -------------------------------------------------------------
    llm_provider: str = "gemini"            # gemini | mock
    gemini_api_key: str = ""
    gemini_model: str = "gemini-3.5-flash-lite"
    gemini_base_url: str = "https://generativelanguage.googleapis.com/v1beta"
    max_output_tokens: int = 1024
    llm_timeout_s: float = 60.0
    # ---- Embeddings ------------------------------------------------------
    embedder: str = "minilm"                # minilm | gemini | hash
    embedder_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    gemini_embed_model: str = "gemini-embedding-001"
    # ---- Cache behaviour -------------------------------------------------
    cache_mode: str = "semantic+verify"     # exact | semantic | semantic+verify
    similarity_threshold: float = 0.85
    verifier: str = "rules"                 # none | rules | cross | rules+cross
    cross_encoder_model: str = "cross-encoder/quora-distilroberta-base"
    cross_encoder_threshold: float = 0.5
    ttl_seconds: int = 86400
    max_entries: int = 10000
    top_k: int = 5
    index_backend: str = "auto"             # auto | faiss | numpy
    persist_dir: str = "data/cache_store"
    autosave_every: int = 25
    # ---- Notional pricing (USD per 1M tokens) for the "money saved" metric.
    # Placeholders: set them to your model's real price. Free tier = $0 real cost.
    price_in_per_m: float = 0.30
    price_out_per_m: float = 2.50

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv()
        return cls(
            llm_provider=_s("LLM_PROVIDER", "gemini").lower(),
            gemini_api_key=_s("GEMINI_API_KEY", ""),
            gemini_model=_s("GEMINI_MODEL", cls.gemini_model),
            gemini_base_url=_s("GEMINI_BASE_URL", cls.gemini_base_url),
            max_output_tokens=_i("MAX_OUTPUT_TOKENS", cls.max_output_tokens),
            llm_timeout_s=_f("LLM_TIMEOUT_S", cls.llm_timeout_s),
            embedder=_s("EMBEDDER", "minilm").lower(),
            embedder_model=_s("EMBEDDER_MODEL", cls.embedder_model),
            gemini_embed_model=_s("GEMINI_EMBED_MODEL", cls.gemini_embed_model),
            cache_mode=_s("CACHE_MODE", cls.cache_mode).lower(),
            similarity_threshold=_f("SIMILARITY_THRESHOLD", cls.similarity_threshold),
            verifier=_s("VERIFIER", cls.verifier).lower(),
            cross_encoder_model=_s("CROSS_ENCODER_MODEL", cls.cross_encoder_model),
            cross_encoder_threshold=_f("CROSS_ENCODER_THRESHOLD", cls.cross_encoder_threshold),
            ttl_seconds=_i("TTL_SECONDS", cls.ttl_seconds),
            max_entries=_i("MAX_ENTRIES", cls.max_entries),
            top_k=_i("TOP_K", cls.top_k),
            index_backend=_s("INDEX_BACKEND", cls.index_backend).lower(),
            persist_dir=_s("PERSIST_DIR", cls.persist_dir),
            autosave_every=_i("AUTOSAVE_EVERY", cls.autosave_every),
            price_in_per_m=_f("PRICE_IN_PER_M", cls.price_in_per_m),
            price_out_per_m=_f("PRICE_OUT_PER_M", cls.price_out_per_m),
        )

    def public_dict(self) -> dict:
        """Settings safe to show in the API (the API key is never included)."""
        d = {k: getattr(self, k) for k in self.__dataclass_fields__}
        d["gemini_api_key"] = "set" if self.gemini_api_key else "missing"
        return d
