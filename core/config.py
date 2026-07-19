"""Central settings, loaded from .env once. Import ``settings`` everywhere."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")


def _b(name: str, default: bool = False) -> bool:
    return os.getenv(name, str(int(default))).strip().lower() in ("1", "true", "yes")


@dataclass
class Settings:
    # --- LLM ---
    llm_provider: str = os.getenv("LLM_PROVIDER", "gemini")
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
    llm_model: str = os.getenv("LLM_MODEL", "gemini-2.5-flash")
    llm_vision_model: str = os.getenv("LLM_VISION_MODEL", "gemini-2.5-flash")
    embeddings_model: str = os.getenv("EMBEDDINGS_MODEL", "text-embedding-004")
    # local (deterministic, offline, zero-quota) | gemini (semantic, uses quota)
    embeddings_provider: str = os.getenv("EMBEDDINGS_PROVIDER", "local")
    # Rate-limit hygiene for the free tier
    llm_min_interval_s: float = float(os.getenv("LLM_MIN_INTERVAL_S", "6.0"))
    llm_max_retry_wait_s: float = float(os.getenv("LLM_MAX_RETRY_WAIT_S", "65.0"))

    # --- Stores ---
    graph_backend: str = os.getenv("GRAPH_BACKEND", "networkx")
    vector_backend: str = os.getenv("VECTOR_BACKEND", "local")

    # --- Paths ---
    data_dir: Path = ROOT / os.getenv("DATA_DIR", "data")
    llm_cache_dir: Path = ROOT / os.getenv("LLM_CACHE_DIR", "data/cache/llm")
    embed_cache_dir: Path = ROOT / os.getenv("EMBED_CACHE_DIR", "data/cache/embed")
    seed_dir: Path = field(default=ROOT / "data" / "seed")
    graph_path: Path = field(default=ROOT / "data" / "graph" / "graph.json")
    vector_dir: Path = field(default=ROOT / "data" / "vectors")

    # --- Demo reliability ---
    demo_strict: bool = _b("DEMO_STRICT", False)

    @property
    def has_llm_key(self) -> bool:
        return bool(self.gemini_api_key)

    def ensure_dirs(self) -> None:
        for p in (self.llm_cache_dir, self.embed_cache_dir,
                  self.graph_path.parent, self.vector_dir):
            p.mkdir(parents=True, exist_ok=True)


settings = Settings()
