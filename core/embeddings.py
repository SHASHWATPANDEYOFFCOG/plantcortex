"""Text embeddings: Gemini (cached to disk) with a deterministic offline fallback.

The fallback (hashed char n-grams) is not semantically strong, but it is fully
offline and deterministic — BM25 carries the exact-code retrieval load, so the
hybrid retriever still works without a network. Cached Gemini vectors and fallback
vectors never collide because the cache key includes the embedder id.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re

import numpy as np

from core.config import settings

log = logging.getLogger("plantcortex.embed")

DIM = 768  # Gemini text-embedding-004 dimensionality; fallback matches it.
_NGRAM = re.compile(r"[a-z0-9]+")


class Embeddings:
    def __init__(self) -> None:
        self.cache_dir = settings.embed_cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.model = settings.embeddings_model
        # Local hash embeddings by default: deterministic, offline, zero-quota.
        # Set EMBEDDINGS_PROVIDER=gemini to use the semantic API (batched, cached).
        use_gemini = (settings.embeddings_provider == "gemini"
                      and settings.has_llm_key and not settings.demo_strict)
        self._use_api = use_gemini
        self._embedder_id = self.model if use_gemini else "hash-fallback"
        self._client = None

    # -- fallback ---------------------------------------------------------- #
    @staticmethod
    def _hash_embed(text: str, dim: int = DIM) -> np.ndarray:
        vec = np.zeros(dim, dtype=np.float32)
        toks = _NGRAM.findall(text.lower())
        grams = toks + [a + "_" + b for a, b in zip(toks, toks[1:])]
        for g in grams:
            h = int(hashlib.md5(g.encode()).hexdigest(), 16)
            vec[h % dim] += 1.0
        n = np.linalg.norm(vec)
        return vec / n if n else vec

    # -- cache ------------------------------------------------------------- #
    def _key(self, text: str) -> str:
        return hashlib.sha256(f"{self._embedder_id}|{text}".encode()).hexdigest()[:24]

    def _read(self, text: str):
        p = self.cache_dir / f"{self._key(text)}.json"
        if p.exists():
            return np.array(json.loads(p.read_text())["v"], dtype=np.float32)
        return None

    def _write(self, text: str, vec: np.ndarray) -> None:
        (self.cache_dir / f"{self._key(text)}.json").write_text(
            json.dumps({"v": vec.tolist()}))

    # -- api --------------------------------------------------------------- #
    def _api_embed(self, texts: list[str]) -> list[np.ndarray]:
        if self._client is None:
            from google import genai
            self._client = genai.Client(api_key=settings.gemini_api_key)
        try:
            resp = self._client.models.embed_content(model=self.model, contents=texts)
            return [np.array(e.values, dtype=np.float32) for e in resp.embeddings]
        except Exception as e:  # noqa: BLE001
            log.warning("embed API failed (%s); using hash fallback", str(e)[:120])
            return [self._hash_embed(t) for t in texts]

    def embed(self, texts: list[str]) -> np.ndarray:
        out: list[np.ndarray | None] = [None] * len(texts)
        misses: list[int] = []
        for i, t in enumerate(texts):
            cached = self._read(t)
            if cached is not None:
                out[i] = cached
            else:
                misses.append(i)
        if misses:
            miss_texts = [texts[i] for i in misses]
            if self._use_api:
                vecs = self._api_embed(miss_texts)
            else:
                vecs = [self._hash_embed(t) for t in miss_texts]
            for i, v in zip(misses, vecs):
                self._write(texts[i], v)
                out[i] = v
        return np.vstack([o for o in out])  # type: ignore[arg-type]

    def embed_one(self, text: str) -> np.ndarray:
        return self.embed([text])[0]


_EMB: Embeddings | None = None


def get_embeddings() -> Embeddings:
    global _EMB
    if _EMB is None:
        _EMB = Embeddings()
    return _EMB
