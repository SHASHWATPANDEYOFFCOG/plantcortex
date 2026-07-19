"""VectorStore — embedded dense (numpy cosine) + BM25, fused with RRF.

BM25 matters for exact industrial codes (``OISD-STD-105``, ``P-101A``) that dense
embeddings blur; dense matters for paraphrase. Reciprocal-Rank Fusion combines them
without score-scale tuning. Persisted to disk so ingestion is a one-time cost.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from rank_bm25 import BM25Okapi

from core.config import settings
from core.embeddings import get_embeddings

_TOK = re.compile(r"[a-z0-9][a-z0-9\-/]*")


def tokenize(text: str) -> list[str]:
    """Keep hyphenated codes whole so BM25 can match P-101A / OISD-STD-105 exactly."""
    return _TOK.findall(text.lower())


@dataclass
class VectorStore:
    dir: Path = field(default_factory=lambda: settings.vector_dir)
    ids: list[str] = field(default_factory=list)
    texts: list[str] = field(default_factory=list)
    metas: list[dict] = field(default_factory=list)
    _emb: Optional[np.ndarray] = None            # (n, d)
    _bm25: Optional[BM25Okapi] = None
    _dirty_bm25: bool = True

    # -- mutation ---------------------------------------------------------- #
    def add(self, chunk_id: str, text: str, embedding: np.ndarray,
            meta: dict) -> None:
        if chunk_id in self.ids:      # idempotent: re-ingest must not duplicate chunks
            return
        self.ids.append(chunk_id)
        self.texts.append(text)
        self.metas.append(meta)
        emb = embedding.reshape(1, -1).astype(np.float32)
        self._emb = emb if self._emb is None else np.vstack([self._emb, emb])
        self._dirty_bm25 = True

    def add_batch(self, chunk_ids: list[str], texts: list[str],
                  embeddings: np.ndarray, metas: list[dict]) -> None:
        self.ids.extend(chunk_ids)
        self.texts.extend(texts)
        self.metas.extend(metas)
        emb = embeddings.astype(np.float32)
        self._emb = emb if self._emb is None else np.vstack([self._emb, emb])
        self._dirty_bm25 = True

    def __len__(self) -> int:
        return len(self.ids)

    def get(self, chunk_id: str) -> Optional[dict]:
        try:
            i = self.ids.index(chunk_id)
        except ValueError:
            return None
        return {"chunk_id": chunk_id, "text": self.texts[i], "meta": self.metas[i]}

    # -- search ------------------------------------------------------------ #
    def _ensure_bm25(self) -> None:
        if self._dirty_bm25 or self._bm25 is None:
            self._bm25 = BM25Okapi([tokenize(t) for t in self.texts] or [[""]])
            self._dirty_bm25 = False

    def _norm(self, v: np.ndarray) -> np.ndarray:
        n = np.linalg.norm(v)
        return v / n if n else v

    def search_dense(self, query_vec: np.ndarray, k: int = 8) -> list[tuple[int, float]]:
        if self._emb is None or len(self) == 0:
            return []
        q = self._norm(query_vec.astype(np.float32))
        mat = self._emb
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        sims = (mat / norms) @ q
        idx = np.argsort(-sims)[:k]
        return [(int(i), float(sims[i])) for i in idx]

    def search_bm25(self, query: str, k: int = 8) -> list[tuple[int, float]]:
        if len(self) == 0:
            return []
        self._ensure_bm25()
        scores = self._bm25.get_scores(tokenize(query))
        idx = np.argsort(-scores)[:k]
        return [(int(i), float(scores[i])) for i in idx]

    def hybrid(self, query: str, k: int = 8, rrf_k: int = 60,
               query_vec: Optional[np.ndarray] = None) -> list[dict]:
        """RRF-fuse dense + BM25. Returns hit dicts with id/text/meta/score."""
        if query_vec is None:
            query_vec = get_embeddings().embed_one(query)
        dense = self.search_dense(query_vec, k=max(k, 10))
        lexical = self.search_bm25(query, k=max(k, 10))
        rrf: dict[int, float] = {}
        for rank, (i, _) in enumerate(dense):
            rrf[i] = rrf.get(i, 0.0) + 1.0 / (rrf_k + rank + 1)
        for rank, (i, _) in enumerate(lexical):
            rrf[i] = rrf.get(i, 0.0) + 1.0 / (rrf_k + rank + 1)
        ranked = sorted(rrf.items(), key=lambda x: -x[1])[:k]
        return [{"chunk_id": self.ids[i], "text": self.texts[i],
                 "meta": self.metas[i], "score": s} for i, s in ranked]

    # -- persistence ------------------------------------------------------- #
    def save(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        if self._emb is not None:
            np.save(self.dir / "embeddings.npy", self._emb)
        (self.dir / "store.json").write_text(json.dumps(
            {"ids": self.ids, "texts": self.texts, "metas": self.metas}),
            encoding="utf-8")

    @classmethod
    def load(cls, dir: Optional[Path] = None) -> "VectorStore":
        d = dir or settings.vector_dir
        store = cls(dir=d)
        meta_path = d / "store.json"
        if meta_path.exists():
            data = json.loads(meta_path.read_text(encoding="utf-8"))
            store.ids, store.texts, store.metas = (
                data["ids"], data["texts"], data["metas"])
        emb_path = d / "embeddings.npy"
        if emb_path.exists():
            store._emb = np.load(emb_path)
        return store
