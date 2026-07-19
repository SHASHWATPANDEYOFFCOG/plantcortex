"""Shared fixtures. Ensures the seed corpus exists before integrity tests run."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SEED_DIR = ROOT / "data" / "seed"


@pytest.fixture(scope="session")
def seed_corpus():
    """Generate the corpus once if it isn't already present, then expose paths."""
    if not (SEED_DIR / "manifest.json").exists():
        from scripts.generate_seed_corpus import main
        main()
    manifest = json.loads((SEED_DIR / "manifest.json").read_text(encoding="utf-8"))
    gold = json.loads(
        (SEED_DIR / "gold" / "gold_extraction.json").read_text(encoding="utf-8"))
    return {"dir": SEED_DIR, "manifest": manifest, "gold": gold}


@pytest.fixture(scope="session")
def ingested_repos(seed_corpus, tmp_path_factory):
    """Full corpus ingested offline into an isolated temp dir (shared across tests)."""
    from core.embeddings import get_embeddings
    from core.graph_repo import GraphRepo
    from core.vector_repo import VectorStore
    from pipelines.m1_ingest.pipeline import Repos, ingest_corpus

    d = tmp_path_factory.mktemp("graph")
    repos = Repos(graph=GraphRepo(path=d / "g.json"),
                  vector=VectorStore(dir=d / "v"), emb=get_embeddings())
    ingest_corpus(repos, seed_corpus["dir"], llm=None)
    return repos
