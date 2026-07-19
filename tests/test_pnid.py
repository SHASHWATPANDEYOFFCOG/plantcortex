"""D3 tests: P&ID geometric structural reading + hybrid ingest + N1 enrichment."""

from __future__ import annotations

from pathlib import Path

from core.embeddings import get_embeddings
from core.graph_repo import GraphRepo
from core.vector_repo import VectorStore
from pipelines.m1_ingest.pipeline import Repos
from pipelines.m2_pnid import geometric, vision
from pipelines.m2_pnid.pipeline import ingest_pnid

GOLD_CONNECTIONS = {
    frozenset(("T-401", "P-101A")), frozenset(("T-401", "P-101B")),
    frozenset(("P-101A", "FV-112")), frozenset(("P-101B", "FV-112")),
    frozenset(("FV-112", "E-301")), frozenset(("E-301", "V-201")),
    frozenset(("V-201", "C-501")),
}


def _pnid_paths(seed_corpus):
    seed = seed_corpus["dir"]
    entry = next(d for d in seed_corpus["manifest"]["documents"]
                 if d["doc_type"] == "pnid")
    return seed, entry


def test_geometric_read_recovers_exact_topology(seed_corpus):
    seed, entry = _pnid_paths(seed_corpus)
    r = geometric.geometric_read(seed / entry["filename"],
                                 seed / entry["layer_file"])
    found = {frozenset(c) for c in r["connections"]}
    assert found == GOLD_CONNECTIONS            # all 7 process lines, no signal line
    # the dashed PT-108 instrument signal must NOT be a process connection
    assert not any("PT-108" in c for c in r["connections"])


def test_cv_symbol_detection_reports_recall(seed_corpus):
    seed, entry = _pnid_paths(seed_corpus)
    r = geometric.geometric_read(seed / entry["filename"],
                                 seed / entry["layer_file"])
    cv = r["cv_detection"]
    assert cv["vector_symbols"] == 8
    assert cv["detected"]["circles"] >= 5       # best-effort pure-CV detection


def test_vision_normalize_shape():
    raw = {"symbols": [{"cls": "pump", "tag": "P-101A", "bbox": [1, 2, 3, 4]}],
           "tags": [{"text": "P-101A", "bbox": [1, 2, 3, 4]}],
           "connections": [["P-101A", "FV-112"], ["bad"]]}
    norm = vision._normalize(raw)
    assert norm["symbols"][0]["center"] == [2, 3]
    assert norm["connections"] == [["P-101A", "FV-112"]]   # malformed pair dropped


def _repos(tmp_path) -> Repos:
    return Repos(graph=GraphRepo(path=tmp_path / "g.json"),
                 vector=VectorStore(dir=tmp_path / "vec"), emb=get_embeddings())


def test_ingest_pnid_builds_equipment_and_connections(seed_corpus, tmp_path):
    seed, entry = _pnid_paths(seed_corpus)
    repos = _repos(tmp_path)
    ingest_pnid(repos, entry, seed, llm=None)
    g = repos.graph
    assert len(g.nodes_by_type("Equipment")) == 8
    conns = {frozenset((e["source"].split(":")[1], e["target"].split(":")[1]))
             for e in g.all_edges() if e["type"] == "CONNECTED_TO"}
    assert conns == GOLD_CONNECTIONS
    # equipment type came from the drawing; bbox provenance present for citations
    p = g.get_node("Equipment:P-101A")
    assert p["props"]["type"] == "pump"
    assert any("bbox" in pr for pr in p["provenance"])


def test_pnid_merges_into_existing_node_n1(seed_corpus, tmp_path):
    """N1: an existing tag-only P-101A is enriched, not duplicated, by the drawing."""
    seed, entry = _pnid_paths(seed_corpus)
    repos = _repos(tmp_path)
    repos.graph.upsert_node("Equipment:P-101A", "Equipment", {"tag": "P-101A"},
                            [{"doc_id": "DOC-WO-001", "extractor": "tlp"}])
    before = len(repos.graph.nodes_by_type("Equipment"))
    ingest_pnid(repos, entry, seed, llm=None)
    p = repos.graph.get_node("Equipment:P-101A")
    assert p["props"]["type"] == "pump"                     # enriched
    extractors = {pr.get("extractor") for pr in p["provenance"]}
    assert {"tlp", "geometry"} <= extractors                # two modalities, one node
    assert len(repos.graph.nodes_by_type("Equipment")) == before + 7  # +7 new tags
