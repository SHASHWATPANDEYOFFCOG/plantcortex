"""End-to-end offline ingestion into an isolated graph (no LLM, no network)."""

from __future__ import annotations

import pandas as pd

from core.embeddings import get_embeddings
from core.graph_repo import GraphRepo
from core.vector_repo import VectorStore
from pipelines.m1_ingest.pipeline import Repos, ingest_document


def _repos(tmp_path) -> Repos:
    return Repos(graph=GraphRepo(path=tmp_path / "g.json"),
                 vector=VectorStore(dir=tmp_path / "vec"),
                 emb=get_embeddings())


def _entry(manifest, doc_id):
    return next(d for d in manifest["documents"] if d["doc_id"] == doc_id)


def test_ingest_prose_and_voice_builds_linked_graph(seed_corpus, tmp_path):
    repos = _repos(tmp_path)
    seed = seed_corpus["dir"]
    for doc_id in ("SOP-17", "INC-2022-014", "VN-001", "OISD-STD-105"):
        ingest_document(repos, _entry(seed_corpus["manifest"], doc_id), seed, llm=None)
    g = repos.graph

    # Procedure + GOVERNS + REQUIRES from the SOP
    assert g.has_node("Procedure:SOP-17")
    assert g.has_node("Equipment:V-201")
    assert any(t == "GOVERNS" for t, _, _ in g.neighbors("Procedure:SOP-17"))
    assert g.has_node("RegulatoryClause:OISD-STD-105:7.3")

    # Incident occurred at P-101A
    assert g.has_node("Incident:INC-2022-014")
    inc_edges = {(e["type"], e["target"]) for e in g.edges_incident("Incident:INC-2022-014")}
    assert ("OCCURRED_AT", "Equipment:P-101A") in inc_edges

    # Tacit voice note is ABOUT P-101A  (Knowledge Capture threading)
    assert g.has_node("TacitNote:VN-001")
    about = {e["target"] for e in g.edges_incident("TacitNote:VN-001")
             if e["type"] == "ABOUT"}
    assert "Equipment:P-101A" in about

    # Documents + chunks were indexed for retrieval
    assert len(repos.vector) > 0
    assert g.nodes_by_type("Document")


def test_ingest_work_orders_deterministic(seed_corpus, tmp_path):
    # tiny synthetic WO sheet with a seal failure + a vibration WO on P-101A
    d = tmp_path / "wo"
    d.mkdir()
    pd.DataFrame([
        {"WO_ID": "WO-9001", "Date": "2023-01-10", "Equipment_Tag": "P-101A",
         "Type": "corrective", "Problem_Text": "mech seel lkg p101a seel failure",
         "Action_Text": "chng mech seel", "Technician": "R. Sharma"},
        {"WO_ID": "WO-9002", "Date": "2023-01-02", "Equipment_Tag": "P-101A",
         "Type": "corrective", "Problem_Text": "brg noise hi vib",
         "Action_Text": "monitored", "Technician": "A. Khan"},
    ]).to_excel(d / "wo.xlsx", index=False)

    repos = _repos(tmp_path)
    entry = {"doc_id": "DOC-WO-T", "filename": "wo.xlsx", "doc_type": "work_order",
             "source_kind": "xlsx"}
    ingest_document(repos, entry, d, llm=None)
    g = repos.graph
    assert g.has_node("WorkOrder:WO-9001") and g.has_node("Equipment:P-101A")
    # seal-failure WO exhibits external leakage; vibration WO exhibits VIB
    e1 = {(e["type"], e["target"]) for e in g.edges_incident("WorkOrder:WO-9001")}
    assert ("PERFORMED_ON", "Equipment:P-101A") in e1
    assert ("EXHIBITS", "FailureMode:ELP") in e1
    e2 = {(e["type"], e["target"]) for e in g.edges_incident("WorkOrder:WO-9002")}
    assert ("EXHIBITS", "FailureMode:VIB") in e2
