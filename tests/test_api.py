"""Smoke tests for the FastAPI ingestion surface (offline)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from api.main import app
from core.config import settings


@pytest.fixture(scope="module")
def client(seed_corpus):
    with TestClient(app) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_graph_summary_shape(client):
    r = client.get("/graph/summary")
    assert r.status_code == 200
    body = r.json()
    assert "node_types" in body and "edge_types" in body and "linkage" in body


def test_ingest_endpoint_adds_document_and_broadcasts_delta(client):
    sop = settings.seed_dir / "sops" / "SOP-17.pdf"
    with sop.open("rb") as f:
        r = client.post("/ingest", files={"file": ("SOP-17.pdf", f,
                                                    "application/pdf")})
    assert r.status_code == 200
    body = r.json()
    assert body["ingested"] == "SOP-17"
    assert body["doc_type"] == "sop"
    assert body["deltas"] and body["deltas"][0]["type"] == "graph.delta"
    assert body["summary"]["nodes"] > 0


def test_ask_endpoint_returns_answer_contract(client):
    r = client.post("/ask", json={"question": "What does OISD-STD-105 clause 7.3 "
                                              "require for confined space entry?"})
    assert r.status_code == 200
    body = r.json()
    assert set(("answer_markdown", "citations", "confidence", "mode")) <= set(body)
    assert body["citations"]


def test_ask_baseline_vs_hybrid_diverge(client):
    q = ("Which pump had repeated seal failures and an overdue inspection and "
         "appears in a near-miss report?")
    hybrid = client.post("/ask", json={"question": q}).json()
    baseline = client.post("/ask", json={"question": q, "baseline": True}).json()
    assert hybrid["mode"] == "multihop" and hybrid["seeds"]
    assert baseline["mode"] == "lookup" and not baseline["path"]


def test_field_page_served(client):
    r = client.get("/field")
    assert r.status_code == 200 and "Field" in r.text


def test_equipment_dossier_endpoint(client):
    r = client.get("/equipment/p101a")
    assert r.status_code == 200
    d = r.json()
    assert d["found"] and d["tag"] == "P-101A" and d["type"] == "pump"
    assert d["work_order_count"] > 0


def test_capture_endpoint_creates_note(client):
    r = client.post("/capture", json={"transcript": "Watch P-101A seal on high vibration."})
    assert r.status_code == 200
    body = r.json()
    assert body["note_id"].startswith("VN-") and "P-101A" in body["equipment"]


def test_compliance_scan_endpoint_flags_gap(client):
    r = client.post("/compliance/scan", json={"standard_id": "OISD-STD-105"})
    assert r.status_code == 200
    body = r.json()
    assert body["summary"]["GAP"] == 1
    gap = next(v for v in body["verdicts"] if v["verdict"] == "GAP")
    assert gap["clause_no"] == "7.3" and gap["procedure"] == "SOP-17"


def test_compliance_report_pdf(client):
    r = client.get("/compliance/report/OISD-STD-105")
    assert r.status_code == 200
    assert r.headers["content-type"] == "application/pdf"
    assert len(r.content) > 800


def test_timeline_endpoint_for_time_lens(client):
    r = client.get("/timeline/P-101A")
    assert r.status_code == 200
    body = r.json()
    assert body["tag"] == "P-101A" and len(body["events"]) > 10
    assert all(e["date"] for e in body["events"])
    assert any("ELP" in e["codes"] for e in body["events"])      # seal failures present
    assert any("VIB" in e["codes"] for e in body["events"])      # precursors present


def test_websocket_sends_hello_with_summary(client):
    with client.websocket_connect("/ws") as ws:
        msg = ws.receive_json()
        assert msg["type"] == "hello"
        assert "summary" in msg and "nodes" in msg["summary"]


def test_subgraph_around_ingested_procedure(client):
    # ensure SOP-17 is present, then fetch its ego graph
    client.post("/ingest", files={"file": ("SOP-17.pdf",
                (settings.seed_dir / "sops" / "SOP-17.pdf").read_bytes(),
                "application/pdf")})
    r = client.get("/graph/subgraph", params={"center": "Procedure:SOP-17", "hops": 1})
    assert r.status_code == 200
    ids = {n["id"] for n in r.json()["nodes"]}
    assert "Procedure:SOP-17" in ids
