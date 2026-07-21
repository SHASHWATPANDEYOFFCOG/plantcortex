"""D5 tests: compliance gap agent, equipment dossier, knowledge capture."""

from __future__ import annotations


from agents import m5_compliance
from agents.dossier import build_dossier
from pipelines.m1_ingest.pipeline import capture_note


# --------------------------------------------------------------------------- #
# M5 compliance
# --------------------------------------------------------------------------- #
def test_compliance_flags_sop17_gas_testing_gap(ingested_repos, seed_corpus):
    rep = m5_compliance.scan(ingested_repos, seed_corpus["dir"],
                             seed_corpus["manifest"], "OISD-STD-105")
    gaps = [v for v in rep.verdicts if v.verdict == "GAP"]
    assert len(gaps) == 1                              # exactly the seeded gap
    gap = gaps[0]
    assert gap.clause_no == "7.3"
    assert gap.procedure == "SOP-17"                   # the responsible procedure
    covered = {v.clause_no for v in rep.verdicts if v.verdict == "covered"}
    assert {"5.1", "6.2"} <= covered                   # no false positives


def test_compliance_coverage_scoring():
    # clause requiring gas testing vs a procedure that omits it -> GAP
    _score, verdict = m5_compliance.assess(
        "the atmosphere shall be tested for oxygen and flammable gas before entry",
        "Obtain permit, isolate the vessel, ventilate, post an attendant, log entry")
    assert verdict == "GAP"
    _score, verdict2 = m5_compliance.assess(
        "a valid work permit shall be obtained before the job",
        "Raise a work permit and place the loop under maintenance")
    assert verdict2 == "covered"


def test_compliance_export_pdf(ingested_repos, seed_corpus, tmp_path):
    rep = m5_compliance.scan(ingested_repos, seed_corpus["dir"],
                             seed_corpus["manifest"], "OISD-STD-105")
    out = m5_compliance.export_pdf(rep, tmp_path / "evidence.pdf")
    assert out.exists() and out.stat().st_size > 800


# --------------------------------------------------------------------------- #
# M4 dossier
# --------------------------------------------------------------------------- #
def test_dossier_assembles_full_asset_view(ingested_repos):
    d = build_dossier(ingested_repos.graph, "p101a")     # lowercase resolves via N1
    assert d["found"] and d["tag"] == "P-101A"
    assert d["type"] == "pump"
    assert d["work_order_count"] > 0 and len(d["work_orders"]) <= 5
    assert any(p["sop_id"] == "SOP-21" for p in d["procedures"])
    assert d["incidents"]                                 # P-101A near-misses
    assert any(n["note_id"] == "VN-001" for n in d["tacit_notes"])
    assert d["inspection"] and d["inspection"]["overdue"] is True
    assert "T-401" in d["connected_to"]                  # from the P&ID
    assert any(fm["code"] == "ELP" for fm in d["failure_modes"])


def test_dossier_missing_tag():
    from core.graph_repo import GraphRepo
    assert build_dossier(GraphRepo(), "Z-999")["found"] is False


# --------------------------------------------------------------------------- #
# M7 knowledge capture
# --------------------------------------------------------------------------- #
def test_capture_creates_citable_tacit_note(ingested_repos):
    note = ("Reminder from the day shift: watch P-101A closely, the mechanical seal "
            "leaks when vibration climbs.")
    res = capture_note(ingested_repos, note, author_role="Shift Supervisor")
    assert res["note_id"].startswith("VN-")
    assert "P-101A" in res["equipment"]
    g = ingested_repos.graph
    key = f"TacitNote:{res['note_id']}"
    assert g.has_node(key)
    about = {e["target"] for e in g.edges_incident(key) if e["type"] == "ABOUT"}
    assert "Equipment:P-101A" in about                   # immediately linked/citable
