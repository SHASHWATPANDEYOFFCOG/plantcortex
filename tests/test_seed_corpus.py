"""Integrity tests for the generated demo corpus.

These assert the *internal consistency* that makes the demo work: the same tags
thread across doc types (N1), the seeded seal/vibration pattern is present (M6/N3),
the SOP-17 compliance gap exists (M5), and the P-101A inspection is overdue (T2).
"""

from __future__ import annotations

import datetime as dt

import pandas as pd

from core.ontology import normalize_tag
from scripts.generate_seed_corpus import REGS, SOPS


# --------------------------------------------------------------------------- #
# Manifest & gold shape
# --------------------------------------------------------------------------- #
def test_manifest_has_all_doc_types(seed_corpus):
    docs = seed_corpus["manifest"]["documents"]
    types = {d["doc_type"] for d in docs}
    assert {"pnid", "work_order", "sop", "incident", "regulatory",
            "inspection", "voice_note"} <= types
    assert len(docs) >= 25


def test_gold_set_is_substantial_and_typed(seed_corpus):
    gold = seed_corpus["gold"]
    assert len(gold["nodes"]) >= 50           # spec: >= 50 entities/relations
    node_types = {n["type"] for n in gold["nodes"]}
    assert {"Equipment", "Component", "FailureMode", "WorkOrder", "Procedure",
            "RegulatoryClause", "Incident", "TacitNote"} <= node_types
    edge_types = {e["type"] for e in gold["edges"]}
    # The FMEA causal chain and the P&ID topology must both be present.
    assert {"HAS_CAUSE", "CONNECTED_TO", "PERFORMED_ON", "EXHIBITS",
            "GOVERNS", "REQUIRES", "COVERS", "OCCURRED_AT", "ABOUT"} <= edge_types


def test_every_equipment_endpoint_resolves_to_a_node(seed_corpus):
    """No dangling Equipment reference — the graph is internally consistent."""
    gold = seed_corpus["gold"]
    equip_keys = {n["key"] for n in gold["nodes"] if n["type"] == "Equipment"}
    for e in gold["edges"]:
        for endpoint in (e["source"], e["target"]):
            if endpoint.startswith("Equipment:"):
                assert endpoint in equip_keys, f"dangling equipment ref {endpoint}"


# --------------------------------------------------------------------------- #
# N1 — cross-modal threading of P-101A
# --------------------------------------------------------------------------- #
def test_p101a_threads_across_many_doc_types(seed_corpus):
    gold, manifest = seed_corpus["gold"], seed_corpus["manifest"]
    doc_type = {d["doc_id"]: d["doc_type"] for d in manifest["documents"]}
    target = "Equipment:P-101A"
    doc_ids: set[str] = set()
    for e in gold["edges"]:
        if target in (e["source"], e["target"]):
            doc_ids.update(e["doc_ids"])
    doc_types = {doc_type.get(d) for d in doc_ids}
    # P-101A must appear via the drawing, work orders, an SOP, an incident, voice...
    assert len([t for t in doc_types if t]) >= 4, doc_types


# --------------------------------------------------------------------------- #
# M6 / N3 — the seeded seal-failure + vibration precursor pattern
# --------------------------------------------------------------------------- #
def _load_wo(seed_corpus) -> pd.DataFrame:
    df = pd.read_csv(seed_corpus["dir"] / "work_orders" / "work_orders.csv")
    df["Date"] = pd.to_datetime(df["Date"]).dt.date
    return df


def test_four_p101a_seal_failures(seed_corpus):
    df = _load_wo(seed_corpus)
    seals = df[(df["Equipment_Tag"] == "P-101A")
               & df["Problem_Text"].str.contains("seel failure", case=False)]
    assert len(seals) == 4


def test_three_seal_failures_have_vibration_precursor_within_30d(seed_corpus):
    df = _load_wo(seed_corpus)
    p = df[df["Equipment_Tag"] == "P-101A"]
    seal_dates = sorted(p[p["Problem_Text"].str.contains("seel failure", case=False)]
                        ["Date"].tolist())
    vib_dates = sorted(p[p["Problem_Text"].str.contains("hi vib", case=False)]
                       ["Date"].tolist())
    preceded = 0
    for sd in seal_dates:
        if any(dt.timedelta(0) < (sd - vd) <= dt.timedelta(days=30)
               for vd in vib_dates):
            preceded += 1
    assert preceded == 3, f"expected 3 preceded seal failures, got {preceded}"


# --------------------------------------------------------------------------- #
# M5 — the deliberate SOP-17 compliance gap
# --------------------------------------------------------------------------- #
def _gas_test_terms(text: str) -> bool:
    t = text.lower()
    return any(k in t for k in ("gas test", "gas tested", "atmosphere shall be tested",
                                "oxygen", "lel", "atmospheric"))


def test_sop17_omits_gas_testing_but_standard_requires_it():
    sop17 = next(s for s in SOPS if s["sop_id"] == "SOP-17")
    steps_text = " ".join(sop17["steps"])
    assert not _gas_test_terms(steps_text), "SOP-17 should NOT contain gas testing"

    oisd = next(r for r in REGS if r["std"] == "OISD-STD-105")
    cl73 = dict(oisd["clauses"])["7.3"]
    assert _gas_test_terms(cl73), "OISD-STD-105 cl 7.3 must require gas testing"
    # And SOP-17 is bound to exactly that clause -> a detectable GAP.
    assert ("OISD-STD-105", "7.3") in sop17["requires"]


# --------------------------------------------------------------------------- #
# T2 join — overdue P-101A inspection
# --------------------------------------------------------------------------- #
def test_p101a_inspection_is_overdue(seed_corpus):
    insp = [d for d in seed_corpus["manifest"]["documents"]
            if d["doc_type"] == "inspection"]
    p101a = [d for d in insp if d.get("equipment") == "P-101A"]
    assert p101a and p101a[0]["overdue"] is True
    # All the other seed inspections are within period (single overdue signal).
    assert sum(1 for d in insp if d.get("overdue")) == 1


def test_tags_are_all_canonical_in_gold(seed_corpus):
    for n in seed_corpus["gold"]["nodes"]:
        if n["type"] == "Equipment":
            tag = n["props"]["tag"]
            assert normalize_tag(tag) == tag
