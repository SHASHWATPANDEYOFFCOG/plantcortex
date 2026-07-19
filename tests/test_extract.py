"""Tests for extraction: tags, failure codes, rule-based and structured extractors."""

from __future__ import annotations

from pipelines.m1_ingest.extract import (
    extract_tags, failure_codes, rule_based_extract, structured_extract,
)
from core.ontology import NodeType, EdgeType


def test_extract_tags_known_prefixes_only():
    tags = extract_tags("Pump P-101A feeds V-201 via L-2001; check PT-108.")
    assert "P-101A" in tags and "V-201" in tags and "PT-108" in tags
    assert "L-2001" not in tags       # L#### process lines are not Equipment


def test_failure_codes_priority():
    assert "ELP" in failure_codes("mechanical seal leakage on baseplate")
    assert failure_codes("brg noise high vibration")[0] == "VIB"
    assert "STU" in failure_codes("valve stuck partially open")


def test_rule_based_extract_equipment_and_failuremode():
    r = rule_based_extract("seal leak on P-101A with high vibration")
    types = {n.type for n in r.nodes}
    assert NodeType.EQUIPMENT in types and NodeType.FAILURE_MODE in types


def test_structured_sop_extract():
    text = ("SOP-17  Confined Space Entry Procedure  (Rev 1)\n"
            "1. Scope\nVessel entry V-201\n"
            "2. Applicable Equipment\nV-201\n"
            "3. Referenced Standards\nOISD-STD-105 clause 7.3\n"
            "4. Procedure\n1. Obtain permit.\n")
    r = structured_extract(text, "sop")
    proc = [n for n in r.nodes if n.type == NodeType.PROCEDURE]
    assert proc and proc[0].properties["sop_id"] == "SOP-17"
    assert any(e.type == EdgeType.GOVERNS and e.target_ref == "V-201" for e in r.edges)
    assert any(e.type == EdgeType.REQUIRES and e.target_ref == "OISD-STD-105:7.3"
               for e in r.edges)


def test_structured_incident_extract():
    text = ("Incident Report INC-2022-014\nIncident ID\nINC-2022-014\n"
            "Date\n2021-11-30\nSeverity\nnear-miss\nEquipment\nP-101A\n"
            "Description\nMechanical seal on P-101A failed, external leakage. "
            "High vibration reported earlier.\n")
    r = structured_extract(text, "incident")
    inc = [n for n in r.nodes if n.type == NodeType.INCIDENT]
    assert inc and inc[0].properties["incident_id"] == "INC-2022-014"
    assert inc[0].properties["severity"] == "near-miss"
    assert any(e.type == EdgeType.OCCURRED_AT and e.target_ref == "P-101A"
               for e in r.edges)


def test_structured_regulatory_extract():
    text = ("OISD-STD-105  Work Permit System\n"
            "Clause 7.3\nPrior to confined space entry, the atmosphere shall be "
            "tested for oxygen and flammable gas.\n"
            "Applicability (demo annotation)\nReferenced equipment: V-201, P-101A\n")
    r = structured_extract(text, "regulatory")
    clauses = [n for n in r.nodes if n.type == NodeType.REGULATORY_CLAUSE]
    assert clauses and clauses[0].properties["clause_no"] == "7.3"
    assert clauses[0].properties["standard"] == "OISD-STD-105"
    assert any(e.type == EdgeType.COVERS for e in r.edges)
