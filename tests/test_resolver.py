"""Tests for cross-modal entity resolution (N1)."""

from __future__ import annotations

from core.graph_repo import GraphRepo
from core.ontology import (
    ExtractedEdge, ExtractedNode, ExtractionResult, NodeType, EdgeType,
)
from core.resolver import (
    canonical_key, fuzzy_equipment_match, key_for_ref, resolve_extraction,
    ResolveCtx, _edit_distance_le1,
)


def test_canonical_key_equipment_normalizes():
    assert canonical_key(NodeType.EQUIPMENT, {"tag": "p101a"}) == "Equipment:P-101A"
    assert canonical_key(NodeType.EQUIPMENT, {"tag": "P-101 A"}) == "Equipment:P-101A"


def test_canonical_key_regclause_and_component():
    assert canonical_key(NodeType.REGULATORY_CLAUSE,
                         {"standard": "OISD-STD-105", "clause_no": "7.3"}) \
        == "RegulatoryClause:OISD-STD-105:7.3"
    assert canonical_key(NodeType.COMPONENT,
                         {"name": "Mechanical Seal", "parent_equipment": "p101a"}) \
        == "Component:P-101A:mechanical seal"


def test_key_for_ref_variants():
    assert key_for_ref(NodeType.FAILURE_MODE, "elp") == "FailureMode:ELP"
    assert key_for_ref(NodeType.EQUIPMENT, "v201") == "Equipment:V-201"
    assert key_for_ref(NodeType.WORK_ORDER, "WO-1002") == "WorkOrder:WO-1002"


def test_edit_distance_le1():
    assert _edit_distance_le1("P-101A", "P-101A")
    assert _edit_distance_le1("P101A", "P1O1A")        # single substitution
    assert _edit_distance_le1("P-101A", "P-1013A")     # single insertion
    assert not _edit_distance_le1("P-101A", "P-202B")


def test_fuzzy_equipment_match_same_prefix_only():
    existing = ["P-101A", "V-201"]
    # single-char substitution within the same prefix class -> match
    assert fuzzy_equipment_match("P-101A", ["P-101B"]) == "P-101B"
    # different prefix class -> never match, even if close
    assert fuzzy_equipment_match("X-999", existing) is None
    # too far apart -> no match
    assert fuzzy_equipment_match("P-777C", ["P-101A"]) is None


def test_resolve_extraction_builds_typed_delta():
    graph = GraphRepo()
    extraction = ExtractionResult(
        nodes=[
            ExtractedNode(type=NodeType.INCIDENT,
                          properties={"incident_id": "INC-1", "severity": "minor"},
                          confidence=1.0),
            ExtractedNode(type=NodeType.EQUIPMENT,
                          properties={"tag": "p101a"}, confidence=0.9),
            ExtractedNode(type=NodeType.FAILURE_MODE,
                          properties={"code": "ELP"}, confidence=0.8),
        ],
        edges=[
            ExtractedEdge(type=EdgeType.OCCURRED_AT, source_ref="INC-1",
                          target_ref="p101a", confidence=1.0),
            ExtractedEdge(type=EdgeType.EXHIBITS, source_ref="INC-1",
                          target_ref="ELP", confidence=0.8),
        ],
    )
    ctx = ResolveCtx(doc_id="INC-1", extractor="llm_text", page=1)
    delta = resolve_extraction(extraction, ctx, graph)
    keys = {n["key"] for n in delta.nodes}
    assert keys == {"Incident:INC-1", "Equipment:P-101A", "FailureMode:ELP"}
    edge_set = {(e["type"], e["source"], e["target"]) for e in delta.edges}
    assert ("OCCURRED_AT", "Incident:INC-1", "Equipment:P-101A") in edge_set
    assert ("EXHIBITS", "Incident:INC-1", "FailureMode:ELP") in edge_set
    # provenance carries the doc id
    assert delta.nodes[0]["provenance"][0]["doc_id"] == "INC-1"


def test_resolve_reuses_existing_equipment_via_fuzzy():
    graph = GraphRepo()
    graph.upsert_node("Equipment:P-101A", "Equipment", {"tag": "P-101A"}, [])
    extraction = ExtractionResult(nodes=[
        ExtractedNode(type=NodeType.EQUIPMENT, properties={"tag": "P101A"},
                      confidence=0.9)])
    ctx = ResolveCtx(doc_id="D2")
    delta = resolve_extraction(extraction, ctx, graph)
    # 'P101A' resolves to the existing canonical node (exact after normalization)
    assert delta.nodes[0]["key"] == "Equipment:P-101A"
