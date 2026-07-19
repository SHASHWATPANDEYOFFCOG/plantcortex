"""Unit tests for the ontology: canonical keys, tag normalization, prompt spec."""

from __future__ import annotations

import pytest

from core.ontology import (
    Edge, EdgeType, Equipment, ExtractionResult, FailureMode, FailureModeCode,
    NodeType, Provenance, WorkOrder, normalize_tag, ontology_prompt_spec,
)


@pytest.mark.parametrize("raw,expected", [
    ("p-101 a", "P-101A"),
    ("P101A", "P-101A"),
    ("p101a", "P-101A"),
    ("V201", "V-201"),
    ("FV112", "FV-112"),
    ("pt 108", "PT-108"),
    ("L-2001", "L-2001"),
    ("P-101B", "P-101B"),
])
def test_normalize_tag_canonicalizes(raw, expected):
    assert normalize_tag(raw) == expected


def test_normalize_tag_passthrough_non_tags():
    # Non equipment-tag strings survive (upper-cased) rather than being mangled.
    assert normalize_tag("OISD-STD-105") == "OISD-STD-105"
    assert normalize_tag("") == ""


def test_equipment_key_is_canonical_regardless_of_spelling():
    # The N1 backbone: three spellings collapse to ONE key.
    a = Equipment(tag="p101a")
    b = Equipment(tag="P-101 A")
    c = Equipment(tag="P-101A", name="Feed Charge Pump A", type="pump")
    assert a.key() == b.key() == c.key() == "Equipment:P-101A"


def test_failuremode_and_workorder_keys():
    assert FailureMode(code=FailureModeCode.ELP).key() == "FailureMode:ELP"
    assert WorkOrder(wo_id="WO-1002").key() == "WorkOrder:WO-1002"


def test_edge_key_roundtrip():
    e = Edge(type=EdgeType.PERFORMED_ON,
             source_key="WorkOrder:WO-1002", target_key="Equipment:P-101A")
    assert e.key() == "WorkOrder:WO-1002-[PERFORMED_ON]->Equipment:P-101A"


def test_provenance_confidence_bounds():
    with pytest.raises(Exception):
        Provenance(doc_id="D1", confidence=1.5)


def test_prompt_spec_shape():
    spec = ontology_prompt_spec()
    assert set(spec) == {"node_types", "edge_types", "failure_mode_codes",
                         "tag_pattern_examples"}
    # Every ontology node type is represented for the extractor.
    assert set(spec["node_types"]) == {nt.value for nt in NodeType}
    # Internal-only SAME_AS must never be offered to the LLM.
    assert "SAME_AS" not in spec["edge_types"]
    assert "ELP" in spec["failure_mode_codes"]


def test_extraction_result_parses_llm_json():
    payload = {
        "nodes": [{"type": "Equipment",
                   "properties": {"tag": "P-101A"}, "confidence": 0.9,
                   "evidence_span": "pump P-101A"}],
        "edges": [{"type": "PERFORMED_ON", "source_ref": "WO-1002",
                   "target_ref": "P-101A", "confidence": 0.8}],
    }
    result = ExtractionResult.model_validate(payload)
    assert result.nodes[0].type == NodeType.EQUIPMENT
    assert result.edges[0].type == EdgeType.PERFORMED_ON
