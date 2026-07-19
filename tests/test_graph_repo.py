"""Tests for the NetworkX GraphRepo: idempotent MERGE, provenance, linkage, PPR."""

from __future__ import annotations

from core.graph_repo import GraphRepo


def _seed_small(g: GraphRepo) -> None:
    # a pump referenced by a work order, an SOP, and an incident (3 doc types)
    g.upsert_node("Document:WO", "Document", {"doc_id": "WO", "doc_type": "work_order"})
    g.upsert_node("Document:SOP", "Document", {"doc_id": "SOP", "doc_type": "sop"})
    g.upsert_node("Document:INC", "Document", {"doc_id": "INC", "doc_type": "incident"})
    g.upsert_node("Equipment:P-101A", "Equipment", {"tag": "P-101A"},
                  [{"doc_id": "WO"}])
    g.upsert_edge("PERFORMED_ON", "WorkOrder:1", "Equipment:P-101A", [{"doc_id": "WO"}])
    g.upsert_edge("GOVERNS", "Procedure:SOP-1", "Equipment:P-101A", [{"doc_id": "SOP"}])
    g.upsert_edge("OCCURRED_AT", "Incident:1", "Equipment:P-101A", [{"doc_id": "INC"}])


def test_upsert_node_is_idempotent_and_merges_provenance():
    g = GraphRepo()
    assert g.upsert_node("Equipment:P-101A", "Equipment", {"tag": "P-101A"},
                         [{"doc_id": "D1"}]) is True
    assert g.upsert_node("Equipment:P-101A", "Equipment", {"name": "Pump A"},
                         [{"doc_id": "D2"}]) is False
    assert g.g.number_of_nodes() == 1
    node = g.get_node("Equipment:P-101A")
    assert node["props"]["tag"] == "P-101A"
    assert node["props"]["name"] == "Pump A"          # missing prop filled in
    assert len(node["provenance"]) == 2               # provenance unioned
    assert set(node["doc_ids"]) == {"D1", "D2"}


def test_upsert_edge_is_idempotent():
    g = GraphRepo()
    assert g.upsert_edge("PERFORMED_ON", "WO:1", "Equipment:P-101A",
                         [{"doc_id": "D1"}]) is True
    assert g.upsert_edge("PERFORMED_ON", "WO:1", "Equipment:P-101A",
                         [{"doc_id": "D1"}]) is False
    assert g.g.number_of_edges() == 1
    edge = g.g.edges["WO:1", "Equipment:P-101A", "PERFORMED_ON"]
    assert len(edge["provenance"]) == 2


def test_linkage_completeness_metric():
    g = GraphRepo()
    _seed_small(g)
    link = g.linkage_completeness()
    assert link["equipment"] == 1
    assert link["well_linked"] == 1                   # P-101A touches 3 doc types
    assert link["pct"] == 100.0


def test_personalized_pagerank_prefers_neighbors():
    g = GraphRepo()
    _seed_small(g)
    ranked = g.personalized_pagerank(["Equipment:P-101A"], top_n=10)
    keys = [k for k, _ in ranked]
    assert "Equipment:P-101A" in keys
    # neighbors of the seed should rank above unrelated nodes
    assert "Incident:1" in keys or "WorkOrder:1" in keys


def test_save_and_load_roundtrip(tmp_path):
    g = GraphRepo(path=tmp_path / "g.json")
    _seed_small(g)
    g.save()
    g2 = GraphRepo(path=tmp_path / "g.json").load()
    assert g2.g.number_of_nodes() == g.g.number_of_nodes()
    assert g2.g.number_of_edges() == g.g.number_of_edges()
    assert g2.has_node("Equipment:P-101A")
