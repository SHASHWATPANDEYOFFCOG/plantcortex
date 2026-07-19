"""D4 tests: router, PPR multi-hop, citations, baseline-vs-hybrid, refusal, global.

Builds a small isolated graph so the retrieval logic is tested deterministically
without depending on the full corpus ingest.
"""

from __future__ import annotations

from core.embeddings import get_embeddings
from core.graph_repo import GraphRepo
from core.vector_repo import VectorStore
from pipelines.m1_ingest.pipeline import Repos
from pipelines.m3_retrieval import router
from pipelines.m3_retrieval.engine import ask


def _mini_repos() -> Repos:
    emb = get_embeddings()   # local hash embedder (offline, deterministic, 768-d)
    g = GraphRepo()
    # equipment referenced across doc types
    g.upsert_node("Equipment:P-101A", "Equipment", {"tag": "P-101A", "type": "pump"})
    g.upsert_node("FailureMode:ELP", "FailureMode", {"code": "ELP"})
    g.upsert_node("Incident:INC-1", "Incident", {"incident_id": "INC-1",
                                                 "severity": "near-miss"})
    for cid, text, ent in [
        ("c_seal", "P-101A mechanical seal failure external leakage on baseplate", "Equipment:P-101A"),
        ("c_insp", "Inspection record for P-101A is OVERDUE", "Equipment:P-101A"),
        ("c_near", "Near-miss incident INC-1 seal leak on P-101A", "Incident:INC-1"),
        ("c_noise", "P-101A routine bearing noise check within limit", "Equipment:P-101A"),
    ]:
        ck = f"Chunk:{cid}"
        g.upsert_node(ck, "Chunk", {"chunk_id": cid, "doc_id": cid.upper(),
                                    "page": 1, "text": text})
        g.upsert_edge("MENTIONS", ck, ent, [{"doc_id": cid.upper()}])
    g.upsert_edge("OCCURRED_AT", "Incident:INC-1", "Equipment:P-101A", [{"doc_id": "INC-1"}])
    g.upsert_edge("EXHIBITS", "Incident:INC-1", "FailureMode:ELP", [{"doc_id": "INC-1"}])

    vec = VectorStore()
    for cid, text in [("c_seal", "P-101A mechanical seal failure external leakage"),
                      ("c_insp", "Inspection record for P-101A is OVERDUE"),
                      ("c_near", "Near-miss incident seal leak on P-101A"),
                      ("c_noise", "routine bearing noise check")]:
        vec.add(cid, text, emb.embed_one(text), {"doc_id": cid.upper(), "page": 1})
    return Repos(graph=g, vector=vec, emb=emb)


def test_router_heuristics():
    g = _mini_repos().graph
    assert router.classify_heuristic("What does OISD-STD-105 clause 7.3 require?", g) == "lookup"
    assert router.classify_heuristic(
        "Which pump had seal failures and an overdue inspection?", g) == "multihop"
    assert router.classify_heuristic(
        "What failure patterns recur across the last five years?", g) == "global"


def test_multihop_uses_ppr_and_returns_path_and_citations():
    repos = _mini_repos()
    ans = ask("Which pump had a seal failure and an overdue inspection near-miss?",
              repos, llm=None)
    assert ans.mode == "multihop"
    assert "Equipment:P-101A" in ans.seeds
    assert ans.citations and all(c.quote for c in ans.citations)
    # the relevant seal / overdue / near-miss chunks should dominate the top citations
    cited = [c.doc_id for c in ans.citations]
    relevant = {"C_SEAL", "C_INSP", "C_NEAR"}
    assert len(relevant & set(cited[:3])) >= 2


def test_every_answer_has_citations():
    repos = _mini_repos()
    ans = ask("Tell me about P-101A seal failure", repos, llm=None)
    assert ans.citations
    for c in ans.citations:
        assert c.doc_id and c.quote
        assert len(c.quote.split()) <= 16


def test_baseline_bypasses_graph():
    repos = _mini_repos()
    base = ask("Which pump had seal failure and overdue inspection?", repos,
               llm=None, baseline=True)
    hybrid = ask("Which pump had seal failure and overdue inspection?", repos,
                 llm=None)
    assert base.mode == "lookup" and not base.path
    assert hybrid.mode == "multihop" and hybrid.seeds        # graph engaged


def test_out_of_corpus_refuses():
    repos = _mini_repos()
    ans = ask("What is the recommended tyre pressure for a delivery truck?",
              repos, llm=None)
    assert ans.mode == "refusal"
    assert "couldn't find" in ans.answer_markdown.lower()


def test_global_uses_communities_not_refusal():
    repos = _mini_repos()
    ans = ask("What failure patterns recur across the last five years?", repos,
              llm=None)
    assert ans.mode == "global"
    assert "cluster" in ans.answer_markdown.lower()
