"""Retrieval strategies.

* lookup   — hybrid (BM25 + dense) chunk search, RRF-fused.
* multihop — HippoRAG-style: link query entities -> Personalized PageRank over the graph
             -> gather chunks MENTIONS-linked to top nodes, fused with hybrid hits.
* global   — GraphRAG-style: map-reduce over precomputed community summaries.

The multihop path is the one that makes hybrid beat a vector-only baseline on questions
that require joining facts across documents.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import networkx as nx

from pipelines.m1_ingest.extract import extract_tags, failure_codes

_ENTITY_PREFIXES = ("Equipment:", "FailureMode:", "WorkOrder:", "Incident:",
                    "Procedure:", "RegulatoryClause:", "TacitNote:", "Component:")

_STOP = set("the a an of to in on for and or is are was were be with at by from as "
           "what which who how why when where did do does has have had this that any "
           "all it its plant unit had a report".split())


def _qtokens(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9\-]+", text.lower())
            if t not in _STOP and len(t) > 2}


@dataclass
class RetrievalResult:
    mode: str
    chunks: list[dict] = field(default_factory=list)          # {chunk_id,text,meta,score}
    seeds: list[str] = field(default_factory=list)
    ppr: list[tuple[str, float]] = field(default_factory=list)
    path: list[tuple[str, str, str]] = field(default_factory=list)
    communities: list[dict] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# entity linking
# --------------------------------------------------------------------------- #
def link_entities(question: str, graph) -> list[str]:
    seeds: list[str] = []

    def add(k: str) -> None:
        if graph.has_node(k) and k not in seeds:
            seeds.append(k)

    for t in extract_tags(question):
        add(f"Equipment:{t}")
    for c in failure_codes(question):
        add(f"FailureMode:{c}")
    for m in re.findall(r"\bSOP-\d+\b", question, re.I):
        add(f"Procedure:{m.upper()}")
    for m in re.findall(r"\bINC-\d{4}-\d+\b", question, re.I):
        add(f"Incident:{m.upper()}")
    for m in re.findall(r"\bWO-\d+\b", question, re.I):
        add(f"WorkOrder:{m.upper()}")
    return seeds


def _chunk_from_graph(graph, chunk_key: str) -> dict | None:
    n = graph.get_node(chunk_key)
    if not n:
        return None
    p = n["props"]
    return {"chunk_id": p.get("chunk_id"), "text": p.get("text", ""),
            "meta": {"doc_id": p.get("doc_id"), "page": p.get("page")}}


# --------------------------------------------------------------------------- #
# strategies
# --------------------------------------------------------------------------- #
def lookup(question: str, repos, k: int = 6) -> RetrievalResult:
    hits = repos.vector.hybrid(question, k=k)
    return RetrievalResult(mode="lookup", chunks=hits)


def multihop(question: str, repos, k: int = 6, graph_only: bool = False
             ) -> RetrievalResult:
    graph = repos.graph
    seeds = link_entities(question, graph)
    # graph_only bypasses the vector index entirely (the eval comparison condition)
    hybrid_hits = [] if graph_only else repos.vector.hybrid(question, k=k)

    if not seeds and not graph_only:
        # seed PPR from entities mentioned by the top hybrid chunks
        for h in hybrid_hits[:4]:
            ck = f"Chunk:{h['chunk_id']}"
            for etype, nb, direction in graph.neighbors(ck):
                if etype == "MENTIONS" and nb.startswith(_ENTITY_PREFIXES):
                    if nb not in seeds:
                        seeds.append(nb)

    ppr = graph.personalized_pagerank(seeds, top_n=30) if seeds else []
    entity_ppr = [(k_, s) for k_, s in ppr if k_.startswith(_ENTITY_PREFIXES)]
    top_entities = {k_ for k_, _ in entity_ppr[:15]}
    qtok = _qtokens(question)

    chunks: dict[str, dict] = {}

    # (a) hybrid (lexical/dense) hits, boosted when graph-grounded in a top PPR entity
    for h in hybrid_hits:
        ck = f"Chunk:{h['chunk_id']}"
        mentioned = {nb for et, nb, _ in graph.neighbors(ck) if et == "MENTIONS"}
        boost = 2.0 if (mentioned & top_entities) else 1.0
        chunks[h["chunk_id"]] = {**h, "score": h["score"] * boost}

    # (b) chunks MENTIONS-linked to top PPR entities, but ONLY those lexically relevant
    #     to the question — this keeps the join answer (seal/overdue/near-miss) from
    #     being buried under the asset's many routine work orders.
    for key, score in entity_ppr[:12]:
        for etype, nb, direction in graph.neighbors(key):
            if etype != "MENTIONS" or direction != "in" or not nb.startswith("Chunk:"):
                continue
            ci = _chunk_from_graph(graph, nb)
            if not ci or not ci["chunk_id"]:
                continue
            lex = len(_qtokens(ci["text"]) & qtok)
            if lex == 0:
                continue
            add = score * (1 + lex)
            cid = ci["chunk_id"]
            if cid in chunks:
                chunks[cid]["score"] += add
            else:
                ci["score"] = add
                chunks[cid] = ci
    ranked = sorted(chunks.values(), key=lambda c: -c.get("score", 0.0))[:k]

    targets = [k_ for k_, _ in entity_ppr if k_ not in seeds][:3]
    path = _build_path(graph, seeds, targets)
    return RetrievalResult(mode="multihop", chunks=ranked, seeds=seeds,
                           ppr=entity_ppr[:12], path=path)


def global_query(question: str, repos, llm=None, k: int = 4) -> RetrievalResult:
    from pipelines.m3_retrieval.communities import get_communities

    comms = get_communities(repos, llm)
    scored = sorted(comms, key=lambda c: -_relevance(question, c))[:k]
    # representative chunks for citations: top hybrid hits
    hits = repos.vector.hybrid(question, k=6)
    return RetrievalResult(mode="global", chunks=hits, communities=scored)


def _relevance(question: str, community: dict) -> float:
    qtok = set(re.findall(r"[a-z0-9\-]+", question.lower()))
    ctok = set(re.findall(r"[a-z0-9\-]+", (community.get("summary", "") + " " +
                                           " ".join(community.get("labels", []))).lower()))
    if not ctok:
        return 0.0
    return len(qtok & ctok) / (len(qtok) + 1)


# --------------------------------------------------------------------------- #
# reasoning path (for the graph panel)
# --------------------------------------------------------------------------- #
def _edge_type(graph, a: str, b: str) -> str:
    for u, v in ((a, b), (b, a)):
        if graph.g.has_edge(u, v):
            d = graph.g.get_edge_data(u, v)
            return next(iter(d)) if d else "REL"
    return "REL"


def _build_path(graph, seeds: list[str], targets: list[str],
                max_edges: int = 6) -> list[tuple[str, str, str]]:
    if not seeds or not targets:
        return []
    ug = graph.g.to_undirected()
    for s in seeds:
        for t in targets:
            if s == t or not ug.has_node(s) or not ug.has_node(t):
                continue
            try:
                nodes = nx.shortest_path(ug, s, t)
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                continue
            if 2 <= len(nodes) <= max_edges + 1:
                return [(a, _edge_type(graph, a, b), b)
                        for a, b in zip(nodes, nodes[1:])]
    return []
