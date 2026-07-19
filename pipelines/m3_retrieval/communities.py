"""GraphRAG-style community detection + summaries for corpus-wide questions.

Louvain communities over the entity subgraph (Chunk/Document excluded so communities
track real assets, not documents). Each community gets a summary — LLM when available,
otherwise a deterministic extractive one. Cached to disk; the demo replays offline.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import networkx as nx

from core.config import settings

log = logging.getLogger("plantcortex.communities")

_CACHE = settings.data_dir / "communities.json"
_ENTITY_TYPES = {"Equipment", "FailureMode", "WorkOrder", "Incident", "Procedure",
                 "RegulatoryClause", "TacitNote", "Component"}


def _label(graph, key: str) -> str:
    n = graph.get_node(key)
    if not n:
        return key
    p = n["props"]
    return (p.get("tag") or p.get("code") or p.get("sop_id") or p.get("incident_id")
            or p.get("wo_id") or p.get("clause_no") or key.split(":", 1)[-1])


def build_communities(graph, llm=None) -> list[dict]:
    ent_nodes = [k for k, d in graph.g.nodes(data=True)
                 if d.get("type") in _ENTITY_TYPES]
    sub = graph.g.subgraph(ent_nodes).to_undirected()
    if sub.number_of_nodes() == 0:
        return []
    try:
        parts = nx.community.louvain_communities(sub, seed=42)
    except Exception:
        parts = list(nx.connected_components(sub))

    communities = []
    for i, part in enumerate(sorted(parts, key=len, reverse=True)):
        members = list(part)
        by_type: dict[str, list[str]] = {}
        for k in members:
            t = graph.g.nodes[k]["type"]
            by_type.setdefault(t, []).append(_label(graph, k))
        labels = [lab for labs in by_type.values() for lab in labs][:20]
        summary = _summarize(by_type, llm)
        communities.append({"id": i, "size": len(members),
                            "labels": labels, "by_type": by_type,
                            "summary": summary})
    return communities


def _summarize(by_type: dict[str, list[str]], llm=None) -> str:
    extractive = "; ".join(
        f"{t}: {', '.join(sorted(set(v))[:6])}" for t, v in sorted(by_type.items()))
    if llm is None or getattr(llm, "quota_blocked", False):
        return extractive
    prompt = ("Summarize this cluster of related plant assets in 1-2 sentences for an "
              "engineer, focusing on what ties them together and any risk. Data:\n"
              + extractive + "\nReturn JSON {\"summary\": \"...\"}")
    res = llm.complete_json("", prompt, max_tokens=200)
    return (res or {}).get("summary") or extractive


def get_communities(repos, llm=None, rebuild: bool = False) -> list[dict]:
    # Tiny graphs (tests, ad-hoc) build fresh and never touch the shared disk cache.
    small = repos.graph.g.number_of_nodes() < 50
    if _CACHE.exists() and not rebuild and not small:
        try:
            return json.loads(_CACHE.read_text(encoding="utf-8"))
        except Exception:
            pass
    comms = build_communities(repos.graph, llm)
    if not small:
        _CACHE.parent.mkdir(parents=True, exist_ok=True)
        _CACHE.write_text(json.dumps(comms, indent=2), encoding="utf-8")
        log.info("built %d communities", len(comms))
    return comms
