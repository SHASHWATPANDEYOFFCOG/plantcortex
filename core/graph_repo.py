"""GraphRepo — the knowledge graph behind a small interface.

Default implementation is NetworkX persisted to disk (no Docker needed). The same
interface can be backed by Neo4j later without touching callers.

Node record shape (JSON-safe):  {key, type, props: dict, provenance: [dict], doc_ids}
Edges are keyed by type between two node keys, so upsert is idempotent MERGE:
re-ingesting a document unions provenance instead of duplicating facts.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Optional

import networkx as nx

from core.config import settings


def _doc_ids_from_prov(provenance: list[dict]) -> set[str]:
    return {p["doc_id"] for p in provenance if p.get("doc_id")}


class GraphRepo:
    """NetworkX-backed knowledge graph with MERGE upserts + PPR."""

    def __init__(self, path: Optional[Path] = None) -> None:
        self.path = path or settings.graph_path
        self.g = nx.MultiDiGraph()

    # -- upserts ----------------------------------------------------------- #
    def upsert_node(self, key: str, ntype: str, props: dict,
                    provenance: Optional[list[dict]] = None) -> bool:
        """MERGE a node. Returns True if newly created."""
        provenance = provenance or []
        new = not self.g.has_node(key)
        if new:
            self.g.add_node(key, type=ntype, props=dict(props),
                            provenance=list(provenance),
                            doc_ids=sorted(_doc_ids_from_prov(provenance)))
            return True
        data = self.g.nodes[key]
        # fill missing / non-null props without clobbering existing values
        for pk, pv in props.items():
            if pv not in (None, "") and data["props"].get(pk) in (None, ""):
                data["props"][pk] = pv
        data["provenance"].extend(provenance)
        data["doc_ids"] = sorted(set(data["doc_ids"]) | _doc_ids_from_prov(provenance))
        return False

    def upsert_edge(self, etype: str, source: str, target: str,
                    provenance: Optional[list[dict]] = None) -> bool:
        """MERGE an edge keyed by (source, type, target). True if newly created."""
        provenance = provenance or []
        if self.g.has_edge(source, target, key=etype):
            data = self.g.edges[source, target, etype]
            data["provenance"].extend(provenance)
            return False
        self.g.add_edge(source, target, key=etype, type=etype,
                        provenance=list(provenance))
        return True

    # -- reads ------------------------------------------------------------- #
    def has_node(self, key: str) -> bool:
        return self.g.has_node(key)

    def get_node(self, key: str) -> Optional[dict]:
        if not self.g.has_node(key):
            return None
        d = self.g.nodes[key]
        return {"key": key, "type": d["type"], "props": d["props"],
                "provenance": d["provenance"], "doc_ids": d["doc_ids"]}

    def nodes_by_type(self, ntype: str) -> list[str]:
        return [k for k, d in self.g.nodes(data=True) if d.get("type") == ntype]

    def neighbors(self, key: str) -> list[tuple[str, str, str]]:
        """(edge_type, neighbor_key, direction) for both in and out edges."""
        out = [(d["type"], v, "out") for _, v, d in self.g.out_edges(key, data=True)]
        inn = [(d["type"], u, "in") for u, _, d in self.g.in_edges(key, data=True)]
        return out + inn

    def edges_incident(self, key: str) -> list[dict]:
        res = []
        for u, v, d in self.g.out_edges(key, data=True):
            res.append({"type": d["type"], "source": u, "target": v,
                        "provenance": d["provenance"]})
        for u, v, d in self.g.in_edges(key, data=True):
            res.append({"type": d["type"], "source": u, "target": v,
                        "provenance": d["provenance"]})
        return res

    def all_nodes(self) -> Iterable[dict]:
        for k, d in self.g.nodes(data=True):
            yield {"key": k, "type": d["type"], "props": d["props"],
                   "provenance": d["provenance"], "doc_ids": d["doc_ids"]}

    def all_edges(self) -> Iterable[dict]:
        for u, v, d in self.g.edges(data=True):
            yield {"type": d["type"], "source": u, "target": v,
                   "provenance": d["provenance"]}

    # -- doc-type map (for linkage metric & citations) --------------------- #
    def _doc_type_map(self) -> dict[str, str]:
        m: dict[str, str] = {}
        for k, d in self.g.nodes(data=True):
            if d.get("type") == "Document":
                did = d["props"].get("doc_id")
                if did:
                    m[did] = d["props"].get("doc_type", "unknown")
        return m

    def linkage_completeness(self) -> dict:
        """% of Equipment connected to >= 3 distinct document types (the M1 metric)."""
        dtmap = self._doc_type_map()
        equip = self.nodes_by_type("Equipment")
        if not equip:
            return {"equipment": 0, "well_linked": 0, "pct": 0.0, "detail": {}}
        detail: dict[str, int] = {}
        well = 0
        for k in equip:
            doc_ids = set(self.g.nodes[k]["doc_ids"])
            for e in self.edges_incident(k):
                doc_ids |= _doc_ids_from_prov(e["provenance"])
            dtypes = {dtmap.get(d, "unknown") for d in doc_ids}
            dtypes.discard("unknown")
            detail[k] = len(dtypes)
            if len(dtypes) >= 3:
                well += 1
        return {"equipment": len(equip), "well_linked": well,
                "pct": round(100.0 * well / len(equip), 1), "detail": detail}

    def summary(self) -> dict:
        node_counts: dict[str, int] = {}
        for _, d in self.g.nodes(data=True):
            node_counts[d["type"]] = node_counts.get(d["type"], 0) + 1
        edge_counts: dict[str, int] = {}
        for _, _, d in self.g.edges(data=True):
            edge_counts[d["type"]] = edge_counts.get(d["type"], 0) + 1
        return {"nodes": self.g.number_of_nodes(),
                "edges": self.g.number_of_edges(),
                "node_types": dict(sorted(node_counts.items())),
                "edge_types": dict(sorted(edge_counts.items())),
                "linkage": self.linkage_completeness()}

    # -- retrieval core: Personalized PageRank (HippoRAG-style) ------------ #
    def personalized_pagerank(self, seed_keys: list[str], alpha: float = 0.85,
                              top_n: int = 20) -> list[tuple[str, float]]:
        seeds = [k for k in seed_keys if self.g.has_node(k)]
        if not seeds or self.g.number_of_nodes() == 0:
            return []
        pers = {k: (1.0 / len(seeds) if k in seeds else 0.0) for k in self.g.nodes()}
        try:
            scores = nx.pagerank(self.g, alpha=alpha, personalization=pers)
        except Exception:  # convergence edge cases on tiny/degenerate graphs
            scores = {k: pers[k] for k in self.g.nodes()}
        ranked = sorted(scores.items(), key=lambda x: -x[1])
        return [(k, s) for k, s in ranked[:top_n]]

    # -- persistence ------------------------------------------------------- #
    def save(self, path: Optional[Path] = None) -> None:
        p = path or self.path
        p.parent.mkdir(parents=True, exist_ok=True)
        data = nx.node_link_data(self.g, edges="links")
        p.write_text(json.dumps(data), encoding="utf-8")

    def load(self, path: Optional[Path] = None) -> "GraphRepo":
        p = path or self.path
        if p.exists():
            data = json.loads(p.read_text(encoding="utf-8"))
            self.g = nx.node_link_graph(data, multigraph=True, directed=True,
                                        edges="links")
        return self

    def clear(self) -> None:
        self.g = nx.MultiDiGraph()
