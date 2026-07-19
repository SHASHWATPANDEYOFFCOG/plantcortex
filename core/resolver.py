"""Cross-modal entity resolution (Novelty N1).

The same pump detected as a P&ID symbol, a spreadsheet row, an SOP mention and a
compliance clause must resolve to ONE graph node. Strategy, in order:
  (a) exact canonical-key match (tag normalization already collapses spelling variants);
  (b) fuzzy tag match (edit distance <= 1, same prefix class) -> merge, lower confidence,
      record a SAME_AS provenance event;
  (c) [optional] name-embedding similarity for non-tagged entities (hook provided).

The resolver converts a raw ``ExtractionResult`` into JSON-safe node/edge records with
provenance, ready for idempotent MERGE upsert.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from core.ontology import (
    EDGE_DOMAIN, EdgeType, ExtractionResult, NodeType, normalize_tag,
)


# --------------------------------------------------------------------------- #
# Key construction
# --------------------------------------------------------------------------- #
def canonical_key(ntype: NodeType, props: dict) -> Optional[str]:
    """Canonical graph key from a node's properties (mirrors ontology .key())."""
    t = ntype.value if isinstance(ntype, NodeType) else str(ntype)
    p = props or {}
    if t == "Equipment":
        tag = p.get("tag")
        return f"Equipment:{normalize_tag(tag)}" if tag else None
    if t == "FailureMode":
        code = p.get("code")
        return f"FailureMode:{str(code).strip().upper()}" if code else None
    if t == "Component":
        name = p.get("name")
        if not name:
            return None
        parent = normalize_tag(p["parent_equipment"]) if p.get("parent_equipment") else "?"
        return f"Component:{parent}:{name.strip().lower()}"
    if t == "WorkOrder":
        return f"WorkOrder:{p['wo_id']}" if p.get("wo_id") else None
    if t == "Incident":
        return f"Incident:{p['incident_id']}" if p.get("incident_id") else None
    if t == "Procedure":
        return f"Procedure:{p['sop_id']}" if p.get("sop_id") else None
    if t == "Permit":
        return f"Permit:{p['permit_id']}" if p.get("permit_id") else None
    if t == "RegulatoryClause":
        if p.get("standard") and p.get("clause_no"):
            return f"RegulatoryClause:{p['standard']}:{p['clause_no']}"
        return None
    if t == "Person":
        return f"Person:{p['name'].strip().lower()}" if p.get("name") else None
    if t == "Document":
        return f"Document:{p['doc_id']}" if p.get("doc_id") else None
    if t == "Chunk":
        return f"Chunk:{p['chunk_id']}" if p.get("chunk_id") else None
    if t == "TacitNote":
        return f"TacitNote:{p['note_id']}" if p.get("note_id") else None
    return None


def key_for_ref(ntype: NodeType, ref: str) -> Optional[str]:
    """Canonical key from a bare natural-id reference used in an edge."""
    t = ntype.value if isinstance(ntype, NodeType) else str(ntype)
    ref = (ref or "").strip()
    if not ref:
        return None
    if t == "Equipment":
        return f"Equipment:{normalize_tag(ref)}"
    if t == "FailureMode":
        return f"FailureMode:{ref.upper()}"
    if t == "RegulatoryClause":
        r = ref.replace(" ", ":").replace("::", ":")
        return f"RegulatoryClause:{r}" if not r.startswith("RegulatoryClause") else r
    if t == "Component":
        if ":" in ref:
            parent, name = ref.split(":", 1)
            return f"Component:{normalize_tag(parent)}:{name.strip().lower()}"
        return f"Component:?:{ref.strip().lower()}"
    if t == "Person":
        return f"Person:{ref.lower()}"
    prefix = {"WorkOrder": "WorkOrder", "Incident": "Incident",
              "Procedure": "Procedure", "Permit": "Permit",
              "Document": "Document", "Chunk": "Chunk", "TacitNote": "TacitNote"}.get(t)
    return f"{prefix}:{ref}" if prefix else None


# --------------------------------------------------------------------------- #
# Fuzzy tag matching
# --------------------------------------------------------------------------- #
def _edit_distance_le1(a: str, b: str) -> bool:
    if a == b:
        return True
    la, lb = len(a), len(b)
    if abs(la - lb) > 1:
        return False
    if la == lb:  # single substitution
        return sum(x != y for x, y in zip(a, b)) == 1
    # single insertion/deletion
    if la > lb:
        a, b = b, a
    i = j = 0
    skipped = False
    while i < len(a) and j < len(b):
        if a[i] != b[j]:
            if skipped:
                return False
            skipped = True
            j += 1
        else:
            i += 1
            j += 1
    return True


def _tag_prefix(tag: str) -> str:
    for i, ch in enumerate(tag):
        if ch.isdigit():
            return tag[:i]
    return tag


def fuzzy_equipment_match(tag: str, existing_tags: list[str]) -> Optional[str]:
    """Return an existing tag within edit-distance 1 and same prefix class, else None."""
    pref = _tag_prefix(tag)
    for cand in existing_tags:
        if _tag_prefix(cand) == pref and _edit_distance_le1(tag, cand):
            return cand
    return None


# --------------------------------------------------------------------------- #
# Resolution
# --------------------------------------------------------------------------- #
@dataclass
class ResolveCtx:
    doc_id: str
    extractor: str = "llm_text"
    page: Optional[int] = None
    row: Optional[int] = None

    def prov(self, confidence: float, evidence: Optional[str]) -> dict:
        p = {"doc_id": self.doc_id, "extractor": self.extractor,
             "confidence": round(float(confidence), 3)}
        if self.page is not None:
            p["page"] = self.page
        if self.row is not None:
            p["row"] = self.row
        if evidence:
            p["evidence_span"] = evidence[:120]
        return p


@dataclass
class ResolvedDelta:
    nodes: list[dict] = field(default_factory=list)
    edges: list[dict] = field(default_factory=list)
    same_as: list[dict] = field(default_factory=list)


def _graph_equipment_tags(graph) -> list[str]:
    return [k.split(":", 1)[1] for k in graph.nodes_by_type("Equipment")]


def resolve_extraction(extraction: ExtractionResult, ctx: ResolveCtx,
                       graph) -> ResolvedDelta:
    """Turn an ExtractionResult into resolved, provenance-carrying node/edge records."""
    delta = ResolvedDelta()
    ref_index: dict[str, str] = {}          # natural-id / alias -> final key
    existing_tags = _graph_equipment_tags(graph)

    # --- nodes ---
    for en in extraction.nodes:
        props = dict(en.properties)
        if en.type == NodeType.EQUIPMENT and props.get("tag"):
            props["tag"] = normalize_tag(props["tag"])
        key = canonical_key(en.type, props)
        if not key:
            continue
        # exact then fuzzy resolution
        final_key = key
        if not graph.has_node(key) and en.type == NodeType.EQUIPMENT:
            match = fuzzy_equipment_match(props["tag"], existing_tags)
            if match:
                final_key = f"Equipment:{match}"
                delta.same_as.append({"from": key, "to": final_key,
                                      "reason": "fuzzy_tag<=1",
                                      "provenance": ctx.prov(en.confidence * 0.8,
                                                             en.evidence_span)})
        if final_key == key and en.type == NodeType.EQUIPMENT:
            existing_tags.append(props["tag"])

        delta.nodes.append({"key": final_key, "type": en.type.value, "props": props,
                            "provenance": [ctx.prov(en.confidence, en.evidence_span)]})
        # index aliases for edge linking
        _register_aliases(ref_index, en.type, props, final_key)

    # --- edges ---
    for ee in extraction.edges:
        pair = _pick_pair(ee.type, ee.source_ref, ee.target_ref, ref_index)
        if pair is None:
            continue
        src_type, tgt_type = pair
        src_key = ref_index.get(ee.source_ref) or _resolve_ref_key(
            src_type, ee.source_ref, graph)
        tgt_key = ref_index.get(ee.target_ref) or _resolve_ref_key(
            tgt_type, ee.target_ref, graph)
        if not src_key or not tgt_key:
            continue
        delta.edges.append({"type": ee.type.value, "source": src_key,
                            "target": tgt_key,
                            "provenance": [ctx.prov(ee.confidence, ee.evidence_span)]})
    return delta


def _register_aliases(index: dict, ntype: NodeType, props: dict, key: str) -> None:
    aliases = []
    if ntype == NodeType.EQUIPMENT and props.get("tag"):
        aliases += [props["tag"], props["tag"].replace("-", "")]
    elif ntype == NodeType.FAILURE_MODE and props.get("code"):
        aliases.append(str(props["code"]).upper())
    elif ntype == NodeType.WORK_ORDER and props.get("wo_id"):
        aliases.append(props["wo_id"])
    elif ntype == NodeType.INCIDENT and props.get("incident_id"):
        aliases.append(props["incident_id"])
    elif ntype == NodeType.PROCEDURE and props.get("sop_id"):
        aliases.append(props["sop_id"])
    elif ntype == NodeType.REGULATORY_CLAUSE and props.get("clause_no"):
        aliases += [f"{props.get('standard')}:{props['clause_no']}",
                    f"{props.get('standard')} {props['clause_no']}"]
    for a in aliases:
        if a:
            index[a] = key


def _resolve_ref_key(ntype: NodeType, ref: str, graph) -> Optional[str]:
    key = key_for_ref(ntype, ref)
    if not key:
        return None
    if ntype == NodeType.EQUIPMENT and not graph.has_node(key):
        tag = key.split(":", 1)[1]
        match = fuzzy_equipment_match(tag, _graph_equipment_tags(graph))
        if match:
            return f"Equipment:{match}"
    return key


def _pick_pair(etype: EdgeType, src_ref: str, tgt_ref: str, index: dict):
    pairs = EDGE_DOMAIN.get(etype, [])
    if not pairs:
        return None
    if len(pairs) == 1:
        return pairs[0]
    # disambiguate multi-domain edges (e.g. EXHIBITS: WO|Incident -> FailureMode)
    for s, t in pairs:
        sk = index.get(src_ref, "")
        if sk.startswith(s.value + ":"):
            return (s, t)
    return pairs[0]
