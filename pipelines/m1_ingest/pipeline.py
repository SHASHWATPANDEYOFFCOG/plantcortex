"""M1 ingestion orchestrator: documents -> multimodal knowledge graph.

Per document:
  route -> parse -> (TLP normalize for work orders) -> chunk -> embed+index
        -> extract (structured for tables, ontology-constrained LLM for prose)
        -> cross-modal resolve (N1) -> idempotent MERGE upsert -> emit graph.delta

Structured work orders use deterministic column extraction (fast, exact, offline);
prose (SOPs / incidents / regulatory / voice) uses the LLM extractor. Scanned
inspections & P&IDs get Document nodes now; their OCR/vision extraction is M2 (D3),
though inspection equipment+overdue is threaded in from manifest metadata so the
multi-hop join works immediately.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

from core.config import settings
from core.embeddings import Embeddings, get_embeddings
from core.graph_repo import GraphRepo
from core.ontology import FAILURE_MODE_NAMES
from core.resolver import ResolveCtx, resolve_extraction
from core.vector_repo import VectorStore
from pipelines.m1_ingest import chunk as chunker
from pipelines.m1_ingest import extract as extractor
from pipelines.m1_ingest import parse
from pipelines.m1_ingest import tlp

log = logging.getLogger("plantcortex.ingest")

Emit = Callable[[dict], None]


@dataclass
class Repos:
    graph: GraphRepo
    vector: VectorStore
    emb: Embeddings

    def save(self) -> None:
        self.graph.save()
        self.vector.save()


def make_repos(fresh: bool = False) -> Repos:
    settings.ensure_dirs()
    graph = GraphRepo()
    vector = VectorStore()
    if not fresh:
        graph.load()
        vector = VectorStore.load()
    return Repos(graph=graph, vector=vector, emb=get_embeddings())


@dataclass
class DocResult:
    doc_id: str
    nodes_added: int = 0
    edges_added: int = 0
    chunks: int = 0
    llm_used: bool = False


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
_TYPE_FROM_PREFIX = {
    "Equipment": "Equipment", "Component": "Component", "FailureMode": "FailureMode",
    "WorkOrder": "WorkOrder", "Incident": "Incident", "Procedure": "Procedure",
    "Permit": "Permit", "RegulatoryClause": "RegulatoryClause", "Person": "Person",
    "Document": "Document", "Chunk": "Chunk", "TacitNote": "TacitNote",
}


def _type_of(key: str) -> str:
    return _TYPE_FROM_PREFIX.get(key.split(":", 1)[0], "Equipment")


def _upsert_delta(graph: GraphRepo, nodes: list[dict], edges: list[dict]) -> tuple[int, int]:
    added_n = added_e = 0
    for n in nodes:
        if graph.upsert_node(n["key"], n["type"], n.get("props", {}),
                             n.get("provenance")):
            added_n += 1
    for e in edges:
        # ensure endpoints exist (stub if an edge references an un-emitted node)
        for endpoint in (e["source"], e["target"]):
            if not graph.has_node(endpoint):
                graph.upsert_node(endpoint, _type_of(endpoint), {}, [])
        if graph.upsert_edge(e["type"], e["source"], e["target"], e.get("provenance")):
            added_e += 1
    return added_n, added_e


def _index_chunk(repos: Repos, chunk: dict) -> None:
    vec = repos.emb.embed_one(chunk["text"])
    repos.vector.add(chunk["chunk_id"], chunk["text"], vec,
                     {"doc_id": chunk["doc_id"], "page": chunk.get("page"),
                      "row": chunk.get("row")})


def _chunk_node(chunk: dict) -> dict:
    return {"key": f"Chunk:{chunk['chunk_id']}", "type": "Chunk",
            "props": {"chunk_id": chunk["chunk_id"], "doc_id": chunk["doc_id"],
                      "page": chunk.get("page"), "text": chunk["text"][:500]},
            "provenance": [{"doc_id": chunk["doc_id"], "extractor": "chunker"}]}


# --------------------------------------------------------------------------- #
# per-doc-type ingestion
# --------------------------------------------------------------------------- #
def _ingest_document_node(graph: GraphRepo, entry: dict) -> None:
    graph.upsert_node(
        f"Document:{entry['doc_id']}", "Document",
        {"doc_id": entry["doc_id"], "filename": entry.get("filename"),
         "doc_type": entry.get("doc_type"), "source_kind": entry.get("source_kind"),
         "page_count": entry.get("page_count"),
         "ingest_time": datetime.now().isoformat()},
        [{"doc_id": entry["doc_id"], "extractor": "seed"}])


def _ingest_work_orders(repos: Repos, entry: dict, seed_dir: Path) -> DocResult:
    res = DocResult(doc_id=entry["doc_id"])
    rows = parse.read_workorder_rows(seed_dir / entry["filename"])
    doc_key = f"Document:{entry['doc_id']}"
    for r in rows:
        tag = r.get("Equipment_Tag")
        if not tag:
            continue
        raw = f"{r.get('Problem_Text', '')}. {r.get('Action_Text', '')}"
        norm, _changes = tlp.normalize(raw)
        wo_id = str(r.get("WO_ID"))
        wtype = str(r.get("Type") or "").lower() or None
        row = r["row"]
        ctx = ResolveCtx(doc_id=entry["doc_id"], extractor="tlp", row=row)
        prov = ctx.prov(0.95, raw[:80])

        equip_key = f"Equipment:{_norm_tag(tag)}"
        wo_key = f"WorkOrder:{wo_id}"
        nodes = [
            {"key": equip_key, "type": "Equipment",
             "props": {"tag": _norm_tag(tag)}, "provenance": [prov]},
            {"key": wo_key, "type": "WorkOrder",
             "props": {"wo_id": wo_id, "type": wtype,
                       "date": _dstr(r.get("Date")),
                       "problem_text": norm,
                       "technician": r.get("Technician")},
             "provenance": [prov]},
        ]
        edges = [{"type": "PERFORMED_ON", "source": wo_key, "target": equip_key,
                  "provenance": [prov]}]
        # personnel: the technician becomes a Person node (entity, not just a string)
        tech = r.get("Technician")
        if tech:
            person_key = f"Person:{str(tech).strip().lower()}"
            nodes.append({"key": person_key, "type": "Person",
                          "props": {"name": str(tech).strip(), "role": "Technician"},
                          "provenance": [prov]})
        # failure modes from normalized text
        codes = extractor.failure_codes(norm)
        chunk = {"chunk_id": f"{entry['doc_id']}::r{row}", "doc_id": entry["doc_id"],
                 "page": 1, "row": row, "text": f"WO {wo_id} on {_norm_tag(tag)}: {norm}"}
        for code in codes:
            fm_key = f"FailureMode:{code}"
            nodes.append({"key": fm_key, "type": "FailureMode",
                          "props": {"code": code,
                                    "name": FAILURE_MODE_NAMES.get(code)},
                          "provenance": [prov]})
            edges.append({"type": "EXHIBITS", "source": wo_key, "target": fm_key,
                          "provenance": [prov]})
            edges.append({"type": "MENTIONS", "source": f"Chunk:{chunk['chunk_id']}",
                          "target": fm_key, "provenance": [prov]})
        # chunk node + index + provenance links
        nodes.append(_chunk_node(chunk))
        edges.append({"type": "MENTIONS", "source": f"Chunk:{chunk['chunk_id']}",
                      "target": equip_key, "provenance": [prov]})
        if tech:
            edges.append({"type": "MENTIONS", "source": f"Chunk:{chunk['chunk_id']}",
                          "target": f"Person:{str(tech).strip().lower()}",
                          "provenance": [prov]})
        edges.append({"type": "EXTRACTED_FROM", "source": wo_key, "target": doc_key,
                      "provenance": [prov]})
        _index_chunk(repos, chunk)
        an, ae = _upsert_delta(repos.graph, nodes, edges)
        res.nodes_added += an
        res.edges_added += ae
        res.chunks += 1
    return res


def _ingest_prose(repos: Repos, entry: dict, seed_dir: Path, llm) -> DocResult:
    res = DocResult(doc_id=entry["doc_id"], llm_used=llm is not None)
    pages = parse.read_pdf_pages(seed_dir / entry["filename"])
    chunks = chunker.chunk_pages(entry["doc_id"], pages)
    doc_key = f"Document:{entry['doc_id']}"
    # 1) index every chunk (retrieval + citation anchors)
    for ch in chunks:
        _index_chunk(repos, ch)
        repos.graph.upsert_node(**_as_upsert(_chunk_node(ch)))
        res.chunks += 1
    if not chunks:
        return res
    # 2) extract once over the whole document (full context; cross-chunk edges)
    full_text = "\n".join(p for p in pages if p.strip())
    extraction = extractor.llm_extract(full_text, entry["doc_id"],
                                       entry["doc_type"], llm)
    ctx = ResolveCtx(doc_id=entry["doc_id"], extractor="llm_text", page=1)
    delta = resolve_extraction(extraction, ctx, repos.graph)
    # 3) anchor each extracted node to the best-matching chunk + Document
    extra_edges = []
    for n in delta.nodes:
        ch = _best_chunk(n, chunks)
        extra_edges.append({"type": "MENTIONS", "source": f"Chunk:{ch['chunk_id']}",
                            "target": n["key"], "provenance": n["provenance"]})
        extra_edges.append({"type": "EXTRACTED_FROM", "source": n["key"],
                            "target": doc_key, "provenance": n["provenance"]})
    an, ae = _upsert_delta(repos.graph, delta.nodes, delta.edges + extra_edges)
    res.nodes_added += an
    res.edges_added += ae
    return res


def _best_chunk(node: dict, chunks: list[dict]) -> dict:
    """Pick the chunk whose text contains the node's identifying token (for citation)."""
    tok = (node["props"].get("tag") or node["props"].get("clause_no")
           or node["props"].get("sop_id") or node["props"].get("incident_id")
           or node["props"].get("code") or "")
    if tok:
        for ch in chunks:
            if tok.lower() in ch["text"].lower():
                return ch
    return chunks[0]


def _ingest_inspection(repos: Repos, entry: dict, seed_dir: Path) -> DocResult:
    """Scanned inspection: real OCR is D3. Thread equipment+overdue from manifest now."""
    res = DocResult(doc_id=entry["doc_id"])
    tag = entry.get("equipment")
    if not tag:
        return res
    equip_key = f"Equipment:{_norm_tag(tag)}"
    status = "OVERDUE" if entry.get("overdue") else "within inspection period"
    text = (f"Equipment inspection record {entry['doc_id']} for {_norm_tag(tag)}. "
            f"Inspection status: {status}.")
    ch = {"chunk_id": f"{entry['doc_id']}::insp", "doc_id": entry["doc_id"],
          "page": 1, "text": text}
    prov = {"doc_id": entry["doc_id"], "extractor": "manifest", "confidence": 0.9}
    _index_chunk(repos, ch)
    nodes = [{"key": equip_key, "type": "Equipment",
              "props": {"tag": _norm_tag(tag)}, "provenance": [prov]},
             _chunk_node(ch)]
    edges = [{"type": "MENTIONS", "source": f"Chunk:{ch['chunk_id']}",
              "target": equip_key, "provenance": [prov]}]
    an, ae = _upsert_delta(repos.graph, nodes, edges)
    res.nodes_added, res.edges_added, res.chunks = an, ae, 1
    return res


def _capture_transcript(repos: Repos, note_id: str, transcript: str,
                        role: str) -> tuple[int, int, list[str], list[str]]:
    """Shared core for voice-note ingest and live Knowledge Capture (M7).

    Creates a TacitNote node, indexes the transcript, and links ABOUT edges to every
    equipment tag and failure mode mentioned — making tacit knowledge a first-class,
    citable source in the graph.
    """
    note_key = f"TacitNote:{note_id}"
    prov = {"doc_id": note_id, "extractor": "asr", "confidence": 0.9}
    ch = {"chunk_id": f"{note_id}::t", "doc_id": note_id, "page": 1, "text": transcript}
    _index_chunk(repos, ch)
    nodes = [{"key": note_key, "type": "TacitNote",
              "props": {"note_id": note_id, "author_role": role,
                        "transcript": transcript[:500]}, "provenance": [prov]},
             _chunk_node(ch)]
    edges = []
    tags = extractor.extract_tags(transcript)
    for t in tags:
        ek = f"Equipment:{t}"
        edges.append({"type": "ABOUT", "source": note_key, "target": ek,
                      "provenance": [prov]})
        edges.append({"type": "MENTIONS", "source": f"Chunk:{ch['chunk_id']}",
                      "target": ek, "provenance": [prov]})
    codes = extractor.failure_codes(transcript)
    for code in codes:
        fk = f"FailureMode:{code}"
        nodes.append({"key": fk, "type": "FailureMode",
                      "props": {"code": code, "name": FAILURE_MODE_NAMES.get(code)},
                      "provenance": [prov]})
        edges.append({"type": "ABOUT", "source": note_key, "target": fk,
                      "provenance": [prov]})
    an, ae = _upsert_delta(repos.graph, nodes, edges)
    return an, ae, tags, codes


def capture_note(repos: Repos, transcript: str, author_role: str | None = None,
                 note_id: str | None = None) -> dict:
    """Live Knowledge Capture: turn a spoken/typed note into a citable graph node."""
    note_id = note_id or f"VN-{datetime.now().strftime('%Y%m%d%H%M%S')}"
    role = author_role or ("Senior Operator" if "operator" in transcript.lower()
                           else "Field Staff")
    an, ae, tags, codes = _capture_transcript(repos, note_id, transcript, role)
    repos.save()
    return {"note_id": note_id, "author_role": role, "equipment": tags,
            "failure_modes": codes, "nodes_added": an, "edges_added": ae}


def _ingest_voice(repos: Repos, entry: dict, seed_dir: Path, llm) -> DocResult:
    res = DocResult(doc_id=entry["doc_id"])
    tfile = entry.get("transcript_file")
    if not tfile:
        return res
    transcript = (seed_dir / tfile).read_text(encoding="utf-8")
    role = "Senior Operator" if "operator" in transcript.lower() else "Field Staff"
    an, ae, _t, _c = _capture_transcript(repos, entry["doc_id"], transcript, role)
    res.nodes_added, res.edges_added, res.chunks = an, ae, 1
    return res


# --------------------------------------------------------------------------- #
# small utils
# --------------------------------------------------------------------------- #
def _norm_tag(tag) -> str:
    from core.ontology import normalize_tag
    return normalize_tag(str(tag))


def _dstr(v) -> Optional[str]:
    if v is None:
        return None
    return str(v)[:10]


def _as_upsert(node: dict) -> dict:
    return {"key": node["key"], "ntype": node["type"], "props": node.get("props", {}),
            "provenance": node.get("provenance")}


# --------------------------------------------------------------------------- #
# public entry points
# --------------------------------------------------------------------------- #
def ingest_document(repos: Repos, entry: dict, seed_dir: Path, llm=None,
                    emit: Optional[Emit] = None) -> DocResult:
    _ingest_document_node(repos.graph, entry)
    dt = entry.get("doc_type")
    if dt == "work_order":
        res = _ingest_work_orders(repos, entry, seed_dir)
    elif dt in ("sop", "incident", "regulatory"):
        res = _ingest_prose(repos, entry, seed_dir, llm)
    elif dt == "inspection":
        res = _ingest_inspection(repos, entry, seed_dir)
    elif dt == "voice_note":
        res = _ingest_voice(repos, entry, seed_dir, llm)
    elif dt == "pnid":
        from pipelines.m2_pnid.pipeline import ingest_pnid
        res = ingest_pnid(repos, entry, seed_dir, llm=llm)   # M2 hybrid vision
    else:
        res = DocResult(doc_id=entry["doc_id"])
    if emit:
        emit({"type": "graph.delta", "doc_id": res.doc_id,
              "doc_type": dt, "nodes_added": res.nodes_added,
              "edges_added": res.edges_added, "chunks": res.chunks,
              "graph": {"nodes": repos.graph.g.number_of_nodes(),
                        "edges": repos.graph.g.number_of_edges()}})
    return res


def ingest_corpus(repos: Repos, seed_dir: Path, llm=None,
                  emit: Optional[Emit] = None, only_types: Optional[set] = None,
                  limit: Optional[int] = None) -> list[DocResult]:
    manifest = json.loads((seed_dir / "manifest.json").read_text(encoding="utf-8"))
    docs = manifest["documents"]
    if only_types:
        docs = [d for d in docs if d.get("doc_type") in only_types]
    if limit:
        docs = docs[:limit]
    results = []
    for entry in docs:
        r = ingest_document(repos, entry, seed_dir, llm=llm, emit=emit)
        results.append(r)
        log.info("ingested %s (+%dn/+%de)", r.doc_id, r.nodes_added, r.edges_added)
    repos.save()
    return results
