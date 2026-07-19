"""M2 P&ID ingestion: drawing -> Equipment nodes + CONNECTED_TO topology.

Hybrid, per the research finding that raw VLMs are unreliable on diagram detail:
  * VLM read (primary) when the model/quota is available;
  * OpenCV structural read (offline fallback) otherwise.

Every P&ID-derived fact carries a **drawing bbox** in its provenance, so clicking a node
in the UI can highlight its region on the sheet. Equipment resolves by tag to the SAME
graph node the work orders/SOPs created (Novelty N1) — and the drawing *enriches* those
nodes with equipment `type` and a visual location.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Optional

from core.ontology import normalize_tag
from pipelines.m2_pnid import geometric, vision

log = logging.getLogger("plantcortex.pnid")

# P&ID symbol class -> ontology Equipment.type
_CLS_TO_TYPE = {"pump": "pump", "compressor": "compressor", "vessel": "vessel",
                "tank": "tank", "heat_exchanger": "heat exchanger",
                "valve": "control valve", "instrument": "instrument"}


def read_pnid(image_path: Path, layer_path: Optional[Path], llm) -> dict:
    """Prefer the VLM; fall back to the deterministic geometric reader."""
    vlm = vision.read_pnid_vlm(llm, image_path)
    if vlm and vlm.get("symbols"):
        # if the VLM gives no connections, borrow the geometric ones
        if not vlm.get("connections") and layer_path and layer_path.exists():
            vlm["connections"] = geometric.geometric_read(image_path, layer_path)["connections"]
        return vlm
    if layer_path and layer_path.exists():
        return geometric.geometric_read(image_path, layer_path)
    return {"symbols": [], "tags": [], "connections": [], "source": "none"}


def ingest_pnid(repos, entry: dict, seed_dir: Path, llm=None,
                emit: Optional[Callable[[dict], None]] = None):
    """Ingest one P&ID. Returns a DocResult-like object with counts."""
    from pipelines.m1_ingest.pipeline import DocResult, _upsert_delta

    res = DocResult(doc_id=entry["doc_id"], llm_used=llm is not None)
    image_path = seed_dir / entry["filename"]
    layer_path = seed_dir / entry["layer_file"] if entry.get("layer_file") else None

    read = read_pnid(image_path, layer_path, llm)
    doc_key = f"Document:{entry['doc_id']}"
    extractor = "llm_vision" if read.get("source") == "vlm" else "geometry"

    nodes, edges = [], []
    tag_seen = {}
    for s in read["symbols"]:
        tag = s.get("tag")
        if not tag:
            continue
        ntag = normalize_tag(tag)
        key = f"Equipment:{ntag}"
        bbox = s.get("bbox")
        prov = {"doc_id": entry["doc_id"], "extractor": extractor, "confidence": 0.85}
        if bbox:
            prov["bbox"] = [round(float(v), 1) for v in bbox]
        props = {"tag": ntag}
        etype = _CLS_TO_TYPE.get(s.get("cls"))
        if etype:
            props["type"] = etype       # P&ID enriches the tag with its equipment type
        nodes.append({"key": key, "type": "Equipment", "props": props,
                      "provenance": [prov]})
        edges.append({"type": "EXTRACTED_FROM", "source": key, "target": doc_key,
                      "provenance": [prov]})
        tag_seen[ntag] = key

    for a, b in read["connections"]:
        ka, kb = f"Equipment:{normalize_tag(a)}", f"Equipment:{normalize_tag(b)}"
        prov = {"doc_id": entry["doc_id"], "extractor": extractor, "confidence": 0.8}
        edges.append({"type": "CONNECTED_TO", "source": ka, "target": kb,
                      "provenance": [prov]})

    an, ae = _upsert_delta(repos.graph, nodes, edges)
    res.nodes_added, res.edges_added = an, ae
    res.chunks = 0
    log.info("pnid %s: %d symbols, %d connections (%s)", entry["doc_id"],
             len(read["symbols"]), len(read["connections"]), read.get("source"))
    if emit:
        emit({"type": "graph.delta", "doc_id": res.doc_id, "doc_type": "pnid",
              "nodes_added": an, "edges_added": ae, "chunks": 0,
              "source": read.get("source"),
              "graph": {"nodes": repos.graph.g.number_of_nodes(),
                        "edges": repos.graph.g.number_of_edges()}})
    return res
