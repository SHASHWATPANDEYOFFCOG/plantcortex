"""Equipment dossier — the field technician's one-tap asset view.

Assembles everything the graph knows about a tag: recent work orders, governing SOPs,
known failure modes, incidents, regulatory clauses, tacit notes, connected equipment,
and inspection status — each item carrying its source doc for one-tap citation.
"""

from __future__ import annotations

from core.ontology import normalize_tag


def _prov_source(node: dict) -> dict:
    for p in node.get("provenance", []):
        if p.get("doc_id"):
            return {"doc_id": p["doc_id"], "page": p.get("page"), "row": p.get("row")}
    return {}


def build_dossier(graph, tag: str) -> dict:
    key = f"Equipment:{normalize_tag(tag)}"
    node = graph.get_node(key)
    if not node:
        return {"tag": normalize_tag(tag), "found": False}

    work_orders, procedures, incidents, clauses, notes, connected = [], [], [], [], [], []
    failure_modes: dict[str, int] = {}
    inspection = None

    for e in graph.edges_incident(key):
        src, tgt, et = e["source"], e["target"], e["type"]
        if et == "PERFORMED_ON" and tgt == key:
            n = graph.get_node(src)
            if n:
                p = n["props"]
                work_orders.append({"wo_id": p.get("wo_id"), "date": p.get("date"),
                                    "type": p.get("type"),
                                    "problem": (p.get("problem_text") or "")[:90],
                                    "source": {"doc_id": "DOC-WO-001",
                                               "row": _prov_source(n).get("row")}})
        elif et == "GOVERNS" and src.startswith("Procedure:"):
            n = graph.get_node(src)
            if n:
                procedures.append({"sop_id": n["props"].get("sop_id"),
                                   "title": n["props"].get("title"),
                                   "source": {"doc_id": n["props"].get("sop_id")}})
        elif et == "OCCURRED_AT" and src.startswith("Incident:"):
            n = graph.get_node(src)
            if n:
                incidents.append({"incident_id": n["props"].get("incident_id"),
                                  "severity": n["props"].get("severity"),
                                  "date": n["props"].get("date"),
                                  "source": {"doc_id": n["props"].get("incident_id")}})
        elif et == "COVERS" and src.startswith("RegulatoryClause:"):
            n = graph.get_node(src)
            if n:
                clauses.append({"standard": n["props"].get("standard"),
                                "clause_no": n["props"].get("clause_no"),
                                "source": {"doc_id": n["props"].get("standard")}})
        elif et == "ABOUT" and src.startswith("TacitNote:"):
            n = graph.get_node(src)
            if n:
                notes.append({"note_id": n["props"].get("note_id"),
                              "author_role": n["props"].get("author_role"),
                              "transcript": (n["props"].get("transcript") or "")[:160],
                              "source": {"doc_id": n["props"].get("note_id")}})
        elif et == "CONNECTED_TO":
            other = tgt if src == key else src
            connected.append(other.split(":", 1)[1])

    # failure modes exhibited by this asset's work orders + incidents
    for wo in work_orders:
        wk = f"WorkOrder:{wo['wo_id']}"
        for e in graph.edges_incident(wk):
            if e["type"] == "EXHIBITS" and e["target"].startswith("FailureMode:"):
                code = e["target"].split(":", 1)[1]
                failure_modes[code] = failure_modes.get(code, 0) + 1
    for inc in incidents:
        ik = f"Incident:{inc['incident_id']}"
        for e in graph.edges_incident(ik):
            if e["type"] == "EXHIBITS" and e["target"].startswith("FailureMode:"):
                code = e["target"].split(":", 1)[1]
                failure_modes[code] = failure_modes.get(code, 0) + 1

    # inspection status: chunk from an inspection doc that mentions this asset
    for e in graph.edges_incident(key):
        if e["type"] == "MENTIONS" and e["source"].startswith("Chunk:"):
            cn = graph.get_node(e["source"])
            if cn and str(cn["props"].get("doc_id", "")).startswith("INSP"):
                txt = cn["props"].get("text", "")
                inspection = {"doc_id": cn["props"]["doc_id"],
                              "overdue": "OVERDUE" in txt.upper(), "text": txt[:120]}

    work_orders.sort(key=lambda w: (w.get("date") or ""), reverse=True)
    return {
        "tag": normalize_tag(tag), "found": True,
        "name": node["props"].get("name"), "type": node["props"].get("type"),
        "location": node["props"].get("location"),
        "work_orders": work_orders[:5], "work_order_count": len(work_orders),
        "procedures": procedures, "incidents": incidents,
        "regulatory": clauses, "tacit_notes": notes,
        "connected_to": sorted(set(connected)),
        "failure_modes": [{"code": c, "count": n}
                          for c, n in sorted(failure_modes.items(), key=lambda x: -x[1])],
        "inspection": inspection,
    }
