"""PlantCortex API (M1 surface for D2).

Endpoints
  GET  /health
  GET  /graph/summary          -> counts by type + linkage-completeness metric
  GET  /graph/subgraph         -> small ego-graph for the UI (center + hops)
  POST /ingest                 -> upload a document, ingest it, broadcast graph.delta
  WS   /ws                     -> live graph.delta stream (headline demo moment)

Ingestion runs in a threadpool; each document's delta is broadcast to all websocket
clients so the UI can show the graph growing live.
"""

from __future__ import annotations

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from dataclasses import asdict

from agents import m5_compliance, m6_patterns
from agents.dossier import build_dossier
from core.config import ROOT, settings
from core.llm import get_llm
from pipelines.m1_ingest.pipeline import (
    Repos, capture_note, ingest_document, make_repos,
)
from pipelines.m3_retrieval.engine import ask as ask_engine

log = logging.getLogger("plantcortex.api")

UPLOAD_DIR = settings.data_dir / "uploads"


def infer_doc_type(filename: str) -> str:
    name = filename.lower()
    if name.endswith((".xlsx", ".xls", ".csv")):
        return "work_order"
    if name.endswith((".png", ".jpg", ".jpeg")):
        return "pnid"
    stem = Path(name).stem
    if stem.startswith("sop"):
        return "sop"
    if stem.startswith("inc"):
        return "incident"
    if stem.startswith("insp"):
        return "inspection"
    if stem.startswith(("oisd", "fact", "pesa")):
        return "regulatory"
    if stem.startswith(("vn", "voice")):
        return "voice_note"
    return "sop"


def infer_source_kind(filename: str) -> str:
    n = filename.lower()
    if n.endswith((".xlsx", ".xls", ".csv")):
        return "xlsx"
    if n.endswith((".png", ".jpg", ".jpeg")):
        return "image"
    if n.endswith(".wav"):
        return "voice"
    return "pdf"


class WSManager:
    def __init__(self) -> None:
        self.clients: set[WebSocket] = set()

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        self.clients.add(ws)

    def disconnect(self, ws: WebSocket) -> None:
        self.clients.discard(ws)

    async def broadcast(self, message: dict) -> None:
        dead = []
        for ws in list(self.clients):
            try:
                await ws.send_json(message)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings.ensure_dirs()
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    app.state.repos = make_repos(fresh=False)
    app.state.llm = get_llm()
    app.state.ws = WSManager()
    mpath = settings.seed_dir / "manifest.json"
    app.state.manifest = (json.loads(mpath.read_text(encoding="utf-8"))
                          if mpath.exists() else {"documents": []})
    # M6: learn HAS_CAUSE edges from history so the graph can anticipate (N3)
    mined = m6_patterns.mine_causal(app.state.repos.graph)
    log.info("API ready: %s nodes, %s causal edges mined",
             app.state.repos.graph.summary()["nodes"], mined)
    yield


app = FastAPI(title="PlantCortex API", version="0.4.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"],
                   allow_headers=["*"])
# serve the raw corpus files for the source-viewer pane
app.mount("/files", StaticFiles(directory=str(settings.seed_dir)), name="files")


WEB_DIR = ROOT / "web"


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    idx = WEB_DIR / "index.html"
    return idx.read_text(encoding="utf-8") if idx.exists() else "<h1>PlantCortex</h1>"


@app.get("/field", response_class=HTMLResponse)
def field() -> str:
    f = WEB_DIR / "field.html"
    return f.read_text(encoding="utf-8") if f.exists() else index()


@app.get("/doc/{doc_id}")
def doc(doc_id: str):
    """Resolve a doc_id (from a citation) to its file for the source viewer."""
    for d in app.state.manifest.get("documents", []):
        if d["doc_id"] == doc_id:
            return FileResponse(settings.seed_dir / d["filename"])
    # fall back to an uploaded file
    for p in UPLOAD_DIR.glob(f"{doc_id}.*"):
        return FileResponse(p)
    return {"error": "not found", "doc_id": doc_id}


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "llm": bool(app.state.llm) and not app.state.llm.quota_blocked}


@app.get("/graph/summary")
def graph_summary() -> dict:
    return app.state.repos.graph.summary()


@app.get("/graph/subgraph")
def subgraph(center: str, hops: int = 1, limit: int = 60) -> dict:
    """Ego-graph around a node key, for the force-graph panel."""
    repos: Repos = app.state.repos
    g = repos.graph
    if not g.has_node(center):
        return {"nodes": [], "links": []}
    frontier = {center}
    seen = {center}
    for _ in range(max(1, hops)):
        nxt = set()
        for k in frontier:
            for _etype, nb, _dir in g.neighbors(k):
                if nb not in seen:
                    nxt.add(nb)
                    seen.add(nb)
                if len(seen) >= limit:
                    break
        frontier = nxt
    nodes = [{"id": k, "type": g.g.nodes[k]["type"],
              "label": _label(g.g.nodes[k])} for k in seen if g.has_node(k)]
    links = []
    for u, v, d in g.g.edges(data=True):
        if u in seen and v in seen:
            links.append({"source": u, "target": v, "type": d["type"]})
    return {"nodes": nodes, "links": links}


def _label(node: dict) -> str:
    p = node.get("props", {})
    return (p.get("tag") or p.get("wo_id") or p.get("sop_id") or p.get("code")
            or p.get("incident_id") or p.get("clause_no") or p.get("doc_id")
            or node.get("type", "?"))


@app.post("/ingest")
async def ingest(file: UploadFile = File(...),
                 doc_type: str = Form(default="")) -> dict:
    repos: Repos = app.state.repos
    filename = file.filename or "upload.bin"
    dest = UPLOAD_DIR / filename
    dest.write_bytes(await file.read())

    dtype = doc_type or infer_doc_type(filename)
    entry = {"doc_id": Path(filename).stem, "filename": filename,
             "doc_type": dtype, "source_kind": infer_source_kind(filename),
             "page_count": 1}
    # P&ID: use a co-located text/vector-layer sidecar if the client uploaded one
    if dtype == "pnid":
        layer = UPLOAD_DIR / f"{Path(filename).stem}.layer.json"
        if layer.exists():
            entry["layer_file"] = layer.name

    deltas: list[dict] = []
    loop = asyncio.get_event_loop()

    def _run() -> None:
        ingest_document(repos, entry, UPLOAD_DIR, llm=app.state.llm,
                        emit=deltas.append)
        repos.save()

    await loop.run_in_executor(None, _run)
    for d in deltas:
        await app.state.ws.broadcast(d)
    return {"ingested": entry["doc_id"], "doc_type": dtype,
            "deltas": deltas, "summary": repos.graph.summary()}


class AskRequest(BaseModel):
    question: str
    mode: Optional[str] = None            # override the router: lookup|multihop|global
    baseline: bool = False                # vector-only (bypass the graph) for comparison


@app.post("/ask")
def ask(req: AskRequest) -> dict:
    ans = ask_engine(req.question, app.state.repos, llm=app.state.llm,
                     mode=req.mode, baseline=req.baseline)
    return ans.model_dump()


@app.post("/ask/from-image")
async def ask_from_image(file: UploadFile = File(...),
                         question: str = Form(default="Give me the dossier"),
                         tag: str = Form(default="")) -> dict:
    """N2: seed retrieval from a photographed equipment tag.

    Reads the tag with the VLM when available; a typed `tag` override keeps the flow
    demoable offline (the field app lets the technician confirm the tag)."""
    detected = tag.strip()
    if not detected and app.state.llm and not app.state.llm.quota_blocked:
        raw = app.state.llm.vision_json(
            "Read the single equipment tag on this nameplate/photo. "
            "Return JSON {\"tag\": \"...\"}.", await file.read(), mime="image/jpeg")
        detected = (raw or {}).get("tag", "").strip()
    effective = f"{detected} {question}".strip() if detected else question
    ans = ask_engine(effective, app.state.repos, llm=app.state.llm, mode="multihop")
    out = ans.model_dump()
    out["detected_tag"] = detected
    return out


@app.get("/communities")
def communities() -> dict:
    from pipelines.m3_retrieval.communities import get_communities
    return {"communities": get_communities(app.state.repos, app.state.llm)}


# --- M4/M7 field surface ---------------------------------------------------- #
@app.get("/equipment/{tag}")
def equipment(tag: str) -> dict:
    return build_dossier(app.state.repos.graph, tag)


class CaptureReq(BaseModel):
    transcript: str
    author_role: Optional[str] = None


@app.post("/capture")
async def capture(req: CaptureReq) -> dict:
    """M7 Knowledge Capture: a note becomes a citable graph node immediately."""
    res = capture_note(app.state.repos, req.transcript, req.author_role)
    await app.state.ws.broadcast({
        "type": "graph.delta", "doc_id": res["note_id"], "doc_type": "voice_note",
        "nodes_added": res["nodes_added"], "edges_added": res["edges_added"],
        "chunks": 1, "graph": {"nodes": app.state.repos.graph.g.number_of_nodes(),
                               "edges": app.state.repos.graph.g.number_of_edges()}})
    return res


# --- M5 compliance ---------------------------------------------------------- #
class ComplianceReq(BaseModel):
    standard_id: str
    scope: Optional[str] = None


@app.post("/compliance/scan")
def compliance_scan(req: ComplianceReq) -> dict:
    rep = m5_compliance.scan(app.state.repos, settings.seed_dir, app.state.manifest,
                             req.standard_id, req.scope)
    return {"standard": rep.standard, "scope": rep.scope, "summary": rep.summary,
            "verdicts": [asdict(v) for v in rep.verdicts]}


@app.get("/timeline/{tag}")
def timeline(tag: str) -> dict:
    """Dated event history for an asset — powers the Time-Lens failure replay."""
    from agents.m6_patterns import asset_events
    from core.ontology import normalize_tag

    key = f"Equipment:{normalize_tag(tag)}"
    g = app.state.repos.graph
    if not g.has_node(key):
        return {"tag": normalize_tag(tag), "events": []}
    events = [{"id": e.id, "kind": e.kind, "date": e.date.isoformat(),
               "codes": sorted(e.codes), "text": (e.text or "")[:90]}
              for e in asset_events(g, key) if e.date]
    return {"tag": normalize_tag(tag), "events": events}


# --- M6 patterns & cause ranking ------------------------------------------- #
@app.get("/patterns")
def patterns(limit: int = 8) -> dict:
    cards = m6_patterns.pattern_cards(app.state.repos.graph)
    return {"cards": [asdict(c) for c in cards[:limit]]}


class CauseReq(BaseModel):
    wo_text: str


@app.post("/patterns/cause")
def cause(req: CauseReq) -> dict:
    return m6_patterns.rank_causes(app.state.repos.graph, req.wo_text)


@app.get("/compliance/report/{standard_id}")
def compliance_report(standard_id: str, scope: Optional[str] = None):
    rep = m5_compliance.scan(app.state.repos, settings.seed_dir, app.state.manifest,
                             standard_id, scope)
    out = UPLOAD_DIR / f"compliance_{standard_id}.pdf"
    m5_compliance.export_pdf(rep, out)
    return FileResponse(out, filename=f"Compliance_{standard_id}.pdf",
                        media_type="application/pdf")


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket) -> None:
    await app.state.ws.connect(ws)
    try:
        await ws.send_json({"type": "hello",
                            "summary": app.state.repos.graph.summary()})
        while True:
            await ws.receive_text()  # keepalive; client may ping
    except WebSocketDisconnect:
        app.state.ws.disconnect(ws)
