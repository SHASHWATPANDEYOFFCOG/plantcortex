"""PlantCortex ontology — the single source of truth for the knowledge graph.

Everything downstream (LLM extractor, cross-modal resolver, graph repository,
retrieval, compliance and pattern agents) imports its node/edge types, the
``Provenance`` model, and the ``FailureMode`` code taxonomy from here.

Design notes
------------
* Every node and edge carries ``provenance`` (``list[Provenance]``) so that every
  fact in the graph is traceable to a document region / spreadsheet row / timestamp.
  Provenance is what powers clickable citations, the audit trail, and cross-modal
  entity resolution (novelty N1).
* Each node exposes ``.key()`` — the canonical, deterministic identifier used for
  idempotent graph ``MERGE`` upserts. For ``Equipment`` this is the tag (``P-101A``),
  which is exactly why the same pump in a drawing, a spreadsheet and an SOP collapses
  to one node.
* ``ontology_prompt_spec()`` renders a compact machine-readable schema that is injected
  into the extraction prompt (``prompts/extract.md``) so the LLM is constrained to this
  ontology and never invents node types.
"""

from __future__ import annotations

from datetime import date, datetime
from enum import Enum
from typing import Annotated, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field


# --------------------------------------------------------------------------- #
# Enums
# --------------------------------------------------------------------------- #
class NodeType(str, Enum):
    EQUIPMENT = "Equipment"
    COMPONENT = "Component"
    FAILURE_MODE = "FailureMode"
    WORK_ORDER = "WorkOrder"
    INCIDENT = "Incident"
    PROCEDURE = "Procedure"
    PERMIT = "Permit"
    REGULATORY_CLAUSE = "RegulatoryClause"
    PERSON = "Person"
    DOCUMENT = "Document"
    CHUNK = "Chunk"
    TACIT_NOTE = "TacitNote"


class EdgeType(str, Enum):
    CONNECTED_TO = "CONNECTED_TO"        # Equipment <-> Equipment (from P&ID)
    PART_OF = "PART_OF"                  # Component -> Equipment
    PERFORMED_ON = "PERFORMED_ON"        # WorkOrder -> Equipment
    EXHIBITS = "EXHIBITS"                # WorkOrder/Incident -> FailureMode
    HAS_CAUSE = "HAS_CAUSE"              # FailureMode -> Component | FailureMode
    GOVERNS = "GOVERNS"                  # Procedure -> Equipment
    REQUIRES = "REQUIRES"                # Procedure -> RegulatoryClause
    COVERS = "COVERS"                    # RegulatoryClause -> Equipment/Activity
    ISSUED_FOR = "ISSUED_FOR"            # Permit -> Equipment/Area
    MENTIONS = "MENTIONS"               # Chunk -> any node
    EXTRACTED_FROM = "EXTRACTED_FROM"    # any node -> Document
    ABOUT = "ABOUT"                      # TacitNote -> Equipment/FailureMode
    OCCURRED_AT = "OCCURRED_AT"          # Incident -> Equipment
    SAME_AS = "SAME_AS"                  # resolution provenance event (N1)


class WorkOrderType(str, Enum):
    CORRECTIVE = "corrective"
    PREVENTIVE = "preventive"


class IncidentSeverity(str, Enum):
    NEAR_MISS = "near-miss"
    MINOR = "minor"
    MAJOR = "major"
    FATAL = "fatal"


class PermitType(str, Enum):
    HOT_WORK = "hot-work"
    CONFINED_SPACE = "confined-space"
    ELECTRICAL = "electrical"


class SourceKind(str, Enum):
    PDF = "pdf"
    XLSX = "xlsx"
    IMAGE = "image"
    VOICE = "voice"
    DOCX = "docx"


class ExtractorKind(str, Enum):
    """Which stage produced a node/edge — recorded in provenance."""
    SEED = "seed"                 # ground-truth from the corpus generator
    LLM_TEXT = "llm_text"
    LLM_VISION = "llm_vision"
    TLP = "tlp"
    GEOMETRY = "geometry"         # P&ID line-adjacency heuristic
    RESOLVER = "resolver"         # cross-modal merge event
    ASR = "asr"                   # speech-to-text (voice notes)


class FailureModeCode(str, Enum):
    """ISO-14224-inspired failure-mode taxonomy (compact subset for the demo)."""
    ELP = "ELP"   # External leakage - process medium
    INL = "INL"   # Internal leakage
    VIB = "VIB"   # Vibration
    OHE = "OHE"   # Overheating
    NOI = "NOI"   # Noise
    STU = "STU"   # Stuck / seized
    AIR = "AIR"   # Abnormal instrument reading
    STD = "STD"   # Fails to start on demand
    BRD = "BRD"   # Breakdown
    PLU = "PLU"   # Plugged / choked
    WEO = "WEO"   # Wear / erosion


FAILURE_MODE_NAMES: dict[str, str] = {
    "ELP": "External leakage (process medium)",
    "INL": "Internal leakage",
    "VIB": "Vibration",
    "OHE": "Overheating",
    "NOI": "Noise",
    "STU": "Stuck / seized",
    "AIR": "Abnormal instrument reading",
    "STD": "Fails to start on demand",
    "BRD": "Breakdown",
    "PLU": "Plugged / choked",
    "WEO": "Wear / erosion",
}


# --------------------------------------------------------------------------- #
# Provenance
# --------------------------------------------------------------------------- #
class Provenance(BaseModel):
    """Where a fact came from. Attached to every node and edge."""
    model_config = ConfigDict(extra="forbid")

    doc_id: str
    extractor: ExtractorKind = ExtractorKind.LLM_TEXT
    confidence: float = Field(1.0, ge=0.0, le=1.0)
    page: Optional[int] = None
    # bbox in pixel coords [x0, y0, x1, y1] for image/PDF regions
    bbox: Optional[tuple[float, float, float, float]] = None
    row: Optional[int] = None            # spreadsheet row index
    timestamp: Optional[datetime] = None  # for voice notes / time-anchored facts
    evidence_span: Optional[str] = None   # verbatim supporting text (<= ~15 words)


# --------------------------------------------------------------------------- #
# Node models
# --------------------------------------------------------------------------- #
class _Node(BaseModel):
    """Base for all graph nodes."""
    model_config = ConfigDict(extra="forbid", use_enum_values=False)

    provenance: list[Provenance] = Field(default_factory=list)

    def key(self) -> str:  # pragma: no cover - overridden by subclasses
        raise NotImplementedError


class Equipment(_Node):
    node_type: Literal[NodeType.EQUIPMENT] = NodeType.EQUIPMENT
    tag: str                              # canonical ID, e.g. "P-101A"
    name: Optional[str] = None
    type: Optional[str] = None            # pump, vessel, valve, exchanger, instrument
    unit: Optional[str] = None
    location: Optional[str] = None

    def key(self) -> str:
        return f"Equipment:{normalize_tag(self.tag)}"


class Component(_Node):
    node_type: Literal[NodeType.COMPONENT] = NodeType.COMPONENT
    name: str                             # mechanical seal, bearing, impeller
    parent_equipment: Optional[str] = None  # tag of parent Equipment

    def key(self) -> str:
        parent = normalize_tag(self.parent_equipment) if self.parent_equipment else "?"
        return f"Component:{parent}:{self.name.strip().lower()}"


class FailureMode(_Node):
    node_type: Literal[NodeType.FAILURE_MODE] = NodeType.FAILURE_MODE
    code: FailureModeCode
    name: Optional[str] = None

    def key(self) -> str:
        code = self.code.value if isinstance(self.code, FailureModeCode) else self.code
        return f"FailureMode:{code}"


class WorkOrder(_Node):
    node_type: Literal[NodeType.WORK_ORDER] = NodeType.WORK_ORDER
    wo_id: str
    date: Optional[date] = None
    type: Optional[WorkOrderType] = None
    problem_text: Optional[str] = None
    action_text: Optional[str] = None
    technician: Optional[str] = None

    def key(self) -> str:
        return f"WorkOrder:{self.wo_id}"


class Incident(_Node):
    node_type: Literal[NodeType.INCIDENT] = NodeType.INCIDENT
    incident_id: str
    date: Optional[date] = None
    severity: Optional[IncidentSeverity] = None
    description: Optional[str] = None

    def key(self) -> str:
        return f"Incident:{self.incident_id}"


class Procedure(_Node):
    node_type: Literal[NodeType.PROCEDURE] = NodeType.PROCEDURE
    sop_id: str
    title: Optional[str] = None
    revision: Optional[str] = None
    scope: Optional[str] = None

    def key(self) -> str:
        return f"Procedure:{self.sop_id}"


class Permit(_Node):
    node_type: Literal[NodeType.PERMIT] = NodeType.PERMIT
    permit_id: str
    type: Optional[PermitType] = None
    area: Optional[str] = None
    valid_from: Optional[date] = None
    valid_to: Optional[date] = None

    def key(self) -> str:
        return f"Permit:{self.permit_id}"


class RegulatoryClause(_Node):
    node_type: Literal[NodeType.REGULATORY_CLAUSE] = NodeType.REGULATORY_CLAUSE
    standard: str                         # e.g. "OISD-STD-105"
    clause_no: str
    text_summary: Optional[str] = None

    def key(self) -> str:
        return f"RegulatoryClause:{self.standard}:{self.clause_no}"


class Person(_Node):
    node_type: Literal[NodeType.PERSON] = NodeType.PERSON
    name: str                             # synthetic / anonymized
    role: Optional[str] = None

    def key(self) -> str:
        return f"Person:{self.name.strip().lower()}"


class Document(_Node):
    node_type: Literal[NodeType.DOCUMENT] = NodeType.DOCUMENT
    doc_id: str
    filename: Optional[str] = None
    doc_type: Optional[str] = None        # work_order | sop | incident | pnid | ...
    page_count: Optional[int] = None
    ingest_time: Optional[datetime] = None
    source_kind: Optional[SourceKind] = None

    def key(self) -> str:
        return f"Document:{self.doc_id}"


class Chunk(_Node):
    node_type: Literal[NodeType.CHUNK] = NodeType.CHUNK
    chunk_id: str
    doc_id: str
    page: Optional[int] = None
    bbox: Optional[tuple[float, float, float, float]] = None
    text: str = ""
    embedding_ref: Optional[str] = None

    def key(self) -> str:
        return f"Chunk:{self.chunk_id}"


class TacitNote(_Node):
    node_type: Literal[NodeType.TACIT_NOTE] = NodeType.TACIT_NOTE
    note_id: str
    author_role: Optional[str] = None
    date: Optional[date] = None
    transcript: str = ""

    def key(self) -> str:
        return f"TacitNote:{self.note_id}"


# Discriminated union of all concrete node types.
AnyNode = Annotated[
    Union[
        Equipment, Component, FailureMode, WorkOrder, Incident, Procedure,
        Permit, RegulatoryClause, Person, Document, Chunk, TacitNote,
    ],
    Field(discriminator="node_type"),
]

NODE_MODELS: dict[NodeType, type[_Node]] = {
    NodeType.EQUIPMENT: Equipment,
    NodeType.COMPONENT: Component,
    NodeType.FAILURE_MODE: FailureMode,
    NodeType.WORK_ORDER: WorkOrder,
    NodeType.INCIDENT: Incident,
    NodeType.PROCEDURE: Procedure,
    NodeType.PERMIT: Permit,
    NodeType.REGULATORY_CLAUSE: RegulatoryClause,
    NodeType.PERSON: Person,
    NodeType.DOCUMENT: Document,
    NodeType.CHUNK: Chunk,
    NodeType.TACIT_NOTE: TacitNote,
}


# --------------------------------------------------------------------------- #
# Edge model
# --------------------------------------------------------------------------- #
class Edge(BaseModel):
    """A typed, provenance-carrying relationship between two node keys."""
    model_config = ConfigDict(extra="forbid")

    type: EdgeType
    source_key: str                       # source node .key()
    target_key: str                       # target node .key()
    provenance: list[Provenance] = Field(default_factory=list)

    def key(self) -> str:
        t = self.type.value if isinstance(self.type, EdgeType) else self.type
        return f"{self.source_key}-[{t}]->{self.target_key}"


# Which (source NodeType -> target NodeType) pairs each edge type allows.
# Used by the resolver/validator and to steer the extraction prompt.
EDGE_DOMAIN: dict[EdgeType, list[tuple[NodeType, NodeType]]] = {
    EdgeType.CONNECTED_TO: [(NodeType.EQUIPMENT, NodeType.EQUIPMENT)],
    EdgeType.PART_OF: [(NodeType.COMPONENT, NodeType.EQUIPMENT)],
    EdgeType.PERFORMED_ON: [(NodeType.WORK_ORDER, NodeType.EQUIPMENT)],
    EdgeType.EXHIBITS: [
        (NodeType.WORK_ORDER, NodeType.FAILURE_MODE),
        (NodeType.INCIDENT, NodeType.FAILURE_MODE),
    ],
    EdgeType.HAS_CAUSE: [
        (NodeType.FAILURE_MODE, NodeType.COMPONENT),
        (NodeType.FAILURE_MODE, NodeType.FAILURE_MODE),
    ],
    EdgeType.GOVERNS: [(NodeType.PROCEDURE, NodeType.EQUIPMENT)],
    EdgeType.REQUIRES: [(NodeType.PROCEDURE, NodeType.REGULATORY_CLAUSE)],
    EdgeType.COVERS: [(NodeType.REGULATORY_CLAUSE, NodeType.EQUIPMENT)],
    EdgeType.ISSUED_FOR: [(NodeType.PERMIT, NodeType.EQUIPMENT)],
    EdgeType.MENTIONS: [(NodeType.CHUNK, nt) for nt in NodeType],
    EdgeType.EXTRACTED_FROM: [(nt, NodeType.DOCUMENT) for nt in NodeType],
    EdgeType.ABOUT: [
        (NodeType.TACIT_NOTE, NodeType.EQUIPMENT),
        (NodeType.TACIT_NOTE, NodeType.FAILURE_MODE),
    ],
    EdgeType.OCCURRED_AT: [(NodeType.INCIDENT, NodeType.EQUIPMENT)],
    EdgeType.SAME_AS: [(nt, nt) for nt in NodeType],
}


# --------------------------------------------------------------------------- #
# Extraction I/O (what the LLM produces before validation/resolution)
# --------------------------------------------------------------------------- #
class ExtractedNode(BaseModel):
    """Loose node as returned by the extraction LLM (validated into AnyNode later)."""
    model_config = ConfigDict(extra="allow")

    type: NodeType
    properties: dict = Field(default_factory=dict)
    confidence: float = Field(1.0, ge=0.0, le=1.0)
    evidence_span: Optional[str] = None


class ExtractedEdge(BaseModel):
    model_config = ConfigDict(extra="allow")

    type: EdgeType
    source_ref: str                       # references an ExtractedNode (by tag/id/text)
    target_ref: str
    confidence: float = Field(1.0, ge=0.0, le=1.0)
    evidence_span: Optional[str] = None


class ExtractionResult(BaseModel):
    nodes: list[ExtractedNode] = Field(default_factory=list)
    edges: list[ExtractedEdge] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
import re

_TAG_RE = re.compile(r"^([A-Za-z]+)[\s\-_]*(\d+)[\s\-_]*([A-Za-z])?$")


def normalize_tag(tag: str) -> str:
    """Canonicalize an equipment tag: ``P101a`` / ``p-101 A`` -> ``P-101A``.

    This is the backbone of cross-modal entity resolution (N1): the same asset
    written slightly differently across a drawing, a spreadsheet and an SOP maps to
    one canonical key. Non-matching strings are returned upper-cased and trimmed.
    """
    if tag is None:
        return ""
    raw = tag.strip()
    m = _TAG_RE.match(raw)
    if not m:
        return raw.upper()
    prefix, number, suffix = m.group(1), m.group(2), m.group(3) or ""
    return f"{prefix.upper()}-{number}{suffix.upper()}"


def ontology_prompt_spec() -> dict:
    """Compact schema injected into the extraction prompt (prompts/extract.md).

    Kept small and human-readable on purpose — a full Pydantic JSON schema is too
    verbose and noisy for reliable JSON-mode extraction.
    """
    node_props: dict[str, list[str]] = {}
    for nt, model in NODE_MODELS.items():
        props = [
            name for name in model.model_fields
            if name not in ("node_type", "provenance")
        ]
        node_props[nt.value] = props
    edges = {
        et.value: [f"{s.value}->{t.value}" for (s, t) in pairs]
        for et, pairs in EDGE_DOMAIN.items()
        if et not in (EdgeType.SAME_AS,)  # internal, never LLM-emitted
    }
    return {
        "node_types": node_props,
        "edge_types": edges,
        "failure_mode_codes": FAILURE_MODE_NAMES,
        "tag_pattern_examples": ["P-101A", "P-101B", "V-201", "E-301", "FV-112", "PT-108"],
    }


__all__ = [
    "NodeType", "EdgeType", "WorkOrderType", "IncidentSeverity", "PermitType",
    "SourceKind", "ExtractorKind", "FailureModeCode", "FAILURE_MODE_NAMES",
    "Provenance", "Equipment", "Component", "FailureMode", "WorkOrder", "Incident",
    "Procedure", "Permit", "RegulatoryClause", "Person", "Document", "Chunk",
    "TacitNote", "AnyNode", "NODE_MODELS", "Edge", "EDGE_DOMAIN",
    "ExtractedNode", "ExtractedEdge", "ExtractionResult",
    "normalize_tag", "ontology_prompt_spec",
]
