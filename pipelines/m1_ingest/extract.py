"""Ontology-constrained extraction.

* ``llm_extract`` — the primary path: a JSON-mode LLM call constrained to the ontology.
* ``rule_based_extract`` — deterministic fallback (regex tags + failure-keyword map).
  Used offline (no key), as a safety net when the LLM returns nothing, and for the
  structured work-order rows where columns already give us the tag/date/type.
"""

from __future__ import annotations

import json
import re

from core.config import ROOT
from core.ontology import (
    ExtractedEdge, ExtractedNode, ExtractionResult, NodeType, EdgeType,
    normalize_tag, ontology_prompt_spec,
)

# Equipment tag prefixes present in this plant (excludes L-#### process lines).
_EQUIP_PREFIXES = {"P", "V", "E", "FV", "PT", "T", "C", "LT", "TT", "FT",
                   "PV", "LV", "TV"}
_TAG_RE = re.compile(r"\b([A-Za-z]{1,3})[- ]?(\d{2,4})([A-Za-z])?\b")

# Failure keyword -> FailureMode code, in priority order (first match wins).
_FAILURE_RULES: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\b(seal|seel)\b.*\b(leak|lkg|fail|leakage)\b|external leak", re.I), "ELP"),
    (re.compile(r"internal leak", re.I), "INL"),
    (re.compile(r"\bvib(ration|n)?\b", re.I), "VIB"),
    (re.compile(r"overheat|over ?heating|high temp|hot bearing", re.I), "OHE"),
    (re.compile(r"stuck|seiz(e|ed)|stiff|jam", re.I), "STU"),
    (re.compile(r"chok(e|ed)|plug(ged)?|block(ed|age)", re.I), "PLU"),
    (re.compile(r"abnormal (instrument )?reading|reading drift|drift|spurious", re.I), "AIR"),
    (re.compile(r"noise|noisy", re.I), "NOI"),
    (re.compile(r"wear|erosion", re.I), "WEO"),
    (re.compile(r"fail(s|ed)? to start|won'?t start", re.I), "STD"),
    (re.compile(r"breakdown|broke down", re.I), "BRD"),
    (re.compile(r"\bleak(age)?\b|weep", re.I), "ELP"),
]


def extract_tags(text: str) -> list[str]:
    """Return canonical equipment tags found in free text (known prefixes only)."""
    found: list[str] = []
    for m in _TAG_RE.finditer(text or ""):
        prefix = m.group(1).upper()
        if prefix not in _EQUIP_PREFIXES:
            continue
        tag = normalize_tag(f"{prefix}{m.group(2)}{m.group(3) or ''}")
        if tag not in found:
            found.append(tag)
    return found


def failure_codes(text: str, max_codes: int = 2) -> list[str]:
    codes: list[str] = []
    for pat, code in _FAILURE_RULES:
        if pat.search(text or "") and code not in codes:
            codes.append(code)
        if len(codes) >= max_codes:
            break
    return codes


def rule_based_extract(text: str) -> ExtractionResult:
    """Deterministic Equipment + FailureMode extraction with co-occurrence EXHIBITS."""
    tags = extract_tags(text)
    codes = failure_codes(text)
    nodes: list[ExtractedNode] = []
    for t in tags:
        nodes.append(ExtractedNode(type=NodeType.EQUIPMENT,
                                   properties={"tag": t}, confidence=0.6,
                                   evidence_span=t))
    for c in codes:
        nodes.append(ExtractedNode(type=NodeType.FAILURE_MODE,
                                   properties={"code": c}, confidence=0.55))
    # Note: rule-based cannot know if it's a WO or Incident exhibiting the mode,
    # so it does not emit EXHIBITS edges (avoids wrong-typed edges). The structured
    # work-order path adds those edges explicitly.
    return ExtractionResult(nodes=nodes, edges=[])


# --------------------------------------------------------------------------- #
# Doc-type-aware structured extraction (deterministic, offline, high precision).
# Used as the fallback when the LLM is unavailable, and for the well-structured
# prose in this corpus. The LLM path generalizes to arbitrary unseen documents.
# --------------------------------------------------------------------------- #
_SOP_TITLE = re.compile(r"(SOP-\d+)\s+(.*?)\s*\(Rev\s*([\w.]+)\)", re.I)
# clause number without a trailing sentence period (e.g. "clause 6.2." -> "6.2")
_STD_CLAUSE = re.compile(
    r"\b([A-Z][A-Z0-9]+(?:-[A-Z0-9]+)+)\s+clause\s+(\d+(?:\.\d+)*)", re.I)
_INC_ID = re.compile(r"(INC-\d{4}-\d+)")
_INC_DATE = re.compile(r"Date\s*\n?\s*(\d{4}-\d{2}-\d{2})", re.I)
_INC_SEV = re.compile(r"Severity\s*\n?\s*(near-miss|minor|major|fatal)", re.I)
_CLAUSE_BLOCK = re.compile(r"Clause\s+([\d.]+)\s*\n(.+?)(?=\n\s*Clause\s+[\d.]+|\n\s*Applicability|\Z)",
                           re.I | re.S)
_REF_EQUIP = re.compile(r"(?:Referenced equipment|Applicable Equipment)[:\s]*\n?([^\n]+(?:\n[^\n]+)?)", re.I)


def _section_after(text: str, header: str, stop_headers: list[str]) -> str:
    m = re.search(rf"{re.escape(header)}\s*\n(.*?)(?=\n\s*(?:{'|'.join(stop_headers)})|\Z)",
                  text, re.I | re.S)
    return m.group(1).strip() if m else ""


def _sop_extract(text: str) -> ExtractionResult:
    nodes, edges = [], []
    tm = _SOP_TITLE.search(text)
    if not tm:
        return rule_based_extract(text)
    sop_id, title, rev = tm.group(1).upper(), tm.group(2).strip(), tm.group(3)
    nodes.append(ExtractedNode(type=NodeType.PROCEDURE,
                               properties={"sop_id": sop_id, "title": title,
                                           "revision": rev}, confidence=0.9,
                               evidence_span=f"{sop_id} {title}"))
    equip_region = _section_after(text, "Applicable Equipment",
                                  ["Referenced Standards", "Procedure", r"\d\.\s"])
    for t in extract_tags(equip_region or text):
        nodes.append(ExtractedNode(type=NodeType.EQUIPMENT, properties={"tag": t},
                                   confidence=0.85))
        edges.append(ExtractedEdge(type=EdgeType.GOVERNS, source_ref=sop_id,
                                   target_ref=t, confidence=0.85))
    for std, clause in _STD_CLAUSE.findall(text):
        edges.append(ExtractedEdge(type=EdgeType.REQUIRES, source_ref=sop_id,
                                   target_ref=f"{std.upper()}:{clause}", confidence=0.9,
                                   evidence_span=f"{std} clause {clause}"))
    return ExtractionResult(nodes=nodes, edges=edges)


def _regulatory_extract(text: str) -> ExtractionResult:
    nodes, edges = [], []
    first = text.strip().splitlines()[0] if text.strip() else ""
    stdm = re.match(r"\s*([A-Z][A-Z0-9]+(?:-[A-Z0-9]+)+)\s+(.*)", first)
    standard = stdm.group(1) if stdm else "UNKNOWN"
    ref_m = _REF_EQUIP.search(text)
    ref_tags = extract_tags(ref_m.group(1)) if ref_m else []
    for clause_no, body in _CLAUSE_BLOCK.findall(text):
        key = f"{standard}:{clause_no}"
        nodes.append(ExtractedNode(type=NodeType.REGULATORY_CLAUSE,
                                   properties={"standard": standard,
                                               "clause_no": clause_no,
                                               "text_summary": body.strip()[:160]},
                                   confidence=0.9, evidence_span=body.strip()[:60]))
        for t in ref_tags:
            edges.append(ExtractedEdge(type=EdgeType.COVERS, source_ref=key,
                                       target_ref=t, confidence=0.8))
    return ExtractionResult(nodes=nodes, edges=edges) if nodes else rule_based_extract(text)


def _incident_extract(text: str) -> ExtractionResult:
    nodes, edges = [], []
    idm = _INC_ID.search(text)
    if not idm:
        return rule_based_extract(text)
    inc_id = idm.group(1)
    props = {"incident_id": inc_id}
    if (dm := _INC_DATE.search(text)):
        props["date"] = dm.group(1)
    if (sm := _INC_SEV.search(text)):
        props["severity"] = sm.group(1).lower()
    nodes.append(ExtractedNode(type=NodeType.INCIDENT, properties=props,
                               confidence=0.9, evidence_span=inc_id))
    tags = extract_tags(text)
    for t in tags:
        nodes.append(ExtractedNode(type=NodeType.EQUIPMENT, properties={"tag": t},
                                   confidence=0.85))
        edges.append(ExtractedEdge(type=EdgeType.OCCURRED_AT, source_ref=inc_id,
                                   target_ref=t, confidence=0.85))
    for code in failure_codes(text):
        nodes.append(ExtractedNode(type=NodeType.FAILURE_MODE,
                                   properties={"code": code}, confidence=0.7))
        edges.append(ExtractedEdge(type=EdgeType.EXHIBITS, source_ref=inc_id,
                                   target_ref=code, confidence=0.7))
    return ExtractionResult(nodes=nodes, edges=edges)


def structured_extract(text: str, doc_type: str) -> ExtractionResult:
    """Deterministic extraction dispatched by document type."""
    if doc_type == "sop":
        return _sop_extract(text)
    if doc_type == "regulatory":
        return _regulatory_extract(text)
    if doc_type == "incident":
        return _incident_extract(text)
    return rule_based_extract(text)


_EXTRACT_TMPL: str | None = None


def _build_system_prompt(doc_id: str, doc_type: str) -> str:
    global _EXTRACT_TMPL
    if _EXTRACT_TMPL is None:
        _EXTRACT_TMPL = (ROOT / "prompts" / "extract.md").read_text(encoding="utf-8")
    spec = json.dumps(ontology_prompt_spec(), indent=2)
    return (_EXTRACT_TMPL
            .replace("{ontology_spec}", spec)
            .replace("{doc_id}", doc_id)
            .replace("{doc_type}", doc_type))


def _coerce(result_dict: dict) -> ExtractionResult:
    """Validate LLM JSON into ExtractionResult, dropping malformed entries."""
    nodes, edges = [], []
    for n in (result_dict.get("nodes") or []):
        try:
            nodes.append(ExtractedNode.model_validate(n))
        except Exception:
            continue
    for e in (result_dict.get("edges") or []):
        try:
            edges.append(ExtractedEdge.model_validate(e))
        except Exception:
            continue
    return ExtractionResult(nodes=nodes, edges=edges)


def llm_extract(text: str, doc_id: str, doc_type: str, llm) -> ExtractionResult:
    """Primary extraction path. Falls back to doc-type structured extraction when the
    LLM is unavailable (no key / quota) or returns nothing."""
    if llm is None or getattr(llm, "quota_blocked", False):
        return structured_extract(text, doc_type)
    system = _build_system_prompt(doc_id, doc_type)
    raw = llm.complete_json(system, f"TEXT:\n{text}", max_tokens=3072)
    result = _coerce(raw)
    if not result.nodes and not result.edges:
        return structured_extract(text, doc_type)
    return result
