"""M5 — Compliance Gap Agent.

For a regulatory standard, check each clause's obligation against the procedures that
are supposed to satisfy it (procedures that REQUIRE the clause, or that GOVERN the
equipment the clause COVERS). Verdict per clause: covered | partial | GAP, with
side-by-side clause-vs-procedure evidence.

Coverage is measured deterministically (offline) as the fraction of the clause's
obligation terms present in the procedure text — so the seeded SOP-17 gap (confined-space
entry that omits the pre-entry gas testing OISD-STD-105 cl.7.3 demands) surfaces as a GAP
because the procedure text lacks the atmosphere/oxygen/gas-testing terms. An LLM can
refine obligation extraction when available; the deterministic path stands alone.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from pipelines.m1_ingest import parse

# Generic procedural / legal words that are not obligations in themselves.
_STOP = set(
    "shall prior during before after obtain obtained obtaining displayed valid "
    "activity area areas work maintenance intrusive routine non-routine job jobs "
    "process equipment person persons personnel any all the and for with from that "
    "this these those into out any repeated defined until unless likely present "
    "wear worn made carried out place placed within during time times each other "
    "clause standard procedure applicable referenced scope demo synthetic text "
    "prepared official published not this is are be been being have has had will "
    "provided identify identified logged log entrant supervisor standby "
    "commence commenced commences step steps".split())

_COVERED = 0.5
_PARTIAL = 0.25

# Safety-critical obligation terms. A clause's requirement hinges on these; if the
# procedure omits the clause's critical terms, it is a GAP even if generic wording
# ("confined space entry") overlaps.
_CRITICAL = set(
    "gas gases atmosphere atmospheric oxygen flammable toxic testing tested test "
    "fire watch lock locked tag tagged isolate isolated isolation blind blinded "
    "ventilate ventilation relief monitor monitored permit calibration inspection "
    "vibration lel h2s breathing".split())


def _terms(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z][a-z0-9\-]{3,}", (text or "").lower())
            if t not in _STOP}


def coverage(clause_text: str, proc_text: str) -> float:
    obl = _terms(clause_text)
    if not obl:
        return 1.0
    return len(obl & _terms(proc_text)) / len(obl)


def assess(clause_text: str, proc_text: str) -> tuple[float, str]:
    """Return (score, verdict). Weighted toward the clause's critical safety terms."""
    obl = _terms(clause_text)
    ptok = _terms(proc_text)
    gen = len(obl & ptok) / len(obl) if obl else 1.0
    crit = obl & _CRITICAL
    crit_cov = (len(crit & ptok) / len(crit)) if crit else gen
    score = crit_cov if crit else gen
    if crit_cov >= _COVERED and gen >= 0.3:
        verdict = "covered"
    elif crit_cov < _PARTIAL:
        verdict = "GAP"
    else:
        verdict = "partial"
    return round(score, 2), verdict


@dataclass
class ClauseVerdict:
    standard: str
    clause_no: str
    clause_text: str
    verdict: str                       # covered | partial | GAP
    coverage: float
    procedure: str | None = None       # best-matching procedure sop_id
    procedure_evidence: str = ""
    covers_equipment: list[str] = field(default_factory=list)


@dataclass
class ComplianceReport:
    standard: str
    scope: str | None
    verdicts: list[ClauseVerdict]

    @property
    def summary(self) -> dict:
        c = {"covered": 0, "partial": 0, "GAP": 0}
        for v in self.verdicts:
            c[v.verdict] = c.get(v.verdict, 0) + 1
        return c


def _proc_text_reader(seed_dir: Path, manifest: dict):
    fn = {d["doc_id"]: d["filename"] for d in manifest.get("documents", [])}

    @lru_cache(maxsize=64)
    def read(sop_id: str) -> str:
        f = fn.get(sop_id)
        if not f:
            return ""
        return " ".join(parse.read_pdf_pages(seed_dir / f))

    return read


def scan(repos, seed_dir: Path, manifest: dict, standard_id: str,
         scope: str | None = None) -> ComplianceReport:
    graph = repos.graph
    read_proc = _proc_text_reader(seed_dir, manifest)
    clause_keys = [k for k in graph.nodes_by_type("RegulatoryClause")
                   if k.startswith(f"RegulatoryClause:{standard_id}:")]

    verdicts: list[ClauseVerdict] = []
    for ck in clause_keys:
        node = graph.get_node(ck)
        props = node["props"]
        ctext = props.get("text_summary", "")
        if not ctext or not props.get("clause_no"):
            continue                                  # skip malformed/stub clause nodes
        # The procedure that REQUIRES a clause owns its implementation; only if no
        # procedure cites the clause do we fall back to procedures governing the
        # equipment it covers. (This stops an unrelated SOP that merely shares safety
        # vocabulary from masking a real gap.)
        requiring: set[str] = set()
        covered_equip: list[str] = []
        for e in graph.edges_incident(ck):
            if e["type"] == "REQUIRES" and e["target"] == ck:
                requiring.add(e["source"])            # Procedure:...
            if e["type"] == "COVERS" and e["source"] == ck:
                covered_equip.append(e["target"])     # Equipment:...
        candidates: set[str] = set(requiring)
        if not candidates:
            for eq in covered_equip:
                if scope and scope.upper() not in eq.upper():
                    continue
                for e in graph.edges_incident(eq):
                    if e["type"] == "GOVERNS" and e["target"] == eq:
                        candidates.add(e["source"])

        if scope:
            eq_in_scope = [eq for eq in covered_equip if scope.upper() in eq.upper()]
            if covered_equip and not eq_in_scope:
                continue                              # clause not in requested scope

        best_proc, best_cov, best_verdict = None, -1.0, "GAP"
        rank = {"GAP": 0, "partial": 1, "covered": 2}
        for proc_key in candidates:
            sop_id = proc_key.split(":", 1)[1]
            score, verdict = assess(ctext, read_proc(sop_id))
            # keep the most-satisfying procedure; name it even on a GAP so the report
            # can say WHICH procedure was responsible but fell short.
            if (rank[verdict], score) >= (rank[best_verdict], best_cov):
                best_proc, best_cov, best_verdict = sop_id, score, verdict

        verdict = "GAP" if not candidates else best_verdict
        best_cov = max(best_cov, 0.0)

        evidence = ""
        if best_proc:
            evidence = _best_snippet(read_proc(best_proc), ctext)
        verdicts.append(ClauseVerdict(
            standard=standard_id, clause_no=props.get("clause_no", "?"),
            clause_text=ctext, verdict=verdict, coverage=round(best_cov, 2),
            procedure=best_proc, procedure_evidence=evidence,
            covers_equipment=[e.split(":", 1)[1] for e in covered_equip]))

    verdicts.sort(key=lambda v: {"GAP": 0, "partial": 1, "covered": 2}[v.verdict])
    return ComplianceReport(standard=standard_id, scope=scope, verdicts=verdicts)


def export_pdf(report: ComplianceReport, out_path: Path) -> Path:
    """Audit-ready Compliance Evidence Package (PDF)."""
    from datetime import datetime

    from fpdf import FPDF
    from fpdf.enums import XPos, YPos

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    def cell(h, txt, style="", size=10, r=0, g=0, b=0):
        pdf.set_font("Helvetica", style, size)
        pdf.set_text_color(r, g, b)
        pdf.multi_cell(pdf.epw, h, txt, new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    cell(9, f"Compliance Evidence Package - {report.standard}", "B", 15)
    s = report.summary
    cell(6, f"Generated {datetime.now():%Y-%m-%d %H:%M} | scope: {report.scope or 'all'} "
            f"| covered {s['covered']}  partial {s['partial']}  GAP {s['GAP']}",
         "", 10, 90, 90, 90)
    pdf.ln(2)

    colors = {"covered": (22, 150, 90), "partial": (200, 140, 0), "GAP": (200, 40, 40)}
    for v in report.verdicts:
        r, gg, b = colors.get(v.verdict, (0, 0, 0))
        cell(7, f"Clause {v.clause_no}  -  {v.verdict.upper()}  (coverage {int(v.coverage*100)}%)",
             "B", 12, r, gg, b)
        cell(6, f"Requirement: {v.clause_text}")
        if v.covers_equipment:
            cell(6, f"Applies to: {', '.join(v.covers_equipment)}", "", 9, 90, 90, 90)
        if v.verdict == "GAP":
            resp = f" (responsible procedure: {v.procedure})" if v.procedure else ""
            cell(6, f"FINDING: No procedure satisfies this obligation{resp}. "
                    f"The clause's safety-critical requirement is not evidenced.",
                 "I", 10, 150, 30, 30)
        else:
            cell(6, f"Satisfied by {v.procedure}: \"{v.procedure_evidence}\"",
                 "", 9, 40, 90, 40)
        pdf.ln(2)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    pdf.output(str(out_path))
    return out_path


def _best_snippet(proc_text: str, clause_text: str) -> str:
    qtok = _terms(clause_text)
    best, best_ov = "", -1
    for s in re.split(r"(?<=[.\n])\s+", proc_text):
        ov = len(_terms(s) & qtok)
        if ov > best_ov and s.strip():
            best, best_ov = s.strip(), ov
    return best[:200]
