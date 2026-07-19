"""M6 — Lessons-Learned & Failure Intelligence (Novelty N3).

Three capabilities over the ontology-typed graph:
  * pattern_cards  — recurring failures per asset, with *precursor* insight
                     ("P-101A: 4 seal-leak events; 3 preceded by high vibration <=30 days").
  * mine_causal    — write learned HAS_CAUSE(FailureMode->FailureMode) edges when one
                     mode reliably precedes another. The graph turns history into a causal
                     model — from answering to anticipating.
  * rank_causes    — for a NEW work-order text, extract the asset + observed mode and rank
                     probable causes by learned HAS_CAUSE + historical precursors + graph
                     proximity. Decision support with confidence, never certainty.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from datetime import date

from core.ontology import FAILURE_MODE_NAMES, normalize_tag
from pipelines.m1_ingest.extract import extract_tags, failure_codes

_WINDOW_DAYS = 30
_MIN_EVENTS = 3
_PRECURSOR_FRAC = 0.5     # a mode is a "cause" if it precedes >=50% of target events

# component signatures let us separate e.g. mechanical-seal failures from gland leaks,
# so the recurring-failure card tracks a real part rather than a coarse failure code.
_COMPONENTS = ["mechanical seal", "seal", "bearing", "gland", "impeller", "gasket",
               "coupling", "actuator", "positioner", "tube", "relief valve"]


def _component(text: str) -> str | None:
    t = (text or "").lower()
    for c in _COMPONENTS:
        if c in t:
            return "mechanical seal" if c == "seal" else c
    return None


def _parse_date(s) -> date | None:
    try:
        y, m, d = str(s)[:10].split("-")
        return date(int(y), int(m), int(d))
    except Exception:
        return None


@dataclass
class Event:
    id: str
    kind: str            # WorkOrder | Incident
    date: date | None
    codes: set[str]
    text: str


def asset_events(graph, equip_key: str) -> list[Event]:
    events: list[Event] = []
    for e in graph.edges_incident(equip_key):
        src, tgt, et = e["source"], e["target"], e["type"]
        if et == "PERFORMED_ON" and tgt == equip_key:
            ev_key, kind = src, "WorkOrder"
        elif et == "OCCURRED_AT" and tgt == equip_key:
            ev_key, kind = src, "Incident"
        else:
            continue
        n = graph.get_node(ev_key)
        if not n:
            continue
        p = n["props"]
        codes = {ie["target"].split(":", 1)[1] for ie in graph.edges_incident(ev_key)
                 if ie["type"] == "EXHIBITS" and ie["source"] == ev_key}
        events.append(Event(id=ev_key.split(":", 1)[1], kind=kind,
                            date=_parse_date(p.get("date")), codes=codes,
                            text=(p.get("problem_text") or p.get("description") or "")))
    events.sort(key=lambda x: (x.date or date.min))
    return events


def _precursor(events: list[Event], target_events: list[Event], target_code: str,
               window: int = _WINDOW_DAYS) -> dict:
    tgt = [e for e in target_events if e.date]
    if not tgt:
        return {}
    counts: Counter = Counter()
    matched_ids: dict[str, list[str]] = {}
    for te in tgt:
        preceding_codes: set[str] = set()
        for e in events:
            if e.date and e is not te and 0 < (te.date - e.date).days <= window:
                preceding_codes |= (e.codes - {target_code})
        for c in preceding_codes:
            counts[c] += 1
            matched_ids.setdefault(c, []).append(te.id)
    if not counts:
        return {"target_total": len(tgt)}
    code, matched = counts.most_common(1)[0]
    return {"precursor_code": code, "precursor_name": FAILURE_MODE_NAMES.get(code, code),
            "matched": matched, "target_total": len(tgt), "window_days": window,
            "matched_events": matched_ids.get(code, [])}


@dataclass
class PatternCard:
    equipment: str
    failure_code: str
    failure_name: str
    count: int
    span_years: float
    evidence: list[str]
    precursor: dict = field(default_factory=dict)
    recommendation: str = ""
    strength: float = 0.0


def pattern_cards(graph) -> list[PatternCard]:
    cards: list[PatternCard] = []
    for eq_key in graph.nodes_by_type("Equipment"):
        events = asset_events(graph, eq_key)
        if not events:
            continue
        tag = eq_key.split(":", 1)[1]
        # group events by (failure code, component signature)
        groups: dict[tuple, list[Event]] = {}
        for e in events:
            comp = _component(e.text)
            for code in e.codes:
                groups.setdefault((code, comp), []).append(e)
        for (code, comp), ev in groups.items():
            if len(ev) < _MIN_EVENTS:
                continue
            dates = [e.date for e in ev if e.date]
            span = ((max(dates) - min(dates)).days / 365.0) if len(dates) > 1 else 0.0
            prec = _precursor(events, ev, code)
            frac = (prec.get("matched", 0) / prec.get("target_total", 1)
                    if prec.get("target_total") else 0.0)
            name = FAILURE_MODE_NAMES.get(code, code)
            label = f"{comp} {name.lower()}" if comp else name
            rec = _recommend(tag, label, prec, frac)
            cards.append(PatternCard(
                equipment=tag, failure_code=code, failure_name=label,
                count=len(ev), span_years=round(span, 1),
                evidence=[e.id for e in ev][:8], precursor=prec, recommendation=rec,
                # precursor strength dominates: a 100%-precursor pattern is far more
                # actionable than a high-count pattern with chance-level co-occurrence.
                strength=round(len(ev) * (0.1 + 3 * frac * frac), 2)))
    cards.sort(key=lambda c: -c.strength)
    return cards


def _recommend(tag: str, label: str, prec: dict, frac: float) -> str:
    if prec.get("precursor_code") and frac >= _PRECURSOR_FRAC:
        return (f"{prec['precursor_name']} is a leading indicator: it preceded "
                f"{prec['matched']}/{prec['target_total']} {label} events on {tag} within "
                f"{prec['window_days']} days. Trend it and act proactively at the next "
                f"opportunity instead of running to failure.")
    return (f"{tag} shows recurring {label}. Review the maintenance strategy and "
            f"root-cause the repeat failures.")


def mine_causal(graph) -> int:
    """Write learned HAS_CAUSE(target_mode -> precursor_mode) edges. Returns count."""
    written = 0
    for card in pattern_cards(graph):
        prec = card.precursor
        if not prec.get("precursor_code"):
            continue
        frac = prec["matched"] / prec["target_total"]
        if frac < _PRECURSOR_FRAC or prec["precursor_code"] == card.failure_code:
            continue
        src = f"FailureMode:{card.failure_code}"
        tgt = f"FailureMode:{prec['precursor_code']}"
        prov = [{"doc_id": "M6-mined", "extractor": "pattern_mining",
                 "confidence": round(frac, 2),
                 "evidence_span": f"{card.equipment}: {prec['matched']}/"
                                  f"{prec['target_total']} within {prec['window_days']}d"}]
        for k, t in ((src, "FailureMode"), (tgt, "FailureMode")):
            if not graph.has_node(k):
                graph.upsert_node(k, t, {"code": k.split(":")[1],
                                         "name": FAILURE_MODE_NAMES.get(k.split(":")[1])})
        if graph.upsert_edge("HAS_CAUSE", src, tgt, prov):
            written += 1
    return written


def rank_causes(graph, wo_text: str, top: int = 3) -> dict:
    """Rank probable causes for a new work-order description (N3)."""
    tags = extract_tags(wo_text)
    codes = failure_codes(wo_text)
    equip = f"Equipment:{normalize_tag(tags[0])}" if tags else None
    observed = codes[0] if codes else None

    candidates: dict[str, dict] = {}

    # 1) learned HAS_CAUSE edges for the observed failure mode
    if observed:
        fk = f"FailureMode:{observed}"
        for e in graph.edges_incident(fk):
            if e["type"] == "HAS_CAUSE" and e["source"] == fk:
                cause = e["target"].split(":", 1)[1]
                conf = max((p.get("confidence", 0.5) for p in e["provenance"]), default=0.5)
                candidates[cause] = {"cause": cause,
                                     "name": FAILURE_MODE_NAMES.get(cause, cause),
                                     "score": 2.0 * conf, "basis": "learned causal link",
                                     "evidence": [p.get("evidence_span") for p in e["provenance"]]}

    # 2) historical precursors on this specific asset (optionally component-scoped)
    if equip and observed and graph.has_node(equip):
        events = asset_events(graph, equip)
        comp = _component(wo_text)
        target = [e for e in events if observed in e.codes
                  and (not comp or _component(e.text) == comp)]
        prec = _precursor(events, target, observed)
        if prec.get("precursor_code"):
            c = prec["precursor_code"]
            frac = prec["matched"] / prec["target_total"]
            cur = candidates.get(c, {"cause": c, "name": prec["precursor_name"],
                                     "score": 0.0, "evidence": []})
            cur["score"] += 1.5 * frac
            cur["basis"] = f"preceded {prec['matched']}/{prec['target_total']} past events"
            cur["evidence"] = prec.get("matched_events", [])[:4]
            candidates[c] = cur

    ranked = sorted(candidates.values(), key=lambda x: -x["score"])[:top]
    return {"input": wo_text, "equipment": normalize_tag(tags[0]) if tags else None,
            "observed_failure": observed, "causes": ranked,
            "note": "Decision support ranked from history; confirm before acting."}
