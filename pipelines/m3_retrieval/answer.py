"""Answer synthesis. Every answer carries citations; weak support -> graceful refusal.

Extractive by default (offline, grounded); an LLM composes prose grounded in the SAME
evidence when available. The two never diverge on facts because both are handed the
identical evidence set.
"""

from __future__ import annotations

import re

from core.config import ROOT
from pipelines.m3_retrieval.schemas import (
    Answer, Citation, PathEdge, REFUSAL_TEXT,
)

_STOP = set("the a an of to in on for and or is are was were be with at by from as "
           "what which who how why when where did do does has have had this that "
           "any all it its plant unit".split())


def _tokens(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9\-]+", text.lower())
            if t not in _STOP and len(t) > 2}


def _short_quote(text: str, question: str, max_words: int = 15) -> str:
    sents = re.split(r"(?<=[.\n])\s+", text.strip())
    qtok = _tokens(question)
    best, best_ov = text, -1
    for s in sents:
        ov = len(_tokens(s) & qtok)
        if ov > best_ov and s.strip():
            best, best_ov = s.strip(), ov
    words = best.split()
    return " ".join(words[:max_words]) + ("…" if len(words) > max_words else "")


def _support(question: str, chunks: list[dict]) -> int:
    qtok = _tokens(question)
    return max((len(_tokens(c.get("text", "")) & qtok) for c in chunks), default=0)


def _citations(chunks: list[dict], limit: int, question: str) -> list[Citation]:
    cites = []
    for c in chunks[:limit]:
        meta = c.get("meta", {})
        cites.append(Citation(
            doc_id=meta.get("doc_id") or c.get("chunk_id", "?").split("::")[0],
            page=meta.get("page"), row=meta.get("row"), bbox=meta.get("bbox"),
            quote=_short_quote(c.get("text", ""), question), chunk_id=c.get("chunk_id")))
    return cites


def _label(key: str) -> str:
    return key.split(":", 1)[-1] if ":" in key else key


def _path_md(path) -> str:
    if not path:
        return ""
    parts = [_label(path[0][0])]
    for src, edge, tgt in path:
        parts.append(f"—[{edge}]→ {_label(tgt)}")
    return " ".join(parts)


def synthesize(question: str, retrieval, repos, llm=None) -> Answer:
    from pipelines.m3_retrieval.retrieve import link_entities

    chunks = retrieval.chunks
    mode = retrieval.mode
    # Global sensemaking is answered from community summaries, so it is exempt from the
    # lexical-support refusal (it legitimately mentions no specific entity).
    is_global = mode == "global" and retrieval.communities
    has_entity = bool(retrieval.seeds) or bool(link_entities(question, repos.graph))
    if not is_global and (not chunks or (_support(question, chunks) < 2
                                         and not has_entity)):
        return Answer(answer_markdown=REFUSAL_TEXT, mode="refusal", confidence=0.1,
                      citations=_citations(chunks, 2, question))

    citations = _citations(chunks, 4, question)
    path = [PathEdge(source=_label(s), edge=e, target=_label(t))
            for s, e, t in retrieval.path]
    conf = min(0.95, 0.45 + 0.12 * len(citations)
               + (0.15 if retrieval.path else 0.0))

    llm_live = llm is not None and not getattr(llm, "quota_blocked", False)
    if llm_live:
        md = _llm_answer(question, retrieval, llm)
    else:
        md = _extractive_answer(question, retrieval)

    return Answer(answer_markdown=md, citations=citations, confidence=round(conf, 2),
                  mode=mode, path=path, seeds=retrieval.seeds, llm_used=llm_live)


def _extractive_answer(question: str, retrieval) -> str:
    lines: list[str] = []
    if retrieval.mode == "multihop":
        if retrieval.path:
            lines.append(f"**Graph path:** {_path_md(retrieval.path)}\n")
        # surface the entities the graph reasoning reached (the actual answer nodes),
        # not just supporting text — this is what a vector-only baseline can't do.
        related = [_label(k) for k, _ in retrieval.ppr
                   if k not in retrieval.seeds][:6]
        if related:
            lines.append(f"**Related in the graph:** {', '.join(related)}\n")
    if retrieval.mode == "global" and retrieval.communities:
        lines.append("**Corpus-wide clusters:**")
        for c in retrieval.communities[:3]:
            lines.append(f"- ({c['size']} assets) {c['summary']}")
        lines.append("")
    lines.append("**Supporting evidence:**")
    for i, c in enumerate(retrieval.chunks[:4], 1):
        meta = c.get("meta", {})
        loc = meta.get("doc_id", "?")
        if meta.get("page"):
            loc += f" p.{meta['page']}"
        snippet = re.sub(r"\s+", " ", c.get("text", "")).strip()[:180]
        lines.append(f"{i}. {snippet} — *[{loc}]* [E{i}]")
    return "\n".join(lines)


def _llm_answer(question: str, retrieval, llm) -> str:
    evidence = []
    for i, c in enumerate(retrieval.chunks[:5], 1):
        meta = c.get("meta", {})
        loc = meta.get("doc_id", "?") + (f" p.{meta['page']}" if meta.get("page") else "")
        clean = re.sub(r"\s+", " ", c.get("text", "")).strip()[:400]
        evidence.append(f"E{i} [{loc}]: {clean}")
    if retrieval.mode == "global":
        for c in retrieval.communities[:4]:
            evidence.append(f"CLUSTER ({c['size']}): {c['summary']}")
    tmpl = (ROOT / "prompts" / "answer.md").read_text(encoding="utf-8")
    prompt = (tmpl.replace("{question}", question)
              .replace("{path}", _path_md(retrieval.path) or "n/a")
              .replace("{evidence}", "\n".join(evidence)))
    res = llm.complete_json("", prompt, max_tokens=1024)
    md = (res or {}).get("answer")
    return md or _extractive_answer(question, retrieval)
