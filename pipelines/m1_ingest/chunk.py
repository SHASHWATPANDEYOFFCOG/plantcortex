"""Chunk text with page anchors so every fact keeps a clickable citation."""

from __future__ import annotations

import re


def chunk_pages(doc_id: str, pages: list[str], target_chars: int = 700
                ) -> list[dict]:
    """Split page text into paragraph-ish chunks carrying (doc_id, page) provenance."""
    chunks: list[dict] = []
    n = 0
    for pageno, text in enumerate(pages, start=1):
        text = (text or "").strip()
        if not text:
            continue
        for para in _split(text, target_chars):
            n += 1
            chunks.append({"chunk_id": f"{doc_id}::p{pageno}::c{n}",
                           "doc_id": doc_id, "page": pageno, "text": para})
    return chunks


def _split(text: str, target: int) -> list[str]:
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    out: list[str] = []
    buf = ""
    for p in paras:
        if len(buf) + len(p) + 1 <= target:
            buf = f"{buf}\n{p}".strip()
        else:
            if buf:
                out.append(buf)
            if len(p) <= target:
                buf = p
            else:  # hard-wrap an over-long paragraph
                for i in range(0, len(p), target):
                    out.append(p[i:i + target])
                buf = ""
    if buf:
        out.append(buf)
    return out
