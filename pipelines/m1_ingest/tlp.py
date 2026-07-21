"""Technical Language Processing: expand maintenance shorthand before extraction.

Maintenance work-order text is jargon-dense shorthand ("chng seel pmp101a", "brg noise
p101b hi vib"). Extractors and embeddings do much better on normalized English. This is
deterministic (lexicon-driven) so it runs offline and its before/after is demoable; an
optional LLM cleanup pass is available for the long tail.
"""

from __future__ import annotations

import csv
import re
from functools import lru_cache

from core.config import ROOT

LEXICON_PATH = ROOT / "data" / "tlp_lexicon.csv"
_WORD = re.compile(r"[A-Za-z0-9/][A-Za-z0-9/\-]*")


@lru_cache(maxsize=1)
def _lexicon() -> dict[str, str]:
    lex: dict[str, str] = {}
    with LEXICON_PATH.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            lex[row["abbrev"].strip().lower()] = row["expansion"].strip()
    return lex


@lru_cache(maxsize=1)
def _phrases() -> list[tuple[str, str]]:
    """Multi-word lexicon entries, longest first (applied before single tokens)."""
    items = [(k, v) for k, v in _lexicon().items() if " " in k]
    return sorted(items, key=lambda kv: -len(kv[0]))


def normalize(text: str) -> tuple[str, list[dict]]:
    """Return (normalized_text, changes). Equipment tags are preserved verbatim."""
    if not text:
        return text, []
    lex = _lexicon()
    changes: list[dict] = []

    # phrase-level first
    out = text
    for phrase, repl in _phrases():
        pat = re.compile(rf"\b{re.escape(phrase)}\b", re.IGNORECASE)
        if pat.search(out):
            out = pat.sub(repl, out)
            changes.append({"from": phrase, "to": repl})

    # token-level
    def _sub(m: re.Match) -> str:
        tok = m.group(0)
        low = tok.lower()
        if low in lex:
            repl = lex[low]
            if repl.lower() != low:
                changes.append({"from": tok, "to": repl})
            return repl
        return tok

    out = _WORD.sub(_sub, out)
    return out, changes


def normalize_with_llm(text: str, llm=None) -> tuple[str, list[dict]]:
    """Lexicon-normalize, then (if an LLM is available) a light grammar cleanup."""
    normalized, changes = normalize(text)
    if llm is None:
        return normalized, changes
    prompt_path = ROOT / "prompts" / "tlp.md"
    prompt = prompt_path.read_text(encoding="utf-8").replace("{raw}", normalized)
    res = llm.complete_json("", prompt, max_tokens=256)
    cleaned = (res or {}).get("normalized")
    if cleaned:
        return cleaned, changes
    return normalized, changes
