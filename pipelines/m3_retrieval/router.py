"""Query router: LOOKUP vs MULTIHOP vs GLOBAL.

Heuristic by default (offline, deterministic); an LLM classifier is used when available.
The heuristic keys off corpus-wide language, multi-entity/joining language, and how many
graph entities the question mentions.
"""

from __future__ import annotations

import re

from core.config import ROOT
from pipelines.m1_ingest.extract import extract_tags, failure_codes

_GLOBAL = re.compile(
    r"\b(pattern|patterns|trend|trends|recur|recurr|across the|over the (last|past)|"
    r"overall|in general|common|most frequent|summar|sensemak|five years|5 years|"
    r"what.*(risks|failures).*(recur|common)|systemic)\b", re.I)

# Deliberately conservative: bare words like "require" belong to lookup clause
# questions, so only explicit joining/causal language triggers multihop here (a
# >=2-entity count also triggers it, checked separately).
_MULTIHOP = re.compile(
    r"\b(as well as|which .*\b(and|that)\b|related to|connected to|linked to|"
    r"root cause|caused by|why did|leads? to|both .* and|downstream|upstream|"
    r"preceded by|governed by)\b", re.I)


def _entity_count(question: str, graph) -> int:
    keys = set()
    for t in extract_tags(question):
        if graph.has_node(f"Equipment:{t}"):
            keys.add(t)
    for c in failure_codes(question):
        keys.add(c)
    for m in re.findall(r"\b(SOP-\d+|INC-\d{4}-\d+|OISD-[A-Z0-9-]+|WO-\d+)\b", question):
        keys.add(m)
    return len(keys)


def classify_heuristic(question: str, graph) -> str:
    if _GLOBAL.search(question):
        return "global"
    n = _entity_count(question, graph)
    if _MULTIHOP.search(question) or n >= 2:
        return "multihop"
    # "and" joining two clauses is a weak multihop signal
    if question.lower().count(" and ") >= 1 and n >= 1:
        return "multihop"
    return "lookup"


def classify(question: str, graph, llm=None) -> str:
    if llm is not None and not getattr(llm, "quota_blocked", False):
        prompt = (ROOT / "prompts" / "router.md").read_text(encoding="utf-8") \
            .replace("{question}", question)
        res = llm.complete_json("", prompt, max_tokens=64)
        mode = (res or {}).get("mode")
        if mode in ("lookup", "multihop", "global"):
            return mode
    return classify_heuristic(question, graph)
