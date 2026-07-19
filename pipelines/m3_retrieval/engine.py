"""Ask orchestrator: route -> retrieve -> synthesize a cited Answer.

`baseline=True` forces a vector-only path that bypasses the graph entirely — this is the
apples-to-apples comparison the eval harness and the demo use to show the graph's value.
"""

from __future__ import annotations

import time
from typing import Optional

from pipelines.m3_retrieval import answer as answer_mod
from pipelines.m3_retrieval import retrieve, router
from pipelines.m3_retrieval.schemas import Answer


def ask(question: str, repos, llm=None, mode: Optional[str] = None,
        baseline: bool = False, graph_only: bool = False) -> Answer:
    t0 = time.time()

    if baseline:
        # vector-only: no router, no graph, no PPR — the comparison condition
        result = retrieve.lookup(question, repos, k=6)
        result.mode = "lookup"
    else:
        mode = mode or router.classify(question, repos.graph, llm)
        if mode == "global":
            result = retrieve.global_query(question, repos, llm, k=4)
        elif mode == "multihop" or graph_only:
            result = retrieve.multihop(question, repos, k=6, graph_only=graph_only)
        else:
            result = retrieve.lookup(question, repos, k=6)

    ans = answer_mod.synthesize(question, result, repos, llm)
    ans.latency_ms = int((time.time() - t0) * 1000)
    return ans
