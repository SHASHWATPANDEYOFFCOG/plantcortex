"""Ingest the seed corpus into the knowledge graph + vector store.

Examples
--------
  python -m scripts.ingest_corpus --fresh                     # full, with LLM
  python -m scripts.ingest_corpus --fresh --no-llm            # offline, rule-based
  python -m scripts.ingest_corpus --fresh --only work_order,voice_note
"""

from __future__ import annotations

import argparse
import json
import logging

from core.config import settings
from core.llm import get_llm
from pipelines.m1_ingest.pipeline import ingest_corpus, make_repos

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fresh", action="store_true", help="rebuild from empty stores")
    ap.add_argument("--no-llm", action="store_true", help="deterministic only")
    ap.add_argument("--only", default="", help="comma-separated doc_types")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    repos = make_repos(fresh=args.fresh)
    llm = None if args.no_llm else get_llm()
    only = set(t.strip() for t in args.only.split(",") if t.strip()) or None

    def emit(ev: dict) -> None:
        print(f"  + {ev['doc_id']:<14} [{ev['doc_type']}]  "
              f"+{ev['nodes_added']}n/+{ev['edges_added']}e  "
              f"graph={ev['graph']['nodes']}n/{ev['graph']['edges']}e")

    print(f"Ingesting corpus (llm={'on' if llm else 'off'}) ...")
    ingest_corpus(repos, settings.seed_dir, llm=llm, emit=emit,
                  only_types=only, limit=args.limit)

    summary = repos.graph.summary()
    print("\n=== GRAPH SUMMARY ===")
    print(json.dumps(summary, indent=2))
    link = summary["linkage"]
    print(f"\nLinkage completeness: {link['well_linked']}/{link['equipment']} "
          f"equipment linked to >=3 doc types = {link['pct']}%")
    print(f"Vector store: {len(repos.vector)} chunks")
    if llm:
        print(f"LLM calls: {llm.stats.calls} (cache hits {llm.stats.cache_hits}, "
              f"~{llm.stats.total_tokens} tokens)")


if __name__ == "__main__":
    main()
