"""`make demo` — cold-start the full demo on a clean machine.

1. Generate the corpus if missing.
2. Ingest everything offline (deterministic; no quota needed), mine causal edges,
   pre-build community summaries.
3. Stage live-ingest RESERVE files (a fresh work-order sheet + a copy of the P&ID with
   its sidecar) so the demo can drag them in and watch the graph grow live.
4. Print how to launch the app.

Everything here runs without a network, so `DEMO_STRICT=1` (cache/offline only) works.
"""

from __future__ import annotations

import shutil
from datetime import date

import pandas as pd

from agents import m6_patterns
from core.config import settings
from pipelines.m1_ingest.pipeline import ingest_corpus, make_repos
from pipelines.m3_retrieval.communities import get_communities

RESERVE = settings.seed_dir / "reserve"


def stage_reserve() -> None:
    """Files intentionally NOT in the base ingest — dragged in live during the demo."""
    RESERVE.mkdir(parents=True, exist_ok=True)
    # a fresh batch of work orders incl. a brand-new asset (P-102A) so new nodes pop
    rows = [
        ("WO-9001", "2026-06-02", "P-101A", "corrective",
         "mech seel weep p101a, hi vib noted prior", "seel chng planned"),
        ("WO-9002", "2026-06-11", "P-102A", "corrective",
         "new charge pump p102a brg noise hi vib", "monitored"),
        ("WO-9003", "2026-06-20", "P-102A", "preventive",
         "routine pm p102a", "insp ok"),
        ("WO-9004", "2026-06-25", "E-301", "corrective",
         "channel gskt weep e301", "tightnd bolts"),
        ("WO-9005", "2026-07-01", "P-102A", "corrective",
         "mech seel lkg p102a seel failure", "chng mech seel"),
    ]
    df = pd.DataFrame(rows, columns=["WO_ID", "Date", "Equipment_Tag", "Type",
                                     "Problem_Text", "Action_Text"])
    df.to_excel(RESERVE / "reserve_work_orders.xlsx", index=False)
    # a P&ID copy (+ sidecar) so the drawing drag also works offline
    src = settings.seed_dir / "pnid"
    for f in ("PID-U2-001.png", "PID-U2-001.layer.json"):
        if (src / f).exists():
            shutil.copy(src / f, RESERVE / f.replace("001", "002"))
    print(f"  staged reserve files -> {RESERVE}")


def main() -> None:
    if not (settings.seed_dir / "manifest.json").exists():
        from scripts.generate_seed_corpus import main as gen
        print("Generating seed corpus…")
        gen()

    print("Ingesting corpus (offline)…")
    repos = make_repos(fresh=True)
    ingest_corpus(repos, settings.seed_dir, llm=None)
    mined = m6_patterns.mine_causal(repos.graph)
    repos.save()
    print(f"  mined {mined} causal (HAS_CAUSE) edges")
    print("Building community summaries…")
    get_communities(repos, None, rebuild=True)
    stage_reserve()

    s = repos.graph.summary()
    print(f"\nDemo ready: {s['nodes']} nodes / {s['edges']} edges, "
          f"linkage {s['linkage']['pct']}%.")
    print("Launch:  .venv/Scripts/python -m uvicorn api.main:app --port 8000")
    print("Open:    http://localhost:8000  (desktop)   ·   /field  (mobile)")
    print("Offline demo: set DEMO_STRICT=1 in .env to serve cache/offline only.")


if __name__ == "__main__":
    main()
