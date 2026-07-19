# PlantCortex — Unified Asset & Operations Brain

[![CI](https://github.com/SHASHWATPANDEYOFFCOG/plantcortex/actions/workflows/ci.yml/badge.svg)](https://github.com/SHASHWATPANDEYOFFCOG/plantcortex/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/tests-82%20passing-brightgreen.svg)](tests/)
[![Version](https://img.shields.io/badge/version-1.0.0-informational.svg)](https://github.com/SHASHWATPANDEYOFFCOG/plantcortex/releases)

> ET AI Hackathon 2026 · Problem Statement 8 (Industrial Knowledge Intelligence)

PlantCortex ingests heterogeneous plant documents — P&ID drawings, maintenance work
orders, SOPs, inspection reports, incident/near-miss reports, and regulatory standards —
and fuses them into a **multimodal knowledge graph** that is queryable, actionable, and
continuously updated. It closes the loop **from pixels to prediction**.

## Research grounding (why this architecture)

| Thread | We use it for | Reference |
|---|---|---|
| Diagram-to-graph perception | Hybrid P&ID digitization (structure + VLM tag reading, never raw VLM) | PID2Graph / Relationformer, Stürmer et al. arXiv:2411.13929 |
| Hippocampal graph memory | Multi-hop retrieval via graph + Personalized PageRank | HippoRAG, Gutiérrez et al. NeurIPS 2024, arXiv:2405.14831 |
| GraphRAG community summaries | Corpus-wide "pattern" questions | Edge et al. arXiv:2404.16130 |
| Technical Language Processing | Normalizing jargon-dense work-order shorthand | NIST Nestor / MaintNorm |
| Ontology-guided FMEA reasoning | Failure-cause ranking on the typed graph | Okazaki et al. arXiv:2510.15428 |

## Novelties

- **N1 — Cross-modal entity resolution.** `P-101A` seen as a P&ID symbol, a spreadsheet
  row, an SOP mention, and a compliance clause resolves to ONE graph node.
- **N2 — Hippocampal retrieval over a multimodal graph.** PPR seeds can come from a
  *photographed equipment tag*, not just typed text.
- **N3 — From answering to anticipating.** The ontology-typed graph ranks probable
  failure causes and surfaces lessons-learned warnings.
- **Knowledge Capture** — a retiring operator's voice note becomes a first-class,
  citable "tacit knowledge" source. *"The engineer retires. The knowledge doesn't."*

## Problem-statement coverage (all five suggested build areas)

| Suggested build area | PlantCortex module | Where |
|---|---|---|
| Ingestion + Knowledge Graph | M1 universal ingestion → multimodal KG (+ M2 P&ID vision) | `pipelines/m1_ingest`, `m2_pnid`, `core/graph_repo` |
| Expert Copilot / RAG | M3 hybrid retrieval (BM25+dense+PPR+GraphRAG) with cited answers | `pipelines/m3_retrieval`, `/ask` |
| Maintenance / RCA Agent | M6 cause ranking — probable causes for a new work order | `agents/m6_patterns.rank_causes`, `/patterns/cause` |
| Compliance Intelligence | M5 clause-vs-procedure gap scan + evidence PDF (OISD / Factories Act) | `agents/m5_compliance`, `/compliance/*` |
| Lessons Learned Engine | M6 pattern cards + learned causal edges + M7 tacit capture | `agents/m6_patterns`, `/patterns`, `/capture` |

And the three pains named in the problem statement: **document fragmentation** → N1 one-node-per-asset
across six formats; **knowledge loss** → M7 Knowledge Capture; **downtime from incomplete
equipment history** → the field dossier + M6 precursor warnings.

## Status — all milestones complete

| Milestone | Scope | State |
|---|---|---|
| **D1** | Repo + ontology + seed corpus generator | ✅ |
| **D2** | M1 ingestion → graph, websocket, linkage metric | ✅ |
| **D3** | M2 P&ID hybrid vision pipeline | ✅ |
| **D4** | M3 router + PPR + communities + desktop UI | ✅ |
| **D5** | Mobile `/field`, compliance agent, knowledge capture | ✅ |
| **D6** | Pattern/cause engine, eval harness, demo mode | ✅ |

Graph after full ingest: **~1,304 nodes / 2,267 edges**, **100% linkage completeness**,
7/7 P&ID `CONNECTED_TO` process lines recovered from pixels. **82 tests pass.**

## Capabilities

- **M1 Ingestion** — route → TLP-normalize → chunk → embed/index → ontology-constrained
  extraction → cross-modal resolution (N1) → idempotent MERGE upsert, streaming
  `graph.delta` over a websocket.
- **M2 P&ID vision** — hybrid: OpenCV reads process connectivity from the pixels, the
  drawing's text/vector layer (or a VLM) supplies tag identities; each fact keeps a
  drawing bbox for citation. Enriches equipment nodes with `type` + location.
- **M3 Retrieval** — `POST /ask` routes to LOOKUP (hybrid BM25+dense RRF), MULTIHOP
  (Personalized PageRank, HippoRAG-style, N2), or GLOBAL (GraphRAG community summaries).
  Every answer carries clickable citations, a confidence chip, and the graph reasoning
  path. `baseline=true` / `graph_only=true` expose the apples-to-apples comparison.
- **M4 Surfaces** — desktop 3-pane copilot at `/` (chat · source viewer with bbox
  highlight · live graph); thumb-first mobile `/field` (tag lookup dossier, voice, compliance).
- **M5 Compliance** — `POST /compliance/scan` verdicts every clause `covered|partial|GAP`
  (catches the seeded SOP-17 gas-testing gap, 0 false positives); `/compliance/report/{std}`
  exports an audit-ready evidence PDF.
- **M6 Failure Intelligence (N3)** — Pattern Cards ("P-101A: 4 seal failures in 5 yrs,
  vibration preceded them"), a causal miner that writes learned `HAS_CAUSE` edges, and
  cause ranking that anticipates the probable cause of a *new* work order.
- **M7 Knowledge Capture** — a spoken note becomes a citable `TacitNote` in seconds.

## Evaluation (`make eval` → [eval/report.md](eval/report.md))

- **Extraction recall vs gold:** 77% overall — 100% on Equipment / FailureMode / Incident
  / Procedure / RegulatoryClause / WorkOrder / TacitNote (Component & HAS_CAUSE come from
  P&ID component extraction / the M6 miner — honestly reported).
- **QA benchmark (30 Q, 3 tiers × 3 conditions):** hybrid **matches** vector-only on
  simple lookups (T1) and is **clearly ahead on multi-hop (T2: 46% vs 38%) and global
  (T3: 75% vs 63%)** — overall 57% vs 50%.
- **Time-to-answer:** minutes of manual folder search → **sub-second**.
- **Compliance:** seeded gap caught **1/1**, **0** false positives.
- **Linkage completeness:** **100%** (8/8 equipment linked to ≥3 doc types).

## Quantified impact (illustrative economics, assumptions stated)

| Lever | Assumption (industry-typical) | Annual value, one unit |
|---|---|---|
| Engineer search time | 8 engineers × ~45 min/day finding cross-document answers → sub-second cited answers reclaim ~70% | **~1,000 engineer-hours/yr** |
| Avoided seal-failure trip | 1 unplanned charge-pump trip ≈ 4–12 h lost throughput (₹15–50 lakh for a mid-size unit); vibration-precursor warning converts run-to-failure into planned work | **1 avoided trip pays for deployment** |
| Audit preparation | Compliance evidence assembly ~2–3 person-days per standard → one-click evidence PDF | **days → hours per audit cycle** |
| Knowledge retention | Each retiring senior operator carries undocumented failure heuristics; capture cost ≈ 5 min/note | **priceless is not a number — but re-learning P-101A's seal habit the hard way is ~₹15–50 lakh (see row 2)** |

These are stated as *illustrative*, not measured — the honest claim is the mechanism
(cited answers in <1s, gaps caught pre-audit, precursors surfaced), demonstrated live.

## Quickstart (no Docker required)

This machine has no Docker, so PlantCortex runs on its **embedded fallbacks** (sanctioned
by the spec): a NetworkX-persisted graph behind a `GraphRepo` interface and a local
numpy + BM25 vector store — no external services to fail during a live demo.
`docker-compose.yml` (Neo4j + Qdrant) is provided as an optional path.

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -e .[dev]   # Windows path shown
.venv/Scripts/python -m scripts.seed_demo        # generate + ingest + mine + stage reserve
.venv/Scripts/python -m uvicorn api.main:app     # http://localhost:8000  (·  /field mobile)
.venv/Scripts/python -m pytest                   # 82 tests
.venv/Scripts/python -m eval.run                 # regenerate eval/report.md
```

The demo plant is **"Prayag Petro Refinery — Unit 2"**, a fictional but internally
consistent corpus where the same equipment tags thread through every document type.
Set `DEMO_STRICT=1` in `.env` to serve cache/offline only.

## Demo script (7 beats)

1. **Live ingest** — drag `data/seed/reserve/reserve_work_orders.xlsx` onto the app; the
   graph grows live (websocket, new P-102A node).
2. **Multi-hop on mobile** — ask the T2 "seal failures + overdue inspection + near-miss"
   question; click citations open to the exact page.
3. **Baseline vs hybrid** — toggle `baseline`; it degrades to a flat lookup, no path.
4. **Compliance** — scan OISD-STD-105 → SOP-17 gap → export the evidence PDF.
5. **Knowledge capture** — record a voice note; ask a follow-up; it's cited immediately.
6. **Out-of-corpus** — ask about tyre pressure → graceful refusal.
7. **Failure intelligence** — the ⚡ chip: P-101A pattern card + N3 cause anticipation.

## Repository layout

```
core/        ontology (Pydantic), resolver, graph_repo, vector_repo, embeddings, llm, config
pipelines/   m1_ingest/  m2_pnid/  m3_retrieval/
agents/      m5_compliance, m6_patterns, dossier
api/         FastAPI app (ingest, ask, compliance, patterns, capture, ws) + serves web/
web/         self-contained UI: index.html (desktop 3-pane), field.html (mobile)
prompts/     extract, pnid_read, tlp, router, answer
scripts/     generate_seed_corpus, ingest_corpus, seed_demo
eval/        run.py + benchmark_qa.jsonl → report.md
data/seed/   the demo corpus (+ regulatory/, gold/, reserve/)
```

## Honest limitations

- **LLM answer synthesis / VLM P&ID reading / voice ASR are wired but currently run on
  deterministic fallbacks** because the provided Gemini free-tier key is quota-exhausted.
  Retrieval, ranking, citations, compliance, and pattern mining are fully functional
  offline; LLM prose and real VLM tag-reading layer on automatically when quota returns.
- **Component / `HAS_CAUSE` / `PART_OF`** are not extracted by the offline text pass;
  they come from P&ID component detection and the M6 causal miner (reported in the eval).
- **QA accuracy is scored by a deterministic keyword proxy**, not an LLM judge — the
  relative (hybrid vs baseline) story is robust; absolute numbers rise with LLM synthesis.
- The UI is a **self-contained no-build web app**, not the specified Next.js/Tailwind
  stack — chosen for demo reliability and headless verifiability.

## Roadmap

- **Connectors:** SAP-PM / IBM Maximo work-order sync; OSISoft PI historian for real
  vibration trends behind the precursor analysis.
- **Data sovereignty:** on-prem OSS-LLM (Llama/Qwen) + local embeddings so no plant data
  leaves site — the interfaces (`LLMClient`, `Embeddings`) already abstract the provider.
- **Multilingual field UI** (Hindi + regional languages) for shop-floor technicians.
- **Scale:** swap `GraphRepo`→Neo4j and `VectorStore`→Qdrant via the existing interfaces
  (the `docker-compose.yml` path) with no caller changes.

See [ARCHITECTURE.md](ARCHITECTURE.md) for the system diagram and data flow.

## Screenshots

| Desktop copilot (`/`) | Field mobile (`/field`) |
|---|---|
| _screenshot coming soon_ | _screenshot coming soon_ |

> Demo walkthrough script: [docs/DEMO_VIDEO.md](docs/DEMO_VIDEO.md) · judge Q&A prep: [docs/JUDGE_QA.md](docs/JUDGE_QA.md)

## Contributing

Contributions are welcome — see [CONTRIBUTING.md](CONTRIBUTING.md) for the dev
setup, test workflow, and commit conventions. Please also read the
[Code of Conduct](CODE_OF_CONDUCT.md).

## License

Released under the [MIT License](LICENSE) © 2026 Shashwat Kumar Pandey.

## Contact

**Shashwat Kumar Pandey** — [@SHASHWATPANDEYOFFCOG](https://github.com/SHASHWATPANDEYOFFCOG) · shashwatpandeyoffcog3039@gmail.com
