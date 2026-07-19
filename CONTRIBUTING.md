# Contributing to PlantCortex

Thanks for your interest! This project started as an ET AI Hackathon 2026 build and
welcomes improvements — new connectors, better extraction, UI polish, docs.

## Dev setup

```bash
git clone https://github.com/SHASHWATPANDEYOFFCOG/plantcortex.git
cd plantcortex
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt -e .[dev]   # Windows; use .venv/bin/python on macOS/Linux
```

No Docker, database, or API key is required for development — the embedded
NetworkX graph and local vector store are the default backends, and the test
suite generates its own seed corpus.

An optional `GEMINI_API_KEY` in `.env` (copy `.env.example`) enables LLM answer
synthesis, VLM P&ID reading, and voice transcription. **Never commit `.env`.**

## Workflow

1. Fork and create a topic branch: `git checkout -b feat/my-change`
2. Make your change, keeping the style of the surrounding code.
3. Run the tests — they must all pass: `python -m pytest`
4. If your change affects retrieval/extraction quality, regenerate the eval
   report and mention any metric movement in your PR: `python -m eval.run`
5. Open a pull request using the template.

## Commit convention

We use [Conventional Commits](https://www.conventionalcommits.org/):

```
feat: add Maximo work-order connector
fix: handle rotated P&ID scans in geometric extractor
docs: clarify DEMO_STRICT behavior
test: cover multihop router fallback
chore: bump networkx pin
```

## Ground rules

- Every extracted fact must carry provenance (`doc_id`, page/bbox/row) — no
  uncited nodes or edges.
- The offline/deterministic path must keep working with zero network access
  (`DEMO_STRICT=1`); LLM features layer on top, they never become load-bearing.
- New node/edge types belong in `core/ontology.py`, nowhere else.
