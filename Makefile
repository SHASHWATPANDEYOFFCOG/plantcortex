# PlantCortex Makefile.
# NOTE: `make` is often absent on Windows. Every target below has a plain-Python
# equivalent shown in the comment; run those directly in PowerShell if needed.

PY ?= .venv/Scripts/python.exe      # on macOS/Linux: .venv/bin/python

.PHONY: help venv install seed test eval demo serve clean

help:
	@echo "Targets: venv install seed test eval demo serve clean"

venv:                     ## python -m venv .venv
	python -m venv .venv

install:                  ## $(PY) -m pip install -e .[dev]
	$(PY) -m pip install -e .[dev]

seed:                     ## $(PY) -m scripts.generate_seed_corpus
	$(PY) -m scripts.generate_seed_corpus

test:                     ## $(PY) -m pytest
	$(PY) -m pytest

eval:                     ## $(PY) -m eval.run
	$(PY) -m eval.run

demo:                     ## seed corpus + ingest + mine + stage reserve files
	$(PY) -m scripts.seed_demo

serve:                    ## run the API + UI (desktop at /, mobile at /field)
	$(PY) -m uvicorn api.main:app --port 8000

clean:
	rm -rf data/graph data/vectors data/cache
