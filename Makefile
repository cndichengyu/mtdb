PYTHON ?= .venv/bin/python

.PHONY: help install test lint debug standard eval clean

help:
	@echo "install  - create venv and install requirements"
	@echo "test     - run pytest"
	@echo "lint     - ruff real-error check"
	@echo "debug    - 1-minute debug experiment"
	@echo "standard - full standard experiment (CPU ~4 min)"
	@echo "eval     - evaluate the standard checkpoint"

install:
	uv venv .venv --python 3.14
	uv pip install --python $(PYTHON) -r requirements.txt

test:
	$(PYTHON) -m pytest tests -q

lint:
	$(PYTHON) -m ruff check mtdb scripts tests --select F401,F841,F821,E9

debug:
	$(PYTHON) scripts/train.py --config configs/debug.yaml

standard:
	$(PYTHON) scripts/train.py --config configs/standard.yaml --run-name demo_standard

eval:
	$(PYTHON) scripts/eval.py --checkpoint artifacts/demo_standard/checkpoint.pt

clean:
	rm -rf artifacts/demo_* artifacts/ci_smoke __pycache__ .pytest_cache .ruff_cache
