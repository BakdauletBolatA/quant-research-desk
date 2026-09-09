PY := .venv/bin/python

.PHONY: help venv install data pipeline test lint clean all

help:
	@echo "make install   - create venv and install dependencies"
	@echo "make data      - download & cache market + factor data"
	@echo "make pipeline  - run the full research pipeline (reports/)"
	@echo "make test      - run the test suite"
	@echo "make all       - install + data + pipeline + test"

venv:
	python3 -m venv .venv

install: venv
	$(PY) -m pip install --upgrade pip -q
	$(PY) -m pip install -e ".[dev]" -q

data:
	$(PY) -m quantdesk.cli data

pipeline:
	$(PY) -m quantdesk.cli run

test:
	$(PY) -m pytest

lint:
	$(PY) -m ruff check src tests

clean:
	rm -rf reports/figures/*.png reports/*.html reports/*.xlsx data/processed/*
	find . -name '__pycache__' -type d -exec rm -rf {} +

all: install data pipeline test
