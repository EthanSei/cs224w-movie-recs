.PHONY: setup load reload train test venv

env ?= dev
VENV = CS224W-PROJECT

venv:
	python3 -m venv $(VENV)
	$(VENV)/bin/pip install --upgrade pip
	$(VENV)/bin/pip install -e .[test]
	@echo "Activate with: source $(VENV)/bin/activate"

setup:
	pip install --upgrade pip
	pip install -e .[dev]

load:
	python scripts/load_data.py --env $(env)

force_load:
	python scripts/load_data.py --env $(env) --force

train:
	python scripts/train.py

setup_test:
	pip install --upgrade pip
	pip install -e .[test]

test:
	pytest tests -q --disable-warnings --maxfail=1 --cov=src/recommender