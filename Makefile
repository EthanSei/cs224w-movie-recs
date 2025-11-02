.PHONY: setup load reload train test

env ?= dev

setup:
	pip install -e .[dev]

load:
	python scripts/load_data.py --env $(env)

force_load:
	python scripts/load_data.py --env $(env) --force

train:
	python scripts/train.py

test:
	pytest