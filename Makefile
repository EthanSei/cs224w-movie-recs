.PHONY: setup load reload train test

env ?= dev

setup:
	pip install --upgrade pip
	pip install -e .[dev]

load:
	python scripts/load_data.py --env $(env)

force_load:
	python scripts/load_data.py --env $(env) --force

# Modify model for training by setting MODEL variable, 
#	make train MODEL=lightgcn (default is gat)
train:
	python scripts/train.py model=$(MODEL)

setup_test:
	pip install --upgrade pip
	pip install -e .[test]

test:
	pytest tests -q --disable-warnings --maxfail=1 --cov=src/recommender