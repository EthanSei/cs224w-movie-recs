.PHONY: setup load force_load train eval setup_test test

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
	python scripts/train.py

# Evaluate a pre-trained model, or train a new one with --train flag
#	make evaluate MODEL=gat (default is gat)
#	make evaluate MODEL=gat --train (to train a new model)
eval:
	python scripts/evaluate.py $(ARGS)

setup_test:
	pip install --upgrade pip
	pip install -e .[test]

test:
	pytest tests -q --disable-warnings --maxfail=1 --cov=src/recommender