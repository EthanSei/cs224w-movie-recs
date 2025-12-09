.PHONY: setup load force_load train eval tune setup_test test

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

# Modify model for training by setting MODEL variable, 
#	make train MODEL=lightgcn (default is gat)
train:
	python scripts/train.py

# Evaluate a pre-trained model, or train a new one with --train flag
#	make evaluate MODEL=gat (default is gat)
#	make evaluate MODEL=gat --train (to train a new model)
eval:
	python scripts/evaluate.py $(ARGS)

# Hyperparameter tuning with Optuna sweeper
#	make tune                                    - tune config with default sweeper
#	make tune ARGS="model=gat +sweeper=gat"      - tune GAT with GAT-specific sweeper
#	make tune ARGS="+sweeper=default hydra.sweeper.n_trials=10" - quick tune with 10 trials
tune:
	python scripts/tune.py +sweeper=default $(ARGS) --multirun

setup_test:
	pip install --upgrade pip
	pip install -e .[test]

test:
	pytest tests -q --disable-warnings --maxfail=1 --cov=src/recommender