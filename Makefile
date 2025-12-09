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

# Train a model
#	make train                    - train with default model (from configs)
#	make train MODEL=gat          - train GAT model
#	make train MODEL=hgt          - train HGT model
#	make train ARGS="trainer.params.epochs=50" - pass additional hydra overrides
train:
ifdef MODEL
	python scripts/train.py model=$(MODEL) $(ARGS)
else
	python scripts/train.py $(ARGS)
endif

# Evaluate a pre-trained model, or train a new one with --train flag
#	make eval                     - eval default model (from configs)
#	make eval MODEL=gat           - eval GAT model
#	make eval MODEL=gat --train   - train then eval GAT model
#	make eval ARGS="evaluator.params.k=20" - pass additional hydra overrides
eval:
ifdef MODEL
	python scripts/evaluate.py model=$(MODEL) $(ARGS)
else
	python scripts/evaluate.py $(ARGS)
endif

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