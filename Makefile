.PHONY: setup load load_1m load_32m force_load force_load_1m force_load_32m train eval recs tune setup_test test

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

load_1m:
	python scripts/load_data.py --env 1m

load_32m:
	python scripts/load_data.py --env 32m

force_load:
	python scripts/load_data.py --env $(env) --force

force_load_1m:
	python scripts/load_data.py --env 1m --force

force_load_32m:
	python scripts/load_data.py --env 32m --force

# Train a model
#	make train                    - train with default model (from configs)
#	make train MODEL=gat          - train GAT model
#	make train MODEL=hgt          - train HGT model
#	make train DATA=movielens_1m  - use MovieLens 1M dataset
#	make train DATA=movielens_32m - use MovieLens 32M dataset
#	make train ARGS="trainer.params.epochs=50" - pass additional hydra overrides

# Build the command dynamically based on which variables are set
TRAIN_CMD = python scripts/train.py
ifdef MODEL
TRAIN_CMD += model=$(MODEL)
endif
ifdef DATA
TRAIN_CMD += data=$(DATA)
endif

train:
	$(TRAIN_CMD) $(ARGS)

# Evaluate a pre-trained model, or train a new one with --train flag
#	make eval                     - eval default model (from configs)
#	make eval MODEL=gat           - eval GAT model
#	make eval DATA=movielens_1m   - eval on MovieLens 1M dataset
#	make eval ARGS="evaluator.params.k=20" - pass additional hydra overrides

EVAL_CMD = python scripts/evaluate.py
ifdef MODEL
EVAL_CMD += model=$(MODEL)
endif
ifdef DATA
EVAL_CMD += data=$(DATA)
endif

eval:
	$(EVAL_CMD) $(ARGS)

# Get detailed recommendations with movie titles (for LLM reranking)
#	make recs                     - get recommendations for all test users
#	make recs ARGS="+max_users=100 +output=recs.csv" - limit to 100 users, save to recs.csv
#	make recs ARGS="+k=20"        - get top-20 recommendations
#	make recs MODEL=gat           - get recommendations from GAT model
#	make recs DATA=movielens_1m   - get recs from MovieLens 1M dataset

RECS_CMD = python scripts/get_recommendations.py
ifdef MODEL
RECS_CMD += model=$(MODEL)
endif
ifdef DATA
RECS_CMD += data=$(DATA)
endif

recs:
	$(RECS_CMD) $(ARGS)

# Hyperparameter tuning with Optuna sweeper
#	make tune                     - tune default model with default sweeper
#	make tune MODEL=gat           - tune GAT model with GAT-specific sweeper
#	make tune MODEL=hgt           - tune HGT model with HGT-specific sweeper
#	make tune ARGS="hydra.sweeper.n_trials=10" - quick tune with 10 trials

ifdef MODEL
TUNE_CMD = python scripts/tune.py model=$(MODEL) +sweeper=$(MODEL)
else
TUNE_CMD = python scripts/tune.py +sweeper=default
endif

# Hyperparameter tuning with Optuna sweeper
#	make tune                     - tune default model with default sweeper
#	make tune MODEL=gat           - tune GAT model with GAT-specific sweeper
#	make tune MODEL=hgt           - tune HGT model with HGT-specific sweeper
#	make tune ARGS="hydra.sweeper.n_trials=10" - quick tune with 10 trials
tune:
ifdef MODEL
	python scripts/tune.py model=$(MODEL) +sweeper=$(MODEL) $(ARGS) --multirun
else
	python scripts/tune.py +sweeper=default $(ARGS) --multirun
endif

setup_test:
	pip install --upgrade pip
	pip install -e .[test]

test:
	pytest tests -q --disable-warnings --maxfail=1 --cov=src/recommender