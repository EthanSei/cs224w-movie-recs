# cs224w-movie-recs
Repo for the CS224w Movie Recommender Project

## Setup & Installation
1. Clone the repo: `git clone https://github.com/EthanSei/cs224w-movie-recs.git`
2. Install dependencies: `make setup`

## ML Pipeline

All commands support `MODEL=<gat|hgt|lightgcn|two_tower>` to select a model.

```bash
# 1. Load data
make load                  # Dev dataset (default)
make load_1m               # MovieLens 1M dataset
make load_32m              # MovieLens 32M dataset

# 2. Tune hyperparameters (optional)
make tune MODEL=gat

# 3. Train model
make train MODEL=gat

# 4. Evaluate model
make eval MODEL=gat

# 5. Generate recommendations
make recs MODEL=gat
```

## LLM Reranking Experiment

Generate predictions and compare GNN-only vs LLM-reranked recall:

```bash
# Generate top-50 predictions for all models
python scripts/generate_predictions.py

# Run experiment (compares GNN recall@10 vs LLM-reranked recall@10)
python scripts/run_experiment.py
python scripts/run_experiment.py --skip-rerank  # GNN only, no LLM
```

## Testing

```bash
make setup_test # Install test dependencies
make test # Run tests in tests/ directory
```

## Configuration

Configuration files are in `configs/`. The project uses [Hydra](https://hydra.cc/) - pass additional overrides via `ARGS="key=value"`.
