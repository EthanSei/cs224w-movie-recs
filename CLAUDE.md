# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CS224W Movie Recommendation System using Graph Neural Networks (GNNs) and PyTorch Geometric. The project implements multiple recommendation models (GAT, HGT, Two-Tower) trained on the MovieLens dataset using Bayesian Personalized Ranking (BPR) loss.

## Development Commands

### Setup and Installation
```bash
make setup              # Install dependencies and set up development environment
make setup_test         # Install test dependencies only
```

### Data Loading
```bash
make load               # Download dataset and build graph (dev environment)
make load env=prod      # Load production dataset (much larger)
make force_load         # Force reload data even if cached
make force_load env=prod  # Force reload production dataset
```

### Training and Testing
```bash
make train              # Train model with default config (uses Hydra)
make test               # Run pytest with coverage (-q, --disable-warnings, --maxfail=1)
pytest tests/path/to/test.py  # Run specific test file
pytest tests/path/to/test.py::test_function  # Run single test
```

## Architecture

### Hydra Configuration System

The project uses **Hydra** for configuration management with a composable config structure:

- **Main config**: `configs/config.yaml` - defines defaults for data, model, loss, trainer
- **Config groups**:
  - `configs/data/` - dataset loaders (movielens.yaml)
  - `configs/model/` - model architectures (gat.yaml, hgt.yaml, two_tower.yaml)
  - `configs/loss/` - loss functions (bpr.yaml)
  - `configs/trainer/` - training configurations (default.yaml)

Training runs output to `runs/YYYY-MM-DD/HH-MM-SS/` directories with Hydra managing working directories via `hydra.run.dir`.

**Override configs at runtime**:
```bash
python scripts/train.py model=hgt trainer.params.num_epochs=200
```

### Module Loading Pattern

The codebase uses a **dynamic module loader** (`src/recommender/utils/module_loader.py`) that loads classes at runtime via config strings:

```yaml
module: "recommender.models.gat:GAT"  # format: "path.to.module:ClassName"
```

This enables `scripts/train.py` to instantiate models, data loaders, losses, and trainers without hardcoded imports. The loader uses `importlib` to dynamically import and return the class.

### Model Interface Contract

All recommendation models MUST implement this interface (enforced by `SimpleTrainer`):

1. **`encode(x_dict, edge_index_dict) -> dict[str, torch.Tensor]`**
   - Input: `x_dict` (node features by type), `edge_index_dict` (edges by type)
   - Output: dict with `'user'` and `'item'` embedding tensors
   - Used during training to get node embeddings

2. **`score(user_emb, item_emb) -> torch.Tensor`**
   - Input: user embeddings, item embeddings (batched)
   - Output: scalar scores for user-item pairs
   - All models use `WeightedDotProductHead` for scoring

3. **`forward(x_dict, edge_index_dict) -> dict[str, torch.Tensor]`**
   - Should call `encode()` for backward compatibility with tests

### Models

- **GAT** (`src/recommender/models/gat.py`): Graph Attention Network using GATv2Conv wrapped in HeteroConv for heterogeneous graphs
- **HGT** (`src/recommender/models/hgt.py`): Heterogeneous Graph Transformer using HGTConv for different node/edge types
- **TwoTower** (`src/recommender/models/two_tower.py`): Baseline model with separate user/item towers (ignores graph structure)

All models use:
- `TypeProjector` - projects different node types to same hidden dimension
- `WeightedDotProductHead` - computes weighted dot product scores (xW · y)

### Training Pipeline

**`scripts/train.py`** orchestrates the full pipeline:

1. Load config via Hydra (from `configs/`)
2. Instantiate data loader → get train/val/test splits via `RandomLinkSplit`
3. Build model with `_construct_model_params()` which injects `in_dims` and `metadata` from data
4. Instantiate loss function (BPR) and trainer (SimpleTrainer)
5. Call `trainer.fit()` → saves best model to `runs/{data}/{model}/{loss}/{trainer}.pth`

**SimpleTrainer** (`src/recommender/training/simple_trainer.py`):
- Uses BPR loss with negative sampling
- Tracks best validation loss, restores best model state
- Calls `model.encode()` to get embeddings, then `model.score()` for predictions
- Logs every 10 epochs

### Data Loading

**MovielensDataLoader** (`src/recommender/data/movielens.py`):
- Downloads MovieLens dataset via `download_data()` (from `src/recommender/data/download_data.py`)
- Builds heterogeneous graph with:
  - **User nodes**: random features (100-dim, placeholder for future user data)
  - **Item nodes**: multi-hot genre features + one-hot year bucket features (5-year buckets from 1900-2024)
  - **Edges**: `("user", "rates", "item")` with rating and timestamp attributes
- Converts to undirected graph with reverse edges `("item", "rev_rates", "user")`
- Splits into train/val/test (80/10/10) using `RandomLinkSplit`

### BPR Loss

**BPRLoss** (`src/recommender/losses/bpr.py`) implements Bayesian Personalized Ranking:
- Maximizes margin between positive and negative item scores
- Formula: `-log(sigmoid(pos_score - neg_score))`
- Negative items sampled randomly during training (in `SimpleTrainer._compute_bpr_loss`)

## Code Structure

```
src/recommender/
├── data/
│   ├── download_data.py    # Downloads MovieLens dataset
│   └── movielens.py         # Builds HeteroData graph from MovieLens
├── losses/
│   └── bpr.py               # Bayesian Personalized Ranking loss
├── models/
│   ├── gat.py               # GAT model
│   ├── hgt.py               # HGT model
│   ├── two_tower.py         # Two-tower baseline
│   └── model_helpers.py     # TypeProjector, WeightedDotProductHead
├── training/
│   └── simple_trainer.py    # Generic trainer with BPR + negative sampling
└── utils/
    └── module_loader.py     # Dynamic module loading from config strings

scripts/
├── load_data.py             # CLI for downloading and building graph
└── train.py                 # Main training script (Hydra entrypoint)

configs/
├── config.yaml              # Main config with defaults
├── data/movielens.yaml      # Dataset config
├── model/*.yaml             # Model configs (gat, hgt, two_tower)
├── loss/bpr.yaml            # Loss config
└── trainer/default.yaml     # Trainer config

tests/                       # Pytest tests with coverage
```

## Key Implementation Notes

### Model Construction

Models receive `in_dims` and `metadata` dynamically from the data during training:
- `in_dims`: dict mapping node type → feature dimension (e.g., `{"user": 100, "item": 44}`)
- `metadata`: tuple of `(node_types, edge_types)` extracted from `HeteroData`

This is handled by `_construct_model_params()` in `scripts/train.py` which merges config params with data-derived params.

### Edge Types Format

PyTorch Geometric edge types MUST be tuples: `("user", "rates", "item")`. Hydra may convert tuples to lists in configs, so `_construct_model_params()` ensures conversion back to tuples for HGTConv compatibility.

### Seed Management

Training uses fixed seed (default 42) for reproducibility, set via `_set_seed()` in `scripts/train.py` which configures random, numpy, torch, and CUDA seeds.
