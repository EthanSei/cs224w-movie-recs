# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a CS224W movie recommendation system built using PyTorch Geometric. The project implements graph-based recommender models on the MovieLens dataset, specifically using a two-tower architecture for collaborative filtering.

## Development Commands

### Setup and Installation
```bash
make setup              # Install dependencies in editable mode
```

### Data Management
```bash
make load               # Download dev dataset and build graph
make load env=prod      # Download production dataset (larger)
make force_load         # Force re-download dataset
make force_load env=prod # Force re-download prod dataset
```

The data loading process downloads the MovieLens dataset, builds a heterogeneous graph with user and movie nodes connected by "rates" edges, and converts it to an undirected graph.

### Training
```bash
make train              # Run training with default Hydra config (TwoTower)
python scripts/train.py --config-name=config_gat  # Train GAT model
```

### Testing
```bash
pytest                  # Run tests
```

## Architecture

### Configuration System (Hydra)
The project uses Hydra for hierarchical configuration management:
- **Base config**: `configs/config.yaml` - orchestrates data, model, and trainer configs
- **Data configs**: `configs/data/` - defines data loader modules and parameters
- **Model configs**: `configs/model/` - defines model architectures and hyperparameters
- **Trainer configs**: `configs/trainer/` - defines training loop parameters

Configs use a module loading pattern: each config specifies a `module` path (e.g., `recommender.data.movielens:MovielensDataLoader`) and `params` that get passed to the class constructor.

### Module Loader Pattern
The `src/recommender/utils/module_loader.py` provides dynamic class loading using the pattern `module.path:ClassName`. This allows configs to specify any class without hardcoding imports in the training script.

### Data Pipeline
- **Data loader**: `src/recommender/data/movielens.py:MovielensDataLoader`
  - Supports `dev` and `prod` environments for different dataset sizes
  - `get_data()`: Returns heterogeneous graph with user/movie nodes and rates edges
  - `get_train_val_test_data()`: Returns 80/10/10 split using PyG's `RandomLinkSplit`
  - Builds bipartite graph structure with edge attributes (rating, timestamp)
  - Automatically converts to undirected graph with reverse edges

### Models
Located in `src/recommender/models/`:
- **TwoTower**: Separate embedding towers for users and items, outputs embeddings for dot product similarity
- **GAT**: Graph Attention Network using PyG's GATv2Conv for heterogeneous graphs
  - Applies multi-head attention over user-movie edges
  - Learns which interactions are most important
  - Uses message passing to propagate information through the graph
  - Requires GATTrainer that passes graph structure to the model

Models should follow PyTorch conventions and return embeddings that can be used by the trainer.

### Training
- **SimpleTrainer**: `src/recommender/training/simple_trainer.py`
  - Handles node feature initialization if nodes lack attributes (creates random features)
  - Uses BCEWithLogitsLoss by default for link prediction
  - Computes dot product of user/item embeddings for edge prediction
  - Tracks best model by training loss
  - Currently missing validation evaluation (TODO at line 64)
  - Works with TwoTower and other non-graph-aware models

- **GATTrainer**: `src/recommender/training/gat_trainer.py`
  - Extends SimpleTrainer for graph-aware models
  - Passes graph structure (edge_index_dict) to models during forward pass
  - Required for GAT, GCN, GraphSAGE, and other message-passing models
  - Same training loop as SimpleTrainer but with graph connectivity

Training scripts are in `scripts/`:
- `scripts/train.py`: Main Hydra-based training entry point
- `scripts/load_data.py`: Standalone data loading script

### Project Structure
```
src/recommender/
├── data/           # Data loaders and download utilities
├── models/         # Model architectures
├── training/       # Training loops and logic
└── utils/          # Helper utilities (module loader)
```

## Key Implementation Details

### Heterogeneous Graph Structure
The MovieLens graph uses PyTorch Geometric's HeteroData with:
- Node types: `user`, `movie`
- Edge types: `("user", "rates", "movie")` and reverse `("movie", "rev_rates", "user")`
- Edge attributes: `rating` (float32), `timestamp` (int64)

### Feature Handling
If nodes don't have features, SimpleTrainer automatically creates random features based on the model's expected input dimensions (extracted from `model.user_tower[0].in_features` and `model.item_tower[0].in_features`).

### Training Configuration
Training runs are organized by timestamp: `runs/YYYY-MM-DD/HH-MM-SS/` (configurable via Hydra)
