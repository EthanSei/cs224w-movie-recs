# MovieLens Datasets

This project supports three sizes of the MovieLens dataset:

## Dataset Sizes

| Dataset | Ratings | Users | Movies | Download Size | Training Time* |
|---------|---------|-------|--------|---------------|----------------|
| **Small (100k)** | 100,000 | 610 | 9,742 | ~1 MB | ~2-3 min |
| **1M** | 1,000,000 | 6,040 | 3,706 | ~6 MB | ~10-15 min |
| **32M (Latest)** | 32,000,000+ | 280,000+ | 58,000+ | ~265 MB | ~2-3 hours |

*Approximate training time for 100 epochs on GPU

## Quick Start

### Using Small Dataset (default)
```bash
make load                          # Download small dataset
make train MODEL=lightgcn          # Train LightGCN
make recs ARGS="+max_users=100"    # Get recommendations
```

### Using 1M Dataset
```bash
make load_1m                                    # Download 1M dataset
make train MODEL=lightgcn DATA=movielens_1m     # Train on 1M
make recs DATA=movielens_1m ARGS="+max_users=100"
```

### Using 32M Dataset
```bash
make load_32m                                   # Download 32M dataset
make train MODEL=lightgcn DATA=movielens_32m    # Train on 32M
make recs DATA=movielens_32m ARGS="+max_users=100"
```

## Dataset Selection Guide

### Use **Small (100k)** if:
- ✅ You're developing/debugging code
- ✅ You want fast iteration cycles
- ✅ You have limited time (< 2 days)
- ✅ You want to test multiple models quickly
- ✅ Your focus is on the LLM reranking methodology (not raw scale)

### Use **1M** if:
- ✅ You want more users/ratings for robust evaluation
- ✅ You have 2-3 days for experiments
- ✅ You want to show your method works at medium scale
- ✅ You have GPU access

### Use **32M** if:
- ✅ You want to demonstrate scalability
- ✅ You have 1+ week for experiments
- ✅ You have powerful GPU hardware
- ⚠️ **NOT recommended for < 3 days remaining**

## Commands Reference

### Data Loading
```bash
# Small dataset
make load              # or make load env=dev

# 1M dataset
make load_1m           # or make load env=1m

# 32M dataset
make load_32m          # or make load env=32m

# Force reload (clear cache)
make force_load_1m
make force_load_32m
```

### Training
```bash
# Train on specific dataset
make train MODEL=lightgcn DATA=movielens_1m
make train MODEL=gat DATA=movielens_32m

# Train with custom config
make train MODEL=lightgcn DATA=movielens_1m ARGS="trainer.params.epochs=200"
```

### Evaluation
```bash
# Eval on specific dataset
make eval MODEL=lightgcn DATA=movielens_1m

# Get recommendations
make recs MODEL=lightgcn DATA=movielens_1m ARGS="+max_users=100 +k=20"
```

## File Storage

Datasets are cached after first download:
- Raw data: `data/raw/ml-latest-small.zip`, `data/raw/ml-1m.zip`, `data/raw/ml-latest.zip`
- Processed data: `data/processed/movielens_small.pt`, `data/processed/movielens_1m.pt`, `data/processed/movielens_32m.pt`
- Models: `runs/{dataset}/{model}/{loss}/{trainer}.pth`

Each dataset has separate model checkpoints, so you can train on multiple datasets without conflicts.

## Recommendation for Your Project

With **36 hours** left and **LLM reranking** as your novel contribution:

**Option 1: Conservative (Recommended)**
- Use **Small (100k)** only
- Train 3 models (LightGCN, GAT, Two-Tower) - skip HGT
- Focus 80% of time on LLM reranking quality
- Show depth over breadth

**Option 2: Balanced**
- Use **Small** for development (first 12 hours)
- Train on **1M** for final results (last 24 hours)
- 2 models only (LightGCN, GAT)
- Show your method scales to medium data

**Option 3: Risky (Not Recommended)**
- Use **32M** dataset
- Long training times will eat into LLM reranking time
- High risk of running out of time
