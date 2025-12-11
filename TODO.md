# 1M pipeline

# 1. Loading
```bash
make load_1m
```

# 2. Training
```bash
# Train with all four models
make train MODEL=lightgcn DATA=movielens_1m
make train MODEL=hgt DATA=movielens_1m
make train MODEL=gat DATA=movielens_1m
make train MODEL=two_tower DATA=movielens_1m
```

# 3. Get Recommendations with Learned Model
```bash
# Get top 50 recommendations for 100 FIXED users
python scripts/generate_predictions.py data=movielens_1m
```

# 4. Compute Recall@10 for Base Model
```bash
make eval MODEL=lightgcn DATA=movielens_1m ARGS="evaluator.params.k=10 +evaluator.params.max_test_users=100 +evaluator.params.test_user_seed=42"
make eval MODEL=hgt DATA=movielens_1m ARGS="evaluator.params.k=10 +evaluator.params.max_test_users=100 +evaluator.params.test_user_seed=42"
make eval MODEL=gat DATA=movielens_1m ARGS="evaluator.params.k=10 +evaluator.params.max_test_users=100 +evaluator.params.test_user_seed=42"
make eval MODEL=twotower DATA=movielens_1m ARGS="evaluator.params.k=10 +evaluator.params.max_test_users=100 +evaluator.params.test_user_seed=42"
```

# Run 2-4 all at once:
```bash
make train MODEL=lightgcn DATA=movielens_1m && \
make train MODEL=gat DATA=movielens_1m && \
make train MODEL=hgt DATA=movielens_1m && \
make train MODEL=twotower DATA=movielens_1m && \
make recs MODEL=lightgcn DATA=movielens_1m ARGS="+k=50 +max_users=100 +seed=42 +output=lightgcn_recs.csv" && \
make recs MODEL=gat DATA=movielens_1m ARGS="+k=50 +max_users=100 +seed=42 +output=gat_recs.csv" && \
make recs MODEL=hgt DATA=movielens_1m ARGS="+k=50 +max_users=100 +seed=42 +output=hgt_recs.csv" && \
make recs MODEL=twotower DATA=movielens_1m ARGS="+k=50 +max_users=100 +seed=42 +output=tt_recs.csv" && \
make eval MODEL=lightgcn DATA=movielens_1m ARGS="+evaluator.params.max_test_users=100" && \
make eval MODEL=gat DATA=movielens_1m ARGS="+evaluator.params.max_test_users=100" && \
make eval MODEL=hgt DATA=movielens_1m ARGS="+evaluator.params.max_test_users=100" && \
make eval MODEL=two_tower DATA=movielens_1m ARGS="+evaluator.params.max_test_users=100"
```


# 5. LLM Reranking (Top 50 -> Top 10)
```bash
# some code
```

# 6. Computing ranked recall
```bash
# some code
```

# 7. Compare results!
```bash
# some code
```

