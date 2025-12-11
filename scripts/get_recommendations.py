"""
Script to get detailed recommendations with movie titles for each user.
Useful for inspecting recommendations and preparing data for LLM reranking.

Usage:
    python scripts/get_recommendations.py                                    # All test users
    python scripts/get_recommendations.py +max_users=100                     # Limit to 100 users
    python scripts/get_recommendations.py +k=20                              # Top-20 recommendations
    python scripts/get_recommendations.py +output=recs.csv                   # Save to CSV
    python scripts/get_recommendations.py model.params.hidden_dim=256        # Use specific model config
"""

# NOTE: OUTDATED, REPLACED BY scripts/generate_predictions.py

import hydra
import logging
import numpy as np
import os
import pandas as pd
import random
import torch
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
from recommender.utils.module_loader import load_module
from recommender.evaluation.detailed_evaluator import DetailedEvaluator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def _set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _construct_model_params(cfg_model: DictConfig, train_data) -> dict:
    in_dims = {
        node_type: train_data[node_type].x.shape[1]
        for node_type in train_data.node_types
    }
    metadata_tuple = train_data.metadata()
    node_types = list(metadata_tuple[0])
    edge_types = [tuple(et) if isinstance(et, (list, tuple)) else et for et in metadata_tuple[1]]
    metadata = (node_types, edge_types)

    if 'params' in cfg_model:
        model_params = OmegaConf.to_container(cfg_model.params, resolve=True)
        if model_params is None:
            model_params = {}
    else:
        model_params = {}

    if 'in_dims' in model_params:
        model_params['in_dims'] = in_dims

    if 'metadata' in model_params:
        model_params['metadata'] = metadata

    return model_params


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    # Get params from Hydra config (set via +param=value)
    max_users = cfg.get('max_users', None)
    k = cfg.get('k', 10)
    output = cfg.get('output', 'recommendations.csv')
    seed = cfg.get('seed', 42)

    # Set seed for reproducibility
    _set_seed(seed)

    logger.info("=" * 70)
    logger.info("Getting detailed recommendations")
    logger.info(f"Dataset: {cfg.data.name}, Model: {cfg.model.name}")
    logger.info(f"K: {k}, Max users: {max_users or 'all'}")
    logger.info("=" * 70)

    # Load data
    DataLoaderClass = load_module(cfg.data.module)
    data_loader = DataLoaderClass(**cfg.data.params)
    train_data, val_data, test_data = data_loader.get_train_val_test_data(cfg.data.options.force)
    logger.info(f"Loaded data: {train_data['user'].num_nodes} users, {train_data['item'].num_nodes} items")

    # Load model
    save_path = f"runs/{cfg.data.name}/{cfg.model.name}/{cfg.loss.name}/{cfg.trainer.name}.pth"
    if not os.path.exists(save_path):
        logger.error(f"Model not found at {save_path}")
        logger.error("Please train a model first using: make train")
        return

    logger.info(f"Loading model from {save_path}")
    ModelClass = load_module(cfg.model.module)
    model_params = _construct_model_params(cfg.model, train_data)
    model = ModelClass(**model_params)
    model.load_state_dict(torch.load(save_path, map_location=torch.device("cpu"), weights_only=True))
    logger.info("Successfully loaded model")

    # Create detailed evaluator
    evaluator = DetailedEvaluator(
        k=k,
        device="auto",
        max_test_users=max_users,
        test_user_seed=seed
    )

    # Get detailed recommendations
    logger.info("Generating recommendations...")
    hr, ndcg, recall, recommendations_df = evaluator.evaluate(
        model, test_data,
        train_data=train_data,
        val_data=val_data,
        return_details=True
    )

    # Print summary
    logger.info("=" * 70)
    logger.info("Evaluation Metrics:")
    logger.info(f"  Hit@{k}: {hr:.4f}")
    logger.info(f"  NDCG@{k}: {ndcg:.4f}")
    logger.info(f"  Recall@{k}: {recall:.4f}")
    logger.info("=" * 70)

    # Save to CSV
    output_path = Path(output)
    recommendations_df.to_csv(output_path, index=False)
    logger.info(f"Saved {len(recommendations_df)} recommendations to {output_path}")

    # Print sample recommendations for first 3 users
    logger.info("\nSample recommendations (first 3 users):")
    logger.info("=" * 70)
    for user_id in recommendations_df['user_id'].unique()[:3]:
        user_recs = recommendations_df[recommendations_df['user_id'] == user_id]
        logger.info(f"\nUser {user_id}:")
        for _, rec in user_recs.iterrows():
            relevance = "✓ RELEVANT" if rec['is_relevant'] else ""
            logger.info(f"  {rec['rank']:2d}. {rec['movie_title']:50s} (score: {rec['score']:.4f}) {relevance}")

    logger.info("\n" + "=" * 70)
    logger.info(f"Full recommendations saved to {output_path}")
    logger.info("You can now use this CSV for LLM reranking!")


if __name__ == "__main__":
    main()
