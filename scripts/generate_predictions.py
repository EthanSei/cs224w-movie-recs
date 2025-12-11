"""Generate top-50 predictions for 100 sampled users across all models.

This script loads pre-trained models and generates prediction JSON files
for use in the experiment pipeline comparing GNN-only vs LLM-reranked recall.

Usage:
    python scripts/generate_predictions.py
    python scripts/generate_predictions.py seed=123  # Override seed
"""

import hydra
import json
import logging
import numpy as np
import os
import pandas as pd
import random
import re
import torch
from collections import defaultdict
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
from torch_geometric.data import HeteroData
from typing import Dict, List, Optional, Tuple

from recommender.data.download_data import download_data
from recommender.utils.module_loader import load_module

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Models to evaluate
MODELS = ["gat", "hgt", "lightgcn", "two_tower"]
NUM_USERS = 100
TOP_K = 50


def _set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _resolve_device(device: str) -> str:
    """Resolve 'auto' device to actual device string."""
    if device == "auto":
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    return device


def _construct_model_params(cfg_model: DictConfig, train_data: HeteroData) -> dict:
    """Construct model parameters from config and data."""
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


def _load_pretrained_model(
    cfg: DictConfig, 
    model_name: str, 
    train_data: HeteroData
) -> torch.nn.Module:
    """Load a pre-trained model from the standard path."""
    
    # Load model config first to get the proper model name (may differ in casing)
    model_cfg = OmegaConf.load(f"configs/model/{model_name}.yaml")
    config_model_name = model_cfg.name  # e.g., "LightGCN" instead of "lightgcn"
    
    save_path = f"runs/{cfg.data.name}/{config_model_name}/{cfg.loss.name}/{cfg.trainer.name}.pth"
    
    if not os.path.exists(save_path):
        raise FileNotFoundError(f"Pre-trained model not found at {save_path}")
    
    logger.info(f"Loading pre-trained model from {save_path}")
    
    ModelClass = load_module(model_cfg.module)
    model_params = _construct_model_params(model_cfg, train_data)
    model = ModelClass(**model_params)
    
    model.load_state_dict(torch.load(save_path, map_location=torch.device("cpu")))
    logger.info(f"Successfully loaded pre-trained model: {model_name}")
    
    return model


def _compute_scores_for_users(
    model: torch.nn.Module, 
    train_data: HeteroData,
    user_ids: List[int],
    device: str = "cpu"
) -> Dict[int, torch.Tensor]:
    """Compute scores only for specified users (optimized - skips unused users)."""
    model.eval()
    model.to(device)
    
    with torch.no_grad():
        # Encode all nodes (required for GNN message passing)
        z_dict = model.encode(
            {ntype: train_data[ntype].x.to(device) for ntype in train_data.node_types},
            {etype: train_data[etype].edge_index.to(device) for etype in train_data.edge_types}
        )
        user_emb = z_dict["user"]
        item_emb = z_dict["item"]
    
    num_items = item_emb.size(0)
    user_scores = {}
    
    # Only compute scores for the users we need
    for user_id in user_ids:
        user_vec = user_emb[user_id:user_id+1]  # (1, hidden_dim)
        user_expanded = user_vec.expand(num_items, -1)  # (num_items, hidden_dim)
        
        scores = model.score(user_expanded, item_emb)  # (num_items,)
        user_scores[user_id] = scores.cpu()
    
    return user_scores


def _get_user_item_sets(
    train_data: HeteroData,
    val_data: HeteroData,
    test_data: HeteroData
) -> Tuple[Dict[int, set], Dict[int, set], Dict[int, set]]:
    """Extract user-to-item sets for train, val, and test data."""
    
    # Train items (from edge_index, not edge_label_index)
    user_to_train_items = defaultdict(set)
    train_edge_index = train_data[("user", "rates", "item")].edge_index

    for u, i in train_edge_index.T:

        user_to_train_items[u.item()].add(i.item())
    
    # Validation items (positive only from edge_label_index)
    user_to_val_items = defaultdict(set)

    if hasattr(val_data[("user", "rates", "item")], "edge_label_index"):
        val_edge_store = val_data[("user", "rates", "item")]
        val_edge_label_index = val_edge_store.edge_label_index

        if hasattr(val_edge_store, "edge_label"):
            val_pos_mask = val_edge_store.edge_label == 1
            val_edge_label_index = val_edge_label_index[:, val_pos_mask]

        for u, i in val_edge_label_index.T:
            user_to_val_items[u.item()].add(i.item())
    
    # Test items (positive only from edge_label_index)
    user_to_test_items = defaultdict(set)
    edge_store = test_data[("user", "rates", "item")]
    edge_label_index = edge_store.edge_label_index

    if hasattr(edge_store, "edge_label"):
        pos_mask = edge_store.edge_label == 1
        edge_label_index = edge_label_index[:, pos_mask]

    for u, i in edge_label_index.T:
        user_to_test_items[u.item()].add(i.item())
    
    return user_to_train_items, user_to_val_items, user_to_test_items


def _extract_year_from_title(title: str) -> Optional[int]:
    """Extract year from movie title like 'Movie Name (1999)'."""
    match = re.search(r'\((\d{4})\)', title)
    if match:
        try:
            return int(match.group(1))
        except (ValueError, TypeError):
            pass
    return None


def _build_movie_metadata(movies_df) -> Dict[int, dict]:
    """Build movie metadata dictionary from movies DataFrame."""
    movie_metadata = {}
    for _, row in movies_df.iterrows():
        movie_id = row['movieId']
        title = row['title']
        genres = row['genres'].split('|') if pd.notna(row['genres']) else []
        year = _extract_year_from_title(title)
        movie_metadata[movie_id] = {
            'title': title,
            'genres': genres,
            'year': year
        }
    return movie_metadata


def _build_user_context(
    user_id: int,
    train_items: set,
    ratings_df,
    movies_df,
    user_code_to_id: Dict[int, int],
    movie_id_index: List[int]
) -> dict:
    """Build user context with liked and disliked movies from training data only."""
    # Get original user ID
    original_user_id = user_code_to_id.get(user_id)
    if original_user_id is None:
        return {"liked_movies": [], "disliked_movies": []}
    
    # Map train_items (graph indices) back to movieIds to avoid data leakage
    train_movie_ids = {movie_id_index[idx] for idx in train_items if idx < len(movie_id_index)}
    
    # Get user's ratings FILTERED to training items only
    user_ratings = ratings_df[
        (ratings_df['userId'] == original_user_id) & 
        (ratings_df['movieId'].isin(train_movie_ids))
    ]
    
    # Merge with movies to get titles
    user_ratings_with_movies = user_ratings.merge(
        movies_df[['movieId', 'title']], 
        on='movieId', 
        how='left'
    )
    
    # Liked movies (rating >= 4)
    liked = user_ratings_with_movies[user_ratings_with_movies['rating'] >= 4.0]
    liked_movies = liked['title'].dropna().tolist()[:20]  # Limit to 20
    
    # Disliked movies (rating <= 2)
    disliked = user_ratings_with_movies[user_ratings_with_movies['rating'] <= 2.0]
    disliked_movies = disliked['title'].dropna().tolist()[:10]  # Limit to 10
    
    return {
        "liked_movies": liked_movies,
        "disliked_movies": disliked_movies
    }


def _get_top_k_candidates(
    user_scores: torch.Tensor,
    train_items: set,
    val_items: set,
    movie_id_index,
    movie_metadata: Dict[int, dict],
    k: int = 50
) -> List[dict]:
    """Get top-k candidate movies for a user, excluding train/val items."""
    user_scores = user_scores.clone()
    
    # Mask train items
    if train_items:
        train_items_tensor = torch.tensor(list(train_items), dtype=torch.long)
        user_scores[train_items_tensor] = float('-inf')
    
    # Mask validation items
    if val_items:
        val_items_tensor = torch.tensor(list(val_items), dtype=torch.long)
        user_scores[val_items_tensor] = float('-inf')
    
    # Get top-k
    topk_scores, topk_indices = torch.topk(user_scores, k=min(k, user_scores.size(0)))
    
    candidates = []
    for rank, (score, item_idx) in enumerate(zip(topk_scores.tolist(), topk_indices.tolist()), 1):
        # Map item index back to movie ID
        movie_id = movie_id_index[item_idx]
        metadata = movie_metadata.get(movie_id, {})
        
        candidates.append({
            "movie_id": int(movie_id),
            "title": metadata.get('title', f'Unknown Movie {movie_id}'),
            "genres": metadata.get('genres', []),
            "year": metadata.get('year'),
            "gnn_score": float(score),
            "gnn_rank": rank
        })
    
    return candidates


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def generate_predictions(cfg: DictConfig):
    """Generate predictions for all models."""
    seed = cfg.get('seed', 42)
    _set_seed(seed)
    
    logger.info(f"Generating predictions with seed={seed}, num_users={NUM_USERS}, top_k={TOP_K}")
    
    # Load raw data for metadata
    movies_df, tags_df, ratings_df = download_data("movielens", force=False, env=cfg.data.params.env)
    
    # Build movie metadata lookup
    movie_metadata = _build_movie_metadata(movies_df)
    
    # Build movie ID index (maps item index to movie ID)
    movie_id_index = movies_df['movieId'].drop_duplicates().tolist()
    
    # Build user code to original ID mapping
    user_codes, users = pd.factorize(ratings_df['userId'].to_numpy(), sort=True)
    user_code_to_id = {code: user_id for code, user_id in enumerate(users)}
    
    # Load data using the data loader
    DataLoaderClass = load_module(cfg.data.module)
    data_loader = DataLoaderClass(**cfg.data.params)
    train_data, val_data, test_data = data_loader.get_train_val_test_data(cfg.data.options.force)
    
    logger.info(f"Loaded data: {train_data['user'].num_nodes} users, {train_data['item'].num_nodes} items")
    
    # Get user-item sets
    user_to_train_items, user_to_val_items, user_to_test_items = _get_user_item_sets(
        train_data, val_data, test_data
    )
    
    # Sample 100 users from test set
    test_users = list(user_to_test_items.keys())
    if len(test_users) < NUM_USERS:
        logger.warning(f"Only {len(test_users)} users in test set, using all of them")
        sampled_users = test_users
    else:
        sampled_users = random.sample(test_users, NUM_USERS)
    
    logger.info(f"Sampled {len(sampled_users)} users from test set")
    
    # Create experiments directory
    experiments_dir = Path("experiments")
    experiments_dir.mkdir(exist_ok=True)
    
    # Process each model
    for model_name in MODELS:
        logger.info(f"\n{'='*50}")
        logger.info(f"Processing model: {model_name}")
        logger.info(f"{'='*50}")
        
        try:
            # Load model
            model = _load_pretrained_model(cfg, model_name, train_data)
            
            # Compute scores only for sampled users (optimized)
            logger.info(f"Computing scores for {len(sampled_users)} sampled users...")
            device = _resolve_device(cfg.evaluator.params.device)
            user_scores = _compute_scores_for_users(model, train_data, sampled_users, device=device)
            
            # Build predictions for each sampled user
            users_data = []
            for user_id in sampled_users:
                train_items = user_to_train_items.get(user_id, set())
                val_items = user_to_val_items.get(user_id, set())
                test_items = user_to_test_items.get(user_id, set())
                
                # Get top-50 candidates
                candidates = _get_top_k_candidates(
                    user_scores[user_id], train_items, val_items,
                    movie_id_index, movie_metadata, k=TOP_K
                )
                
                # Build user context (filtered to training items only)
                user_context = _build_user_context(
                    user_id, train_items, ratings_df, movies_df, user_code_to_id, movie_id_index
                )
                
                users_data.append({
                    "user_id": user_id,
                    "ground_truth_items": list(test_items),
                    "top_50_candidates": candidates,
                    "user_context": user_context
                })
            
            # Save to JSON
            output = {
                "model": model_name,
                "seed": seed,
                "num_users": len(sampled_users),
                "top_k": TOP_K,
                "users": users_data
            }
            
            output_path = experiments_dir / f"{model_name}_predictions.json"
            with open(output_path, 'w') as f:
                json.dump(output, f, indent=2)
            
            logger.info(f"Saved predictions to {output_path}")
            
        except FileNotFoundError as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            continue
    
    logger.info("\n" + "="*50)
    logger.info("Prediction generation complete!")
    logger.info("="*50)


if __name__ == "__main__":
    generate_predictions()

