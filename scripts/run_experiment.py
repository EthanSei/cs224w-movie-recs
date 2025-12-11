"""Run recall evaluation with and without LLM reranking.

This script loads prediction JSON files and computes:
1. GNN-only recall@10: Top 10 from GNN's top 50
2. LLM-reranked recall@10: Top 10 after LLM reranking of top 50

Usage:
    python scripts/run_experiment.py
    python scripts/run_experiment.py --models gat hgt  # Specific models only
    python scripts/run_experiment.py --skip-rerank  # Skip LLM reranking (GNN only)
"""

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

from recommender.reranker.ai_client import TogetherAIClient
from recommender.reranker.schemas import MovieCandidate, UserContext

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

EXPERIMENTS_DIR = Path("experiments")
DEFAULT_MODELS = ["gat", "hgt", "lightgcn", "two_tower"]


def compute_recall_at_k(
    recommended_items: List[int],
    ground_truth_items: List[int],
    k: int = 10
) -> float:
    """Compute recall@k for a single user."""
    if not ground_truth_items:
        return 0.0
    
    top_k = set(recommended_items[:k])
    ground_truth = set(ground_truth_items)
    
    num_relevant_in_topk = len(top_k & ground_truth)
    return num_relevant_in_topk / len(ground_truth)


def get_gnn_top_k(candidates: List[dict], k: int = 10) -> List[int]:
    """Get top-k movie IDs from GNN candidates (already ranked by gnn_rank)."""
    # Sort by gnn_rank just to be safe
    sorted_candidates = sorted(candidates, key=lambda x: x['gnn_rank'])
    return [c['movie_id'] for c in sorted_candidates[:k]]


def rerank_with_llm(
    candidates: List[dict],
    user_context: dict,
    user_id: int,
    client: TogetherAIClient,
    k: int = 10
) -> List[int]:
    """Rerank candidates using LLM and return top-k movie IDs."""
    # Convert to schema objects
    movie_candidates = [
        MovieCandidate(
            movie_id=c['movie_id'],
            title=c['title'],
            genres=c['genres'],
            year=c['year'],
            gnn_score=c['gnn_score'],
            gnn_rank=c['gnn_rank']
        )
        for c in candidates
    ]
    
    context = UserContext(
        user_id=user_id,
        liked_movies=user_context.get('liked_movies', []),
        disliked_movies=user_context.get('disliked_movies', [])
    )
    
    try:
        response = client.rerank(movie_candidates, context, top_k=k)
        # Sort by rank and return movie IDs
        sorted_recs = sorted(response.recommendations, key=lambda x: x.rank)
        return [r.movie_id for r in sorted_recs[:k]]
    except Exception as e:
        logger.error(f"Reranking failed for user {user_id}: {e}")
        # Fallback to GNN order
        return get_gnn_top_k(candidates, k)


def evaluate_model(
    predictions_path: Path,
    client: Optional[TogetherAIClient] = None,
    skip_rerank: bool = False,
    k: int = 10
) -> Dict[str, float]:
    """Evaluate a single model's predictions."""
    with open(predictions_path, 'r') as f:
        data = json.load(f)
    
    model_name = data['model']
    users = data['users']
    
    logger.info(f"Evaluating {model_name} on {len(users)} users...")
    
    gnn_recalls = []
    reranked_recalls = []
    
    for i, user_data in enumerate(users):
        user_id = user_data['user_id']
        ground_truth = user_data['ground_truth_items']
        candidates = user_data['top_50_candidates']
        user_context = user_data['user_context']
        
        # GNN-only recall
        gnn_top_k = get_gnn_top_k(candidates, k)
        gnn_recall = compute_recall_at_k(gnn_top_k, ground_truth, k)
        gnn_recalls.append(gnn_recall)
        
        # LLM-reranked recall (if enabled)
        if not skip_rerank and client is not None:
            if (i + 1) % 10 == 0:
                logger.info(f"  Reranking user {i + 1}/{len(users)}...")
            
            reranked_top_k = rerank_with_llm(
                candidates, user_context, user_id, client, k
            )
            reranked_recall = compute_recall_at_k(reranked_top_k, ground_truth, k)
            reranked_recalls.append(reranked_recall)
    
    results = {
        f"gnn_recall@{k}": sum(gnn_recalls) / len(gnn_recalls) if gnn_recalls else 0.0
    }
    
    if reranked_recalls:
        results[f"reranked_recall@{k}"] = sum(reranked_recalls) / len(reranked_recalls)
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Run experiment with GNN and LLM reranking")
    parser.add_argument(
        '--models', 
        nargs='+', 
        default=DEFAULT_MODELS,
        help='Models to evaluate (default: all)'
    )
    parser.add_argument(
        '--skip-rerank',
        action='store_true',
        help='Skip LLM reranking (compute GNN recall only)'
    )
    parser.add_argument(
        '--k',
        type=int,
        default=10,
        help='Number of recommendations to evaluate (default: 10)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='experiments/results.json',
        help='Output path for results JSON'
    )
    
    args = parser.parse_args()
    
    logger.info("="*60)
    logger.info("Running Experiment: GNN vs LLM-Reranked Recall")
    logger.info("="*60)
    logger.info(f"Models: {args.models}")
    logger.info(f"Recall@{args.k}")
    logger.info(f"Skip reranking: {args.skip_rerank}")
    logger.info("="*60)
    
    # Initialize LLM client if needed
    client = None
    if not args.skip_rerank:
        try:
            client = TogetherAIClient()
            logger.info(f"Initialized Together AI client with model: {client.model}")
        except ValueError as e:
            logger.error(f"Failed to initialize LLM client: {e}")
            logger.info("Falling back to GNN-only evaluation")
            args.skip_rerank = True
    
    # Process each model
    all_results = {}
    seed = None
    num_users = None
    
    for model_name in args.models:
        predictions_path = EXPERIMENTS_DIR / f"{model_name}_predictions.json"
        
        if not predictions_path.exists():
            logger.warning(f"Predictions file not found: {predictions_path}")
            continue
        
        # Get metadata from first file
        if seed is None:
            with open(predictions_path, 'r') as f:
                data = json.load(f)
                seed = data.get('seed', 42)
                num_users = data.get('num_users', len(data['users']))
        
        results = evaluate_model(
            predictions_path,
            client=client,
            skip_rerank=args.skip_rerank,
            k=args.k
        )
        
        all_results[model_name] = results
        
        logger.info(f"\n{model_name} Results:")
        for metric, value in results.items():
            logger.info(f"  {metric}: {value:.4f}")
    
    # Build final output
    output = {
        "seed": seed,
        "num_users": num_users,
        "k": args.k,
        "skip_rerank": args.skip_rerank,
        "results": all_results
    }
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    logger.info(f"\nResults saved to {output_path}")
    
    # Print summary table
    logger.info("\n" + "="*60)
    logger.info("SUMMARY")
    logger.info("="*60)
    logger.info(f"{'Model':<15} {'GNN Recall@' + str(args.k):<20} {'Reranked Recall@' + str(args.k):<20}")
    logger.info("-"*60)
    
    for model_name, results in all_results.items():
        gnn_recall = results.get(f"gnn_recall@{args.k}", 0.0)
        reranked_recall = results.get(f"reranked_recall@{args.k}", "-")
        if isinstance(reranked_recall, float):
            reranked_str = f"{reranked_recall:.4f}"
        else:
            reranked_str = reranked_recall
        logger.info(f"{model_name:<15} {gnn_recall:<20.4f} {reranked_str:<20}")
    
    logger.info("="*60)


if __name__ == "__main__":
    main()

