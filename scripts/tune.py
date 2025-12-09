"""
Hyperparameter tuning script using Hydra's Optuna sweeper.

Usage:
    # Basic tuning with default sweeper (for LightGCN/TwoTower)
    python scripts/tune.py +sweeper=default --multirun
    
    # Tune GAT model with GAT-specific sweeper
    python scripts/tune.py model=gat +sweeper=gat --multirun
    
    # Tune HGT model with HGT-specific sweeper
    python scripts/tune.py model=hgt +sweeper=hgt --multirun
    
    # Quick test with fewer trials
    python scripts/tune.py +sweeper=default hydra.sweeper.n_trials=5 --multirun
    
    # Save study to SQLite for persistence
    python scripts/tune.py +sweeper=default 'hydra.sweeper.storage=sqlite:///runs/optuna.db' --multirun

Output:
    Best parameters are logged at the end of the sweep.
    Use SQLite storage to persist studies across runs.
"""
import hydra
import logging
import numpy as np
import random
import torch
from omegaconf import DictConfig, OmegaConf
from torch_geometric.data import HeteroData

from recommender.utils.module_loader import load_module

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def tune(cfg: DictConfig) -> float:
    """
    Run a single training trial and return the metric to optimize.
    
    Returns:
        float: The optimization metric (Recall@k by default).
    """
    seed = cfg.get('seed', 42)
    _set_seed(seed)
    
    logger.info("=" * 60)
    logger.info("Starting tuning trial")
    logger.info(f"Model: {cfg.model.name}")
    logger.info(f"Model params: {OmegaConf.to_yaml(cfg.model.params)}")
    logger.info(f"Trainer params: {OmegaConf.to_yaml(cfg.trainer.params)}")
    logger.info("=" * 60)
    
    # Load data
    DataLoaderClass = load_module(cfg.data.module)
    data_loader = DataLoaderClass(**cfg.data.params)
    train_data, val_data, test_data = data_loader.get_train_val_test_data(cfg.data.options.force)
    logger.info(f"Loaded data: {train_data['user'].num_nodes} users, {train_data['item'].num_nodes} items")
    
    # Construct model
    ModelClass = load_module(cfg.model.module)
    model_params = _construct_model_params(cfg.model, train_data)
    model = ModelClass(**model_params)
    
    # Load loss function
    LossClass = load_module(cfg.loss.module)
    loss_fn = LossClass(**cfg.loss.params)
    
    # Load trainer
    TrainerClass = load_module(cfg.trainer.module)
    trainer = TrainerClass(**cfg.trainer.params)
    
    # Train model
    trained_model = trainer.fit(model, train_data, val_data, loss_fn=loss_fn, verbose=False)
    
    # Evaluate on validation set for hyperparameter selection
    EvaluatorClass = load_module(cfg.evaluator.module)
    evaluator = EvaluatorClass(**cfg.evaluator.params)
    
    # Use validation data for tuning (test data reserved for final evaluation)
    hr, ndcg, recall = evaluator.evaluate(trained_model, val_data, train_data=train_data)
    
    logger.info(f"Trial results - Hit@{evaluator.k}: {hr:.4f}, NDCG@{evaluator.k}: {ndcg:.4f}, Recall@{evaluator.k}: {recall:.4f}")
    
    # Return Recall as the optimization metric (Optuna will maximize this)
    return recall


def _set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _construct_model_params(cfg_model: DictConfig, train_data: HeteroData) -> dict:
    """Construct model parameters by merging config with data-derived values."""
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


if __name__ == "__main__":
    tune()

