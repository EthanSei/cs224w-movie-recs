import hydra
import logging
import numpy as np
import os
import random
import torch
from omegaconf import DictConfig, OmegaConf
from torch_geometric.data import HeteroData
from recommender.utils.module_loader import load_module
from recommender.evaluation.evaluator import evaluate

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def run_training(cfg: DictConfig):
    # Set seed for reproducibility
    seed = cfg.get('seed', 42)
    _set_seed(seed)

    logger.info("Starting training on dataset: %s, model: %s, loss: %s, trainer: %s", cfg.data.name, cfg.model.name, cfg.loss.name, cfg.trainer.name)
    
    # Load data
    DataLoaderClass = load_module(cfg.data.module)
    data_loader = DataLoaderClass(**cfg.data.params)
    train_data, val_data, test_data = data_loader.get_train_val_test_data(cfg.data.options.force)
    logger.info(f"Loaded data: {train_data['user'].num_nodes} users, {train_data['item'].num_nodes} items")

    # Load model - construct params from config and data
    ModelClass = load_module(cfg.model.module)
    model_params = _construct_model_params(cfg.model, train_data)
    model = ModelClass(**model_params)

    # Load loss function
    LossClass = load_module(cfg.loss.module)
    loss_fn = LossClass(**cfg.loss.params)

    # Load trainer
    TrainerClass = load_module(cfg.trainer.module)
    trainer = TrainerClass(**cfg.trainer.params)

    # Train
    trained_model = trainer.fit(model, train_data, val_data, loss_fn=loss_fn, verbose=True)
    
    # Create directory structure and save model
    save_path = f"runs/{cfg.data.name}/{cfg.model.name}/{cfg.loss.name}/{cfg.trainer.name}.pth"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(trained_model.state_dict(), save_path)
    logger.info(f"Saved model to {save_path}")
    logger.info("Training completed.")

    
    # Run evaluation on test set if specified
    if cfg.evaluator.params.get('run_test', True):
        logger.info("Evaluating on test set...")
        evaluate(trained_model, test_data, k=cfg.evaluator.params.k, device=cfg.trainer.params.device)


def _set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def _construct_model_params(cfg_model: DictConfig, train_data: HeteroData) -> dict:
    in_dims = {
        node_type: train_data[node_type].x.shape[1] 
        for node_type in train_data.node_types
    }
    # Extract metadata and ensure proper tuple format (OmegaConf may convert tuples to lists)
    metadata_tuple = train_data.metadata()
    # Ensure edge types are tuples, not lists (HGTConv requires this)
    node_types = list(metadata_tuple[0])
    edge_types = [tuple(et) if isinstance(et, (list, tuple)) else et for et in metadata_tuple[1]]
    metadata = (node_types, edge_types)

    # Start with params from config, convert to regular dict
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
    run_training()