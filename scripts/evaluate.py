import hydra
import logging
import numpy as np
import os
import random
import sys
import torch
from omegaconf import DictConfig, OmegaConf
from torch_geometric.data import HeteroData
from recommender.utils.module_loader import load_module
from recommender.utils.device import clear_memory

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def run_evaluation(cfg: DictConfig):
    # Check for --train flag before hydra processes arguments
    should_train = "--train" in sys.argv
    
    # Set seed for reproducibility
    seed = cfg.get('seed', 42)
    _set_seed(seed)

    logger.info("Starting evaluation on dataset: %s, model: %s, loss: %s, trainer: %s", 
                cfg.data.name, cfg.model.name, cfg.loss.name, cfg.trainer.name)
    
    # Load data
    DataLoaderClass = load_module(cfg.data.module)
    data_loader = DataLoaderClass(**cfg.data.params)
    train_data, val_data, test_data = data_loader.get_train_val_test_data(cfg.data.options.force)
    logger.info(f"Loaded data: {train_data['user'].num_nodes} users, {train_data['item'].num_nodes} items")
    
    # Load or train model
    try:
        if should_train:
            logger.info("--train flag provided, training new model...")
            model = _train_new_model(cfg, train_data, val_data)
        else:
            model = _load_pretrained_model(cfg, train_data)
    except FileNotFoundError as e:
        logger.warning(f"{e}")
        logger.info("Training new model instead...")
        model = _train_new_model(cfg, train_data, val_data)
    
    EvaluatorClass = load_module(cfg.evaluator.module)
    evaluator = EvaluatorClass(**cfg.evaluator.params)
    
    logger.info(f"Evaluating on test data with k={evaluator.k}")
    hr, ndcg, recall = evaluator.evaluate(model, test_data, train_data=train_data, val_data=val_data)
    
    logger.info("=" * 50)
    logger.info("Final Evaluation Results:")
    logger.info(f"  Hit@{evaluator.k}: {hr:.4f}")
    logger.info(f"  NDCG@{evaluator.k}: {ndcg:.4f}")
    logger.info(f"  Recall@{evaluator.k}: {recall:.4f}")
    logger.info("=" * 50)
    logger.info("Evaluation completed.")

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

def _load_pretrained_model(cfg: DictConfig, train_data: HeteroData) -> torch.nn.Module:
    """Load a pre-trained model from the standard path."""
    save_path = f"runs/{cfg.data.name}/{cfg.model.name}/{cfg.loss.name}/{cfg.trainer.name}.pth"
    
    if not os.path.exists(save_path):
        raise FileNotFoundError(f"Pre-trained model not found at {save_path}")
    
    logger.info(f"Loading pre-trained model from {save_path}")
    
    # Load model architecture
    ModelClass = load_module(cfg.model.module)
    model_params = _construct_model_params(cfg.model, train_data)
    model = ModelClass(**model_params)
    
    # Load weights
    model.load_state_dict(torch.load(save_path, map_location=torch.device("cpu")))
    logger.info("Successfully loaded pre-trained model")
    
    return model

def _train_new_model(cfg: DictConfig, train_data: HeteroData, val_data: HeteroData) -> torch.nn.Module:
    """Train a new model using the provided config."""
    logger.info("Training new model...")
    
    # Load model
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
    
    # Save model
    save_path = f"runs/{cfg.data.name}/{cfg.model.name}/{cfg.loss.name}/{cfg.trainer.name}.pth"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(trained_model.state_dict(), save_path)
    logger.info(f"Saved model to {save_path}")
    
    # Clean up trainer to free memory before evaluation
    del trainer, loss_fn
    clear_memory()
    
    return trained_model

if __name__ == "__main__":
    run_evaluation()

