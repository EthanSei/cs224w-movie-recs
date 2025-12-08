import logging
import tqdm
import torch
import torch_geometric.data as HeteroData

# Setup logging
logger = logging.getLogger(__name__)

class SimpleTrainer:
    """
    Generic trainer for recommendation models.
    
    All models must implement encode(x_dict, edge_index_dict) -> dict[str, Tensor]
    that returns embeddings with 'user' and 'item' keys.
    """
    def __init__(
        self, 
        num_epochs: int,
        learning_rate: float = 1e-3,
        device: str = "cpu",
        patience: int = None,
        val_frequency: int = 1,
    ):
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.device = torch.device(device)
        self.patience = patience
        self.val_frequency = max(1, val_frequency)

    def fit(
        self,
        model: torch.nn.Module,
        train_data: HeteroData,
        val_data: HeteroData,
        optimizer: torch.optim.Optimizer = None,
        loss_fn: torch.nn.Module = None,
        verbose: bool = True,
    ) -> torch.nn.Module:
        """
        Train the model on the provided data.
        
        All models must implement encode(x_dict, edge_index_dict) -> dict[str, Tensor].
        """
        device = self.device
        model.to(device)

        if optimizer is None:
            optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        if loss_fn is None:
            from recommender.losses.bpr import BPRLoss
            loss_fn = BPRLoss(reduction="mean")
            logger.info("Using BPR loss with mean reduction (default for recommendation models)")

        # Prepare graph data - expect features to exist in data.x
        train_x_dict = {node_type: train_data[node_type].x.to(device) for node_type in train_data.node_types}
        train_edge_index_dict = {edge_type: train_data[edge_type].edge_index.to(device) for edge_type in train_data.edge_types}
        
        val_x_dict = {node_type: val_data[node_type].x.to(device) for node_type in val_data.node_types}
        val_edge_index_dict = {edge_type: val_data[edge_type].edge_index.to(device) for edge_type in val_data.edge_types}
        
        # Extract POSITIVE edges only for BPR loss
        # RandomLinkSplit includes both positives and sampled negatives in edge_label_index
        # edge_label indicates which are positive (1) vs negative (0)
        train_edge_store = train_data[("user", "rates", "item")]
        train_pos_edges = train_edge_store.edge_label_index
        if hasattr(train_edge_store, "edge_label"):
            train_pos_mask = train_edge_store.edge_label == 1
            train_pos_edges = train_pos_edges[:, train_pos_mask]
        train_pos_edges = train_pos_edges.to(device)
        
        val_edge_store = val_data[("user", "rates", "item")]
        val_pos_edges = val_edge_store.edge_label_index
        if hasattr(val_edge_store, "edge_label"):
            val_pos_mask = val_edge_store.edge_label == 1
            val_pos_edges = val_pos_edges[:, val_pos_mask]
        val_pos_edges = val_pos_edges.to(device)

        # Training loop
        best_loss = float("inf")
        best_model_state = None
        epochs_without_improvement = 0
        num_items = train_data["item"].x.size(0)

        for epoch in tqdm.tqdm(range(self.num_epochs), disable=not verbose):
            model.train()
            optimizer.zero_grad(set_to_none=True)

            # Get embeddings using unified interface
            z_dict = model.encode(train_x_dict, train_edge_index_dict)
            user_emb = z_dict.get("user")
            item_emb = z_dict.get("item")
            
            if user_emb is None or item_emb is None:
                raise ValueError("Model must output 'user' and 'item' embeddings")

            # Compute loss with BPR (requires negative sampling)
            # Use only positive edges - BPR will sample its own negatives
            loss = self._compute_bpr_loss(
                user_emb, item_emb, train_pos_edges, 
                num_items, loss_fn, model
            )

            loss.backward()
            optimizer.step()

            if verbose and (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch+1}/{self.num_epochs}, Train loss: {loss.item():.4f}")

            # Validation (only every val_frequency epochs)
            if (epoch + 1) % self.val_frequency == 0:
                model.eval()
                with torch.no_grad():
                    val_z_dict = model.encode(val_x_dict, val_edge_index_dict)
                    val_user_emb = val_z_dict.get("user")
                    val_item_emb = val_z_dict.get("item")
                    
                    val_loss = self._compute_bpr_loss(
                        val_user_emb, val_item_emb, val_pos_edges,
                        num_items, loss_fn, model
                    ).item()
                    
                    if val_loss < best_loss:
                        best_loss = val_loss
                        best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
                        epochs_without_improvement = 0
                    else:
                        epochs_without_improvement += 1
                    
                    # Early stopping check
                    if self.patience is not None and epochs_without_improvement >= self.patience:
                        if verbose:
                            logger.info(f"Early stopping at epoch {epoch+1} (no improvement for {self.patience} epochs)")
                        break

        # Load best model
        if best_model_state is not None:
            model.load_state_dict(best_model_state)

        if verbose:
            logger.info(f"Training completed. Best validation loss: {best_loss:.4f}")
        
        return model

    def _compute_bpr_loss(
        self,
        user_emb: torch.Tensor,
        item_emb: torch.Tensor,
        pos_edge_index: torch.Tensor,
        num_items: int,
        loss_fn: torch.nn.Module,
        model: torch.nn.Module,
    ) -> torch.Tensor:
        """Compute BPR loss with negative sampling using model's score() method.
        
        Args:
            pos_edge_index: Tensor of shape [2, num_pos_edges] containing only positive edges
            num_items: Total number of items for negative sampling
        """
        # Get user embeddings for positive edges (reused for negative scoring)
        pos_user_idx = pos_edge_index[0]
        pos_user_emb = user_emb[pos_user_idx]
        pos_item_emb = item_emb[pos_edge_index[1]]
        pos_scores = model.score(pos_user_emb, pos_item_emb)
        
        # Sample negative items randomly
        num_edges = pos_edge_index.size(1)
        neg_items = torch.randint(0, num_items, (num_edges,), device=pos_edge_index.device)
        neg_item_emb = item_emb[neg_items]
        # Reuse pos_user_emb instead of re-indexing
        neg_scores = model.score(pos_user_emb, neg_item_emb)
        
        return loss_fn(pos_scores, neg_scores)
