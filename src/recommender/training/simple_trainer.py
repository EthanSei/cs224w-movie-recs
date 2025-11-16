import copy
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
    ):
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.device = torch.device(device)

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
            optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        if loss_fn is None:
            from recommender.losses.bpr import BPRLoss
            loss_fn = BPRLoss(reduction="mean")
            logger.info("Using BPR loss with mean reduction (default for recommendation models)")

        # Prepare graph data - expect features to exist in data.x
        train_x_dict = {node_type: train_data[node_type].x.to(device) for node_type in train_data.node_types}
        train_edge_index_dict = {edge_type: train_data[edge_type].edge_index.to(device) for edge_type in train_data.edge_types}
        
        val_x_dict = {node_type: val_data[node_type].x.to(device) for node_type in val_data.node_types}
        val_edge_index_dict = {edge_type: val_data[edge_type].edge_index.to(device) for edge_type in val_data.edge_types}

        # Training loop
        train_losses = []
        best_loss = float("inf")
        best_model_state = None

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
            loss = self._compute_bpr_loss(
                user_emb, item_emb, train_data[("user", "rates", "item")].edge_label_index, 
                loss_fn, device, model
            )

            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

            if verbose and (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch+1}/{self.num_epochs}, Train loss: {loss.item():.4f}")

            # Validation
            model.eval()
            with torch.no_grad():
                val_z_dict = model.encode(val_x_dict, val_edge_index_dict)
                val_user_emb = val_z_dict.get("user")
                val_item_emb = val_z_dict.get("item")
                
                val_loss = self._compute_bpr_loss(
                    val_user_emb, val_item_emb, val_data[("user", "rates", "item")].edge_label_index,
                    loss_fn, device, model
                ).item()
                
                if val_loss < best_loss:
                    best_loss = val_loss
                    best_model_state = copy.deepcopy(model.state_dict())

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
        edge_label_index: torch.Tensor,
        loss_fn: torch.nn.Module,
        device: torch.device,
        model: torch.nn.Module,
    ) -> torch.Tensor:
        """Compute BPR loss with negative sampling using model's score() method."""
        edge_label_index = edge_label_index.to(device)
        pos_user_emb = user_emb[edge_label_index[0]]
        pos_item_emb = item_emb[edge_label_index[1]]
        pos_scores = model.score(pos_user_emb, pos_item_emb)
        
        # Sample negative items
        num_negatives = edge_label_index.size(1)
        neg_items = torch.randint(0, item_emb.size(0), (num_negatives,), device=device)
        neg_user_emb = user_emb[edge_label_index[0]]
        neg_item_emb = item_emb[neg_items]
        neg_scores = model.score(neg_user_emb, neg_item_emb)
        
        return loss_fn(pos_scores, neg_scores)
