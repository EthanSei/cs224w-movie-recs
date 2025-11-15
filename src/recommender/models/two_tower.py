import torch
import torch.nn as nn
import torch.nn.functional as F

class TwoTower(nn.Module):
    """
    Two-tower model for recommendation systems.
    Implements encode() and forward() methods that take x_dict and edge_index_dict.
    Requires user_dim and item_dim to be specified at initialization.
    Supports variable number of layers for flexible architecture depth.
    """
    def __init__(
        self, 
        user_dim: int, 
        item_dim: int, 
        hidden_dim: int = 128, 
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        super(TwoTower, self).__init__()
        self.user_dim = user_dim
        self.item_dim = item_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Build towers in initialization
        self.user_tower = self._build_tower(user_dim, hidden_dim, num_layers, dropout)
        self.item_tower = self._build_tower(item_dim, hidden_dim, num_layers, dropout)
    
    def _build_tower(self, input_dim, hidden_dim, num_layers, dropout):
        """Build a tower network with variable number of layers."""
        layers = []
        
        # First layer: input_dim -> hidden_dim
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.Dropout(dropout))
        layers.append(nn.ReLU())
        
        # Hidden layers: hidden_dim -> hidden_dim
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Dropout(dropout))
            layers.append(nn.ReLU())
        
        return nn.Sequential(*layers)
    
    def encode(
        self, 
        x_dict: dict[str, torch.Tensor], 
        edge_index_dict: dict[tuple[str, str, str], torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """
        Encode nodes into embeddings.
        Takes x_dict and edge_index_dict (edge_index_dict is ignored).
        Returns dict with 'user' and 'item' keys.
        """
        # Extract user and item features from x_dict
        users = x_dict.get("user")
        items = x_dict.get("item")
        
        if users is None or items is None:
            raise ValueError("x_dict must contain 'user' and 'item' keys")
        
        user_embeddings = self.user_tower(users)
        item_embeddings = self.item_tower(items)
        
        return {"user": user_embeddings, "item": item_embeddings}
    
    def forward(
        self, 
        x_dict: dict[str, torch.Tensor], 
        edge_index_dict: dict[tuple[str, str, str], torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass that encodes nodes into embeddings.
        """
        # TODO: Add heads
        return self.encode(x_dict, edge_index_dict)