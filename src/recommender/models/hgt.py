# src/recommender/models/hgt.py

import torch
import torch.nn as nn
from torch_geometric.nn import HGTConv
from recommender.models.model_helpers import TypeProjector, DotProductHead

"""
Heterogeneous Graph Transformer (HGT) model.
"""
class HGT(nn.Module):
    def __init__(
        self, 
        in_dims: dict[str, int],
        metadata: tuple[list[str],list[tuple[str,str,str]]],
        hidden_dim: int = 128,
        heads: int = 4,
        layers: int = 2,
        dropout: float = 0.0,
    ):
        super(HGT, self).__init__()
        self.project = TypeProjector(in_dims, hidden_dim)
        self.convs = nn.ModuleList([
            HGTConv(
                metadata=metadata,
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                heads=heads,
            )
            for _ in range(layers)
        ])
        self.layers = layers
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()
        
        # Dot product head for scoring
        self.head = DotProductHead()
        
        self.reset_parameters()

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x_dict: dict[str, torch.Tensor], edge_index_dict: dict[tuple[str, str, str], torch.Tensor]) -> dict[str, torch.Tensor]:
        """Forward pass that returns embeddings (for backward compatibility with tests)."""
        return self.encode(x_dict, edge_index_dict)
    
    def score(
        self,
        user_emb: torch.Tensor,
        item_emb: torch.Tensor,
    ) -> torch.Tensor:
        """Compute scores using dot product."""
        return self.head(user_emb, item_emb)

    def encode(self, x_dict: dict[str, torch.Tensor], edge_index_dict: dict[tuple[str, str, str], torch.Tensor]) -> dict[str, torch.Tensor]:
        z_dict = self.project(x_dict)
        for i, conv in enumerate(self.convs):
            z_dict = conv(z_dict, edge_index_dict)
            if i < self.layers - 1:
                z_dict = {k: self.dropout(v) for k, v in z_dict.items()}
                z_dict = {k: self.activation(v) for k, v in z_dict.items()}
        return z_dict