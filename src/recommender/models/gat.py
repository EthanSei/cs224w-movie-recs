"""
Graph Attention Network (GAT) for heterogeneous bipartite user-item recommendation.

Based on: "Graph Attention Networks" (Veličković et al., 2017)
https://arxiv.org/pdf/1710.10903

GATv2 is from "How Attentive are Graph Attention Networks?" (Brody et al., 2021)
https://arxiv.org/abs/2105.14491
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, HeteroConv
from recommender.models.model_helpers import TypeProjector, DotProductHead


class GAT(nn.Module):
    """
    Graph Attention Network for heterogeneous bipartite user-item recommendation.
    
    Uses PyG's GATv2Conv wrapped in HeteroConv to handle the heterogeneous graph
    structure with users and items. The model applies multi-head attention to learn
    which user-item interactions are most important for generating embeddings.
    
    Implements encode() and forward() methods that take x_dict and edge_index_dict.
    Returns dict with 'user' and 'item' keys to be compatible with simple_trainer.
    """
    def __init__(
        self,
        in_dims: dict[str, int],
        metadata: tuple[list[str], list[tuple[str, str, str]]],
        hidden_dim: int = 128,
        num_heads: int = 2,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super(GAT, self).__init__()
        
        # Ensure hidden_dim is divisible by num_heads for fair comparison with HGT
        if hidden_dim % num_heads != 0:
            raise ValueError(f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})")
        
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.hidden_dim = hidden_dim
        self.metadata = metadata
        
        # Dimension per head (normalized so final output = hidden_dim, not hidden_dim * num_heads)
        hidden_dim_per_head = hidden_dim // num_heads
        
        # Input projection to the hidden dimension
        self.project = TypeProjector(in_dims, hidden_dim)
        
        # Build GAT layers dynamically based on metadata
        # All layers: hidden_dim -> hidden_dim (normalized so final output = hidden_dim, not hidden_dim * num_heads)
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            in_dim = hidden_dim
            
            # Build conv dict for all edge types in metadata
            conv_dict = {}
            for edge_type in metadata[1]:
                conv_dict[edge_type] = GATv2Conv(
                    in_channels=in_dim,
                    out_channels=hidden_dim_per_head,
                    heads=num_heads,
                    dropout=dropout,
                    add_self_loops=False,
                    concat=True
                )
            
            hetero_conv = HeteroConv(conv_dict, aggr='sum')
            self.convs.append(hetero_conv)
        
        self.head = DotProductHead()
        self.dropout_layer = nn.Dropout(dropout)
        self.activation = nn.ReLU()
        self.reset_parameters()
    
    def reset_parameters(self):
        """Reset parameters of all GAT layers."""
        for conv in self.convs:
            for gat_conv in conv.convs.values():
                gat_conv.reset_parameters()
    
    def forward(
        self, 
        x_dict: dict[str, torch.Tensor], 
        edge_index_dict: dict[tuple[str, str, str], torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Forward pass that returns embeddings (for backward compatibility with tests)."""
        return self.encode(x_dict, edge_index_dict)
    
    def score(
        self,
        user_emb: torch.Tensor,
        item_emb: torch.Tensor,
    ) -> torch.Tensor:
        """Compute scores using dot product."""
        return self.head(user_emb, item_emb)

    def encode(
        self, 
        x_dict: dict[str, torch.Tensor], 
        edge_index_dict: dict[tuple[str, str, str], torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """
        Encode nodes into embeddings.
        Takes x_dict and edge_index_dict.
        Returns dict with 'user' and 'item' keys.
        """
        # Project input features to the hidden dimension
        z_dict = self.project(x_dict)
        
        # Apply GAT layers
        for i, conv in enumerate(self.convs):
            z_dict = conv(z_dict, edge_index_dict)
            
            # Apply dropout and activation between layers (except last layer)
            if i < self.num_layers - 1:
                z_dict = {k: self.dropout_layer(v) for k, v in z_dict.items()}
                z_dict = {k: self.activation(v) for k, v in z_dict.items()}
        
        # Return embeddings for users and items
        return z_dict
