"""
Light Graph Convolutional Network (LightGCN) for heterogeneous bipartite user-item recommendation.

Based on: "LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation" 
(He et al., 2020) https://arxiv.org/pdf/2002.02126.pdf
"""

import torch
import torch.nn as nn
from torch_geometric.nn import LGConv
from recommender.models.model_helpers import TypeProjector, WeightedDotProductHead


class LightGCN(nn.Module):
    """
    Light Graph Convolutional Network for heterogeneous bipartite user-item recommendation.
    
    Uses PyG's LGConv homogenous convolution(with slight modification) to handle the 
    heterogeneous graph structure with users and items. LightGCN simplifies graph convolutions 
    by removing feature transformations and using a simple neighbor aggregation mechanism.
    
    Implements encode() and forward() methods that take x_dict and edge_index_dict.
    Returns dict with 'user' and 'item' keys to be compatible with simple_trainer.
    """
    def __init__(
        self,
        in_dims: dict[str, int],
        metadata: tuple[list[str], list[tuple[str, str, str]]],
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super(LightGCN, self).__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        self.hidden_dim = hidden_dim
        self.metadata = metadata

        # Project input features to hidden_dim
        self.project = TypeProjector(in_dims, hidden_dim)
        
        self.convs = nn.ModuleList([LGConv() for _ in range(num_layers)])
        
        self.head = WeightedDotProductHead(hidden_dim, hidden_dim)
        self.dropout_layer = nn.Dropout(dropout)

        self.reset_parameters()
    
    def reset_parameters(self):
        # Reset parameters of the projection layer
        for projector in self.project.projector.values():
            nn.init.xavier_uniform_(projector.weight)
            if projector.bias is not None:
                nn.init.zeros_(projector.bias)
    
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
        """Compute scores using the weighted dot product head."""
        return self.head(user_emb, item_emb)

    def encode(
        self, 
        x_dict: dict[str, torch.Tensor],
        edge_index_dict: dict[tuple[str, str, str], torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """
        Encode nodes into embeddings using LightGCN.
        
        LightGCN key difference: Instead of applying transformations within each layer,
        we aggregate neighbor embeddings and then average across all layers (including
        the initial embedding layer). This simplification has been shown to be highly
        effective for recommendation tasks.
        
        Takes x_dict and edge_index_dict.
        Returns dict with 'user' and 'item' keys.
        """
        # Project input features to hidden_dim
        z_dict = self.project(x_dict)
        user_features = z_dict['user']
        item_features = z_dict['item']
        
        num_users = user_features.shape[0]
        num_items = item_features.shape[0]

        user_item_edges = edge_index_dict[('user', 'rates', 'item')]
        item_user_edges = edge_index_dict[('item', 'rev_rates', 'user')]

        # Shift item indices by num_users to align with concatenated embeddings
        item_user_edges = item_user_edges.clone()
        user_item_edges = user_item_edges.clone()
        user_item_edges[1] += num_users
        item_user_edges[0] += num_users

        edge_index = torch.cat([user_item_edges, item_user_edges], dim=1)

        # Concatenate user and item features as initial embeddings
        x = torch.cat([user_features, item_features], dim=0)
        
        # Store all layer embeddings for later averaging
        all_embeddings = [x]
        
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < self.num_layers - 1: # No dropout after last layer 
                x = self.dropout_layer(x) 
            all_embeddings.append(x)

        final_x = torch.mean(torch.stack(all_embeddings), dim=0) # Average over all layers
        return {
            "user": final_x[:num_users],
            "item": final_x[num_users:]
        }