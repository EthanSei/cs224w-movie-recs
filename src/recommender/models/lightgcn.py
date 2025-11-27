"""
Light Graph Convolutional Network (LightGCN) for heterogeneous bipartite user-item recommendation.

Based on: "LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation" 
(He et al., 2020) https://arxiv.org/pdf/2002.02126.pdf
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
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
        num_users: int,
        num_items: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super(LightGCN, self).__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        self.hidden_dim = hidden_dim
        self.num_users = num_users
        self.num_items = num_items

        self.user_emb = nn.Embedding(self.num_users, hidden_dim)
        self.item_emb = nn.Embedding(self.num_items, hidden_dim)
        
        self.convs = nn.ModuleList([LGConv() for _ in range(num_layers)])
        
        self.head = WeightedDotProductHead(hidden_dim, hidden_dim)
        self.dropout_layer = nn.Dropout(dropout)

        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.user_emb.weight)
        nn.init.xavier_uniform_(self.item_emb.weight)
    
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

        user_item_edges = edge_index_dict[('user', 'rates', 'item')]
        item_user_edges = edge_index_dict[('item', 'rev_rates', 'user')]

        # Shift item indices by num_users to align with concatenated embeddings
        item_user_edges = item_user_edges.clone()
        user_item_edges = user_item_edges.clone()
        user_item_edges[1] += self.num_users
        item_user_edges[0] += self.num_users

        edge_index = torch.cat([user_item_edges, item_user_edges], dim=1)

        # Project initial embeddings to hidden_dim
        x = torch.cat([self.user_emb.weight, self.item_emb.weight], dim=0)
        
        # Store all layer embeddings for later averaging
        all_embeddings = [x]
        
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < self.num_layers - 1: # No dropout after last layer 
                x = self.dropout_layer(x) 
            all_embeddings.append(x)

        final_x = torch.mean(torch.stack(all_embeddings), dim=0) # Average over all layers
        return {
            "user": final_x[:self.num_users],
            "item": final_x[self.num_users:]
        }