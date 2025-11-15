"""
Graph Attention Network (GAT) for heterogeneous bipartite user-movie recommendation.

Based on: "Graph Attention Networks" (Veličković et al., 2017)
https://arxiv.org/pdf/1710.10903

GATv2 is from "How Attentive are Graph Attention Networks?" (Brody et al., 2021)
https://arxiv.org/abs/2105.14491
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, HeteroConv


class GAT(nn.Module):
    """
    Graph Attention Network for heterogeneous bipartite user-movie recommendation.
    
    Uses PyG's GATv2Conv wrapped in HeteroConv to handle the heterogeneous graph
    structure with users and movies. The model applies multi-head attention to learn
    which user-movie interactions are most important for generating embeddings.
    """
    def __init__(
        self,
        user_dim,
        item_dim,
        hidden_dim=128,
        num_heads=2,
        num_layers=2,
        dropout=0.1
    ):
        super(GAT, self).__init__()
        
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.hidden_dim = hidden_dim
        
        #Input projection to the hidden dimension
        self.user_proj = nn.Linear(user_dim, hidden_dim)
        self.item_proj = nn.Linear(item_dim, hidden_dim)
        
        # Build GAT layers for both type of edges
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            in_dim = hidden_dim if i == 0 else hidden_dim * num_heads
            
            hetero_conv = HeteroConv({
                ('user', 'rates', 'movie'): GATv2Conv(
                    in_channels=in_dim,
                    out_channels=hidden_dim,
                    heads=num_heads,
                    dropout=dropout,
                    add_self_loops=False,
                    concat=True
                ),
                ('movie', 'rev_rates', 'user'): GATv2Conv(
                    in_channels=in_dim,
                    out_channels=hidden_dim,
                    heads=num_heads,
                    dropout=dropout,
                    add_self_loops=False,
                    concat=True
                ),
            }, aggr='sum')
            
            self.convs.append(hetero_conv)
    
    def forward(self, user_features, item_features, edge_index_dict):
        # Project input features to the hidden dimension
        x_dict = {
            'user': self.user_proj(user_features),
            'movie': self.item_proj(item_features)
        }

        # Apply GAT layers for both type of edges
        for i, conv in enumerate(self.convs):
        
            x_dict = conv(x_dict, edge_index_dict)

            #for dropout and activation between layers
            if i < self.num_layers - 1:
                x_dict = {key: F.relu(x) for key, x in x_dict.items()}
                x_dict = {
                    key: F.dropout(x, p=self.dropout, training=self.training)
                    for key, x in x_dict.items()
                }
        
        # Return embeddings for users and movies
        return x_dict['user'], x_dict['movie']
