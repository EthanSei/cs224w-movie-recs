# tests/conftest.py
import torch
import pytest
from torch_geometric.data import HeteroData

@pytest.fixture
def tiny_hetero_graph():
    """Small heterogeneous graph with user and item nodes."""
    U, I = 8, 10
    d_user, d_item = 16, 32
    g = HeteroData()
    g['user'].x = torch.randn(U, d_user)
    g['item'].x = torch.randn(I, d_item)
    # user->item edges
    ui = torch.tensor([[0,0,1,2,3,4,5,6,6,7],
                       [1,3,5,2,4,6,7,0,8,9]], dtype=torch.long)
    g['user','rates','item'].edge_index = ui
    # user<->user (optional)
    uu = torch.tensor([[0,1,2,3],[1,2,3,4]])
    g['user','similar','user'].edge_index = torch.cat([uu, uu.flip(0)], dim=1)
    # item<->item (optional)
    ii = torch.tensor([[0,1,2,3],[1,2,3,4]])
    g['item','similar','item'].edge_index = torch.cat([ii, ii.flip(0)], dim=1)
    return g

@pytest.fixture
def large_hetero_graph():
    """Larger heterogeneous graph for testing different graph sizes."""
    U, I = 20, 30
    d_user, d_item = 16, 32
    g = HeteroData()
    g['user'].x = torch.randn(U, d_user)
    g['item'].x = torch.randn(I, d_item)
    # user->item edges
    user_indices = torch.randint(0, U, (50,), dtype=torch.long)
    item_indices = torch.randint(0, I, (50,), dtype=torch.long)
    g['user', 'rates', 'item'].edge_index = torch.stack([user_indices, item_indices], dim=0)
    # user<->user edges
    uu = torch.randint(0, U, (2, 10), dtype=torch.long)
    g['user', 'similar', 'user'].edge_index = torch.cat([uu, uu.flip(0)], dim=1)
    # item<->item edges
    ii = torch.randint(0, I, (2, 10), dtype=torch.long)
    g['item', 'similar', 'item'].edge_index = torch.cat([ii, ii.flip(0)], dim=1)
    return g

@pytest.fixture
def hetero_graph_100_200():
    """Graph with user_dim=100, item_dim=200 for TwoTower tests."""
    g = HeteroData()
    g['user'].x = torch.randn(10, 100)
    g['item'].x = torch.randn(15, 200)
    return g

@pytest.fixture
def hetero_graph_50_50():
    """Graph with user_dim=50, item_dim=50 for TwoTower tests."""
    g = HeteroData()
    g['user'].x = torch.randn(10, 50)
    g['item'].x = torch.randn(20, 50)
    return g