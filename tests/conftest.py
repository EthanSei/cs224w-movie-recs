# tests/conftest.py
import torch
import pytest
from torch_geometric.data import HeteroData

@pytest.fixture
def tiny_hetero_graph():
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