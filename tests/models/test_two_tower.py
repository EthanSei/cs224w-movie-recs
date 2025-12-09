import torch
import pytest
from recommender.models.two_tower import TwoTower
from tests.conftest import get_in_dims

def test_two_tower_encode_interface(tiny_hetero_graph):
    """Test TwoTower model encode() method"""
    g = tiny_hetero_graph
    model = TwoTower(in_dims=get_in_dims(g), hidden_dim=64, num_layers=2, dropout=0.1)
    z_dict = model.encode(g.x_dict, g.edge_index_dict)
    assert "user" in z_dict
    assert "item" in z_dict
    assert z_dict["user"].shape == (g['user'].num_nodes, 64)
    assert z_dict["item"].shape == (g['item'].num_nodes, 64)

def test_two_tower_different_dims(tiny_hetero_graph, hetero_graph_100_200):
    """Test TwoTower model with different input dimensions"""
    # Test with tiny_hetero_graph dimensions (16, 32)
    g1 = tiny_hetero_graph
    model1 = TwoTower(in_dims=get_in_dims(g1), hidden_dim=64, num_layers=2, dropout=0.1)
    z_dict1 = model1.encode(g1.x_dict, g1.edge_index_dict)
    assert z_dict1["user"].shape == (g1['user'].num_nodes, 64)
    assert z_dict1["item"].shape == (g1['item'].num_nodes, 64)
    
    # Test with larger dimensions (100, 200)
    g2 = hetero_graph_100_200
    model2 = TwoTower(in_dims=get_in_dims(g2), hidden_dim=128, num_layers=2, dropout=0.1)
    z_dict2 = model2.encode(g2.x_dict, {})
    assert z_dict2["user"].shape == (g2['user'].num_nodes, 128)
    assert z_dict2["item"].shape == (g2['item'].num_nodes, 128)

def test_two_tower_variable_layers(hetero_graph_50_50):
    """Test TwoTower model with variable number of layers"""
    g = hetero_graph_50_50
    in_dims = get_in_dims(g)
    
    model1 = TwoTower(in_dims=in_dims, hidden_dim=64, num_layers=1, dropout=0.1)
    z_dict1 = model1.encode(g.x_dict, {})
    assert z_dict1["user"].shape == (g['user'].num_nodes, 64)
    assert z_dict1["item"].shape == (g['item'].num_nodes, 64)
    
    model3 = TwoTower(in_dims=in_dims, hidden_dim=64, num_layers=3, dropout=0.1)
    z_dict3 = model3.encode(g.x_dict, {})
    assert z_dict3["user"].shape == (g['user'].num_nodes, 64)
    assert z_dict3["item"].shape == (g['item'].num_nodes, 64)
    
    assert not torch.allclose(z_dict1["user"], z_dict3["user"], atol=1e-6)

def test_two_tower_forward_interface(tiny_hetero_graph):
    """Test TwoTower forward() method"""
    g = tiny_hetero_graph
    model = TwoTower(in_dims=get_in_dims(g), hidden_dim=64, num_layers=2, dropout=0.0)
    model.eval()
    
    z_dict = model.forward(g.x_dict, g.edge_index_dict)
    assert "user" in z_dict
    assert "item" in z_dict
    assert z_dict["user"].shape == (g['user'].num_nodes, 64)
    assert z_dict["item"].shape == (g['item'].num_nodes, 64)
    
    z_dict2 = model.encode(g.x_dict, g.edge_index_dict)
    assert torch.allclose(z_dict["user"], z_dict2["user"], atol=1e-6)
    assert torch.allclose(z_dict["item"], z_dict2["item"], atol=1e-6)

def test_two_tower_score_method(tiny_hetero_graph):
    """Test TwoTower score() method uses WeightedDotProductHead"""
    g = tiny_hetero_graph
    model = TwoTower(in_dims=get_in_dims(g), hidden_dim=64, num_layers=2, dropout=0.1)
    model.eval()
    
    z_dict = model.encode(g.x_dict, g.edge_index_dict)
    user_emb = z_dict["user"]
    item_emb = z_dict["item"]
    
    # Test score() with batched user-item pairs
    batch_size = 5
    user_batch = user_emb[torch.randint(0, user_emb.size(0), (batch_size,))]
    item_batch = item_emb[torch.randint(0, item_emb.size(0), (batch_size,))]
    
    scores = model.score(user_batch, item_batch)
    assert scores.shape == (batch_size,)
    assert torch.isfinite(scores).all()
    
    # Verify it's using weighted dot product (not just dot product)
    simple_dot = (user_batch * item_batch).sum(dim=1)
    assert not torch.allclose(scores, simple_dot, atol=1e-4)
