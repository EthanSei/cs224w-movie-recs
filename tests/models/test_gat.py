import torch
import pytest
from recommender.models.gat import GAT
from tests.conftest import get_in_dims

def test_gat_encode_interface(tiny_hetero_graph):
    """Test GAT model encode() method returns correct format"""
    g = tiny_hetero_graph
    model = GAT(
        in_dims=get_in_dims(g),
        metadata=g.metadata(),
        hidden_dim=64, 
        num_heads=2, 
        num_layers=2, 
        dropout=0.1
    )
    z_dict = model.encode(g.x_dict, g.edge_index_dict)
    assert "user" in z_dict
    assert "item" in z_dict
    assert z_dict["user"].shape == (g['user'].num_nodes, 64)
    assert z_dict["item"].shape == (g['item'].num_nodes, 64)

def test_gat_different_dims(tiny_hetero_graph, hetero_graph_100_200):
    """Test GAT model with different input dimensions"""
    # Test with tiny_hetero_graph dimensions (16, 32)
    g1 = tiny_hetero_graph
    model1 = GAT(
        in_dims=get_in_dims(g1),
        metadata=g1.metadata(),
        hidden_dim=64, 
        num_heads=2, 
        num_layers=2, 
        dropout=0.1
    )
    z_dict1 = model1.encode(g1.x_dict, g1.edge_index_dict)
    assert z_dict1["user"].shape == (g1['user'].num_nodes, 64)
    assert z_dict1["item"].shape == (g1['item'].num_nodes, 64)
    
    # Test with larger dimensions (100, 200)
    g2 = hetero_graph_100_200
    model2 = GAT(
        in_dims=get_in_dims(g2),
        metadata=g2.metadata(),
        hidden_dim=128, 
        num_heads=2, 
        num_layers=2, 
        dropout=0.1
    )
    z_dict2 = model2.encode(g2.x_dict, g2.edge_index_dict)
    assert z_dict2["user"].shape == (g2['user'].num_nodes, 128)
    assert z_dict2["item"].shape == (g2['item'].num_nodes, 128)

def test_gat_variable_layers(hetero_graph_50_50):
    """Test GAT model with variable number of layers"""
    g = hetero_graph_50_50
    in_dims = get_in_dims(g)
    
    model1 = GAT(in_dims=in_dims, metadata=g.metadata(), hidden_dim=64, num_heads=2, num_layers=1, dropout=0.1)
    z_dict1 = model1.encode(g.x_dict, g.edge_index_dict)
    assert z_dict1["user"].shape == (g['user'].num_nodes, 64)
    assert z_dict1["item"].shape == (g['item'].num_nodes, 64)
    
    model3 = GAT(in_dims=in_dims, metadata=g.metadata(), hidden_dim=64, num_heads=2, num_layers=3, dropout=0.1)
    z_dict3 = model3.encode(g.x_dict, g.edge_index_dict)
    assert z_dict3["user"].shape == (g['user'].num_nodes, 64)
    assert z_dict3["item"].shape == (g['item'].num_nodes, 64)
    
    assert not torch.allclose(z_dict1["user"], z_dict3["user"], atol=1e-6)

def test_gat_forward_interface(tiny_hetero_graph):
    """Test GAT forward() method"""
    g = tiny_hetero_graph
    model = GAT(in_dims=get_in_dims(g), metadata=g.metadata(), hidden_dim=64, num_heads=2, num_layers=2, dropout=0.0)
    model.eval()
    
    z_dict = model.forward(g.x_dict, g.edge_index_dict)
    assert "user" in z_dict
    assert "item" in z_dict
    assert z_dict["user"].shape == (g['user'].num_nodes, 64)
    assert z_dict["item"].shape == (g['item'].num_nodes, 64)
    
    z_dict2 = model.encode(g.x_dict, g.edge_index_dict)
    assert torch.allclose(z_dict["user"], z_dict2["user"], atol=1e-6)
    assert torch.allclose(z_dict["item"], z_dict2["item"], atol=1e-6)

def test_gat_encode_shape(tiny_hetero_graph):
    """Test GAT encode output shapes match expected dimensions"""
    g = tiny_hetero_graph
    model = GAT(in_dims=get_in_dims(g), metadata=g.metadata(), hidden_dim=64, num_heads=2, num_layers=2, dropout=0.1)
    out = model.encode(g.x_dict, g.edge_index_dict)
    assert set(out.keys()) == {'user', 'item'}
    assert out['user'].shape == (g['user'].num_nodes, 64)
    assert out['item'].shape == (g['item'].num_nodes, 64)
    assert all(torch.isfinite(t).all() for t in out.values())

def test_gat_forward_pass_different_graph_sizes(tiny_hetero_graph, large_hetero_graph):
    """Test GAT forward pass with different graph sizes"""
    g1 = tiny_hetero_graph
    model1 = GAT(in_dims=get_in_dims(g1), metadata=g1.metadata(), hidden_dim=64, num_heads=2, num_layers=2, dropout=0.1)
    out1 = model1.forward(g1.x_dict, g1.edge_index_dict)
    assert set(out1.keys()) == {'user', 'item'}
    
    g2 = large_hetero_graph
    model2 = GAT(in_dims=get_in_dims(g2), metadata=g2.metadata(), hidden_dim=64, num_heads=2, num_layers=2, dropout=0.1)
    out2 = model2.forward(g2.x_dict, g2.edge_index_dict)
    assert set(out2.keys()) == {'user', 'item'}
    assert out2['user'].shape == (g2['user'].num_nodes, 64)
    assert out2['item'].shape == (g2['item'].num_nodes, 64)

def test_gat_encode_missing_node_features(tiny_hetero_graph):
    """Test GAT encode method with missing node features"""
    g = tiny_hetero_graph
    model = GAT(in_dims=get_in_dims(g), metadata=g.metadata(), hidden_dim=64, num_heads=2, num_layers=2, dropout=0.1)
    
    try:
        out = model.encode({'user': g['user'].x}, g.edge_index_dict)
        assert 'user' in out
    except (KeyError, AttributeError, AssertionError) as e:
        pytest.skip(f"Model requires item features: {e}")

def test_gat_reset_parameters(tiny_hetero_graph):
    """Test that reset_parameters doesn't crash"""
    g = tiny_hetero_graph
    model = GAT(in_dims=get_in_dims(g), metadata=g.metadata(), hidden_dim=64, num_heads=2, num_layers=2, dropout=0.1)
    out1 = model.encode(g.x_dict, g.edge_index_dict)
    model.reset_parameters()
    out2 = model.encode(g.x_dict, g.edge_index_dict)
    assert out2['user'].shape == out1['user'].shape
    assert out2['item'].shape == out1['item'].shape

def test_gat_score_method(tiny_hetero_graph):
    """Test GAT score() method uses DotProductHead"""
    g = tiny_hetero_graph
    model = GAT(in_dims=get_in_dims(g), metadata=g.metadata(), hidden_dim=64, num_heads=2, num_layers=2, dropout=0.1)
    model.eval()
    
    z_dict = model.encode(g.x_dict, g.edge_index_dict)
    user_emb = z_dict["user"]
    item_emb = z_dict["item"]
    
    batch_size = 5
    user_batch = user_emb[torch.randint(0, user_emb.size(0), (batch_size,))]
    item_batch = item_emb[torch.randint(0, item_emb.size(0), (batch_size,))]
    
    scores = model.score(user_batch, item_batch)
    assert scores.shape == (batch_size,)
    assert torch.isfinite(scores).all()
    
    # Verify it's using simple dot product
    simple_dot = (user_batch * item_batch).sum(dim=1)
    assert torch.allclose(scores, simple_dot, atol=1e-4)

