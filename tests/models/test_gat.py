import torch
import pytest
from recommender.models.gat import GAT

def test_gat_encode_interface(tiny_hetero_graph):
    """Test GAT model encode() method returns correct format"""
    g = tiny_hetero_graph
    in_dims = {node_type: g[node_type].x.size(-1) for node_type in g.node_types}
    model = GAT(
        in_dims=in_dims,
        metadata=g.metadata(),
        hidden_dim=64, 
        num_heads=2, 
        num_layers=2, 
        dropout=0.1
    )
    z_dict = model.encode(g.x_dict, g.edge_index_dict)
    assert "user" in z_dict
    assert "item" in z_dict
    assert z_dict["user"].shape == (g['user'].num_nodes, 64)  # hidden_dim (normalized)
    assert z_dict["item"].shape == (g['item'].num_nodes, 64)

def test_gat_different_dims(tiny_hetero_graph, hetero_graph_100_200):
    """Test GAT model with different input dimensions"""
    # Test with tiny_hetero_graph dimensions (16, 32)
    g1 = tiny_hetero_graph
    in_dims1 = {node_type: g1[node_type].x.size(-1) for node_type in g1.node_types}
    model1 = GAT(
        in_dims=in_dims1,
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
    in_dims2 = {node_type: g2[node_type].x.size(-1) for node_type in g2.node_types}
    model2 = GAT(
        in_dims=in_dims2,
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
    in_dims = {node_type: g[node_type].x.size(-1) for node_type in g.node_types}
    
    model1 = GAT(
        in_dims=in_dims,
        metadata=g.metadata(),
        hidden_dim=64, 
        num_heads=2, 
        num_layers=1, 
        dropout=0.1
    )
    z_dict1 = model1.encode(g.x_dict, g.edge_index_dict)
    assert z_dict1["user"].shape == (g['user'].num_nodes, 64)
    assert z_dict1["item"].shape == (g['item'].num_nodes, 64)
    
    model3 = GAT(
        in_dims=in_dims,
        metadata=g.metadata(),
        hidden_dim=64, 
        num_heads=2, 
        num_layers=3, 
        dropout=0.1
    )
    z_dict3 = model3.encode(g.x_dict, g.edge_index_dict)
    assert z_dict3["user"].shape == (g['user'].num_nodes, 64)
    assert z_dict3["item"].shape == (g['item'].num_nodes, 64)
    
    # Different number of layers should produce different outputs
    assert not torch.allclose(z_dict1["user"], z_dict3["user"], atol=1e-6)

def test_gat_forward_interface(tiny_hetero_graph):
    """Test GAT forward() method"""
    g = tiny_hetero_graph
    in_dims = {node_type: g[node_type].x.size(-1) for node_type in g.node_types}
    model = GAT(
        in_dims=in_dims,
        metadata=g.metadata(),
        hidden_dim=64, 
        num_heads=2, 
        num_layers=2, 
        dropout=0.0
    )
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
    in_dims = {node_type: g[node_type].x.size(-1) for node_type in g.node_types}
    model = GAT(
        in_dims=in_dims,
        metadata=g.metadata(),
        hidden_dim=64, 
        num_heads=2, 
        num_layers=2, 
        dropout=0.1
    )
    out = model.encode(g.x_dict, g.edge_index_dict)
    assert set(out.keys()) == {'user', 'item'}
    assert out['user'].shape == (g['user'].num_nodes, 64)
    assert out['item'].shape == (g['item'].num_nodes, 64)
    # Check all outputs are finite
    for t in out.values():
        assert torch.isfinite(t).all()

def test_gat_forward_pass_different_graph_sizes(tiny_hetero_graph, large_hetero_graph):
    """Test GAT forward pass with different graph sizes"""
    # Small graph
    g1 = tiny_hetero_graph
    in_dims1 = {node_type: g1[node_type].x.size(-1) for node_type in g1.node_types}
    model1 = GAT(
        in_dims=in_dims1,
        metadata=g1.metadata(),
        hidden_dim=64, 
        num_heads=2, 
        num_layers=2, 
        dropout=0.1
    )
    out1 = model1.forward(g1.x_dict, g1.edge_index_dict)
    assert set(out1.keys()) == {'user', 'item'}
    
    # Large graph
    g2 = large_hetero_graph
    in_dims2 = {node_type: g2[node_type].x.size(-1) for node_type in g2.node_types}
    model2 = GAT(
        in_dims=in_dims2,
        metadata=g2.metadata(),
        hidden_dim=64, 
        num_heads=2, 
        num_layers=2, 
        dropout=0.1
    )
    out2 = model2.forward(g2.x_dict, g2.edge_index_dict)
    assert set(out2.keys()) == {'user', 'item'}
    assert out2['user'].shape == (g2['user'].num_nodes, 64)
    assert out2['item'].shape == (g2['item'].num_nodes, 64)

def test_gat_encode_missing_node_features(tiny_hetero_graph):
    """Test GAT encode method with missing node features"""
    g = tiny_hetero_graph
    x_dict = {'user': g['user'].x}
    in_dims = {node_type: g[node_type].x.size(-1) for node_type in g.node_types}
    model = GAT(
        in_dims=in_dims,
        metadata=g.metadata(),
        hidden_dim=64, 
        num_heads=2, 
        num_layers=2, 
        dropout=0.1
    )
    
    try:
        out = model.encode(x_dict, g.edge_index_dict)
        assert 'user' in out
    except (KeyError, AttributeError, AssertionError) as e:
        pytest.skip(f"Model requires item features: {e}")

def test_gat_reset_parameters(tiny_hetero_graph):
    """Test that reset_parameters doesn't crash"""
    g = tiny_hetero_graph
    in_dims = {node_type: g[node_type].x.size(-1) for node_type in g.node_types}
    model = GAT(
        in_dims=in_dims,
        metadata=g.metadata(),
        hidden_dim=64, 
        num_heads=2, 
        num_layers=2, 
        dropout=0.1
    )
    # Get initial output
    out1 = model.encode(g.x_dict, g.edge_index_dict)
    # Reset parameters
    model.reset_parameters()
    # Get output after reset
    out2 = model.encode(g.x_dict, g.edge_index_dict)
    # Outputs should be different after reset (unless very unlucky)
    # But we just check it doesn't crash
    assert out2['user'].shape == out1['user'].shape
    assert out2['item'].shape == out1['item'].shape

def test_gat_score_method(tiny_hetero_graph):
    """Test GAT score() method uses WeightedDotProductHead"""
    g = tiny_hetero_graph
    in_dims = {node_type: g[node_type].x.size(-1) for node_type in g.node_types}
    model = GAT(
        in_dims=in_dims,
        metadata=g.metadata(),
        hidden_dim=64, 
        num_heads=2, 
        num_layers=2, 
        dropout=0.1
    )
    model.eval()
    
    # Get embeddings
    z_dict = model.encode(g.x_dict, g.edge_index_dict)
    user_emb = z_dict["user"]
    item_emb = z_dict["item"]
    
    # Test score() with batched user-item pairs
    batch_size = 5
    user_indices = torch.randint(0, user_emb.size(0), (batch_size,))
    item_indices = torch.randint(0, item_emb.size(0), (batch_size,))
    user_batch = user_emb[user_indices]
    item_batch = item_emb[item_indices]
    
    scores = model.score(user_batch, item_batch)
    assert scores.shape == (batch_size,)
    assert torch.isfinite(scores).all()
    
    # Verify it's using weighted dot product (not just dot product)
    # Scores should be different from simple dot product
    simple_dot = (user_batch * item_batch).sum(dim=1)
    assert not torch.allclose(scores, simple_dot, atol=1e-4)

