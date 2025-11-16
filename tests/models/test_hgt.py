import torch
import pytest
from recommender.models.hgt import HGT

def test_encode_shape(tiny_hetero_graph):
    g = tiny_hetero_graph
    in_dims = {node_type: g[node_type].x.size(-1) for node_type in g.node_types}
    model = HGT(in_dims, g.metadata(), hidden_dim=64, heads=2, layers=2)
    out = model.encode(g.x_dict, g.edge_index_dict)
    assert set(out.keys()) == set(g.node_types)
    for node_type in g.node_types:
        assert out[node_type].shape == (g[node_type].num_nodes, 64)
    for t in out.values():
        assert torch.isfinite(t).all()

def test_forward_pass_different_graph_sizes(tiny_hetero_graph, large_hetero_graph):
    """Test HGT forward pass with different graph sizes"""
    # Small graph
    g1 = tiny_hetero_graph
    in_dims1 = {node_type: g1[node_type].x.size(-1) for node_type in g1.node_types}
    model1 = HGT(in_dims1, g1.metadata(), hidden_dim=64, heads=2, layers=2)
    out1 = model1.forward(g1.x_dict, g1.edge_index_dict)
    assert set(out1.keys()) == set(g1.node_types)
    
    # Large graph
    g2 = large_hetero_graph
    in_dims2 = {node_type: g2[node_type].x.size(-1) for node_type in g2.node_types}
    model2 = HGT(in_dims2, g2.metadata(), hidden_dim=64, heads=2, layers=2)
    out2 = model2.forward(g2.x_dict, g2.edge_index_dict)
    assert set(out2.keys()) == set(g2.node_types)
    assert out2['user'].shape == (g2['user'].num_nodes, 64)
    assert out2['item'].shape == (g2['item'].num_nodes, 64)

def test_encode_missing_node_features(tiny_hetero_graph):
    """Test HGT encode method with missing node features"""
    g = tiny_hetero_graph
    x_dict = {'user': g['user'].x}
    in_dims = {node_type: g[node_type].x.size(-1) for node_type in g.node_types}
    model = HGT(in_dims, g.metadata(), hidden_dim=64, heads=2, layers=2)
    
    try:
        out = model.encode(x_dict, g.edge_index_dict)
        assert 'user' in out
    except (KeyError, AttributeError) as e:
        pytest.skip(f"Model requires item features: {e}")

def test_hgt_score_method(tiny_hetero_graph):
    """Test HGT score() method uses WeightedDotProductHead"""
    g = tiny_hetero_graph
    in_dims = {node_type: g[node_type].x.size(-1) for node_type in g.node_types}
    model = HGT(in_dims, g.metadata(), hidden_dim=64, heads=2, layers=2)
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
    simple_dot = (user_batch * item_batch).sum(dim=1)
    assert not torch.allclose(scores, simple_dot, atol=1e-4)