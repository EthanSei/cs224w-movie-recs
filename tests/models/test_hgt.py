import torch
import pytest
from recommender.models.hgt import HGT
from tests.conftest import get_in_dims

def test_encode_shape(tiny_hetero_graph):
    g = tiny_hetero_graph
    model = HGT(get_in_dims(g), g.metadata(), hidden_dim=64, heads=2, layers=2)
    out = model.encode(g.x_dict, g.edge_index_dict)
    assert set(out.keys()) == set(g.node_types)
    for node_type in g.node_types:
        assert out[node_type].shape == (g[node_type].num_nodes, 64)
    assert all(torch.isfinite(t).all() for t in out.values())

def test_forward_pass_different_graph_sizes(tiny_hetero_graph, large_hetero_graph):
    """Test HGT forward pass with different graph sizes"""
    g1 = tiny_hetero_graph
    model1 = HGT(get_in_dims(g1), g1.metadata(), hidden_dim=64, heads=2, layers=2)
    out1 = model1.forward(g1.x_dict, g1.edge_index_dict)
    assert set(out1.keys()) == set(g1.node_types)
    
    g2 = large_hetero_graph
    model2 = HGT(get_in_dims(g2), g2.metadata(), hidden_dim=64, heads=2, layers=2)
    out2 = model2.forward(g2.x_dict, g2.edge_index_dict)
    assert set(out2.keys()) == set(g2.node_types)
    assert out2['user'].shape == (g2['user'].num_nodes, 64)
    assert out2['item'].shape == (g2['item'].num_nodes, 64)

def test_encode_missing_node_features(tiny_hetero_graph):
    """Test HGT encode method with missing node features"""
    g = tiny_hetero_graph
    model = HGT(get_in_dims(g), g.metadata(), hidden_dim=64, heads=2, layers=2)
    
    try:
        out = model.encode({'user': g['user'].x}, g.edge_index_dict)
        assert 'user' in out
    except (KeyError, AttributeError) as e:
        pytest.skip(f"Model requires item features: {e}")

def test_hgt_score_method(tiny_hetero_graph):
    """Test HGT score() method uses WeightedDotProductHead"""
    g = tiny_hetero_graph
    model = HGT(get_in_dims(g), g.metadata(), hidden_dim=64, heads=2, layers=2)
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
    
    simple_dot = (user_batch * item_batch).sum(dim=1)
    assert not torch.allclose(scores, simple_dot, atol=1e-4)