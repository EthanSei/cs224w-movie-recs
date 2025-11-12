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