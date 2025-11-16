# tests/conftest.py
import torch
import pytest
import pandas as pd
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
    # Reverse edges for GAT compatibility
    g['item','rev_rates','user'].edge_index = ui.flip(0)
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
    ui_edge_index = torch.stack([user_indices, item_indices], dim=0)
    g['user', 'rates', 'item'].edge_index = ui_edge_index
    # Reverse edges for GAT compatibility
    g['item', 'rev_rates', 'user'].edge_index = ui_edge_index.flip(0)
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
    # Add minimal edges for GAT compatibility
    ui_edge_index = torch.tensor([[0, 1, 2], [0, 1, 2]], dtype=torch.long)
    g['user', 'rates', 'item'].edge_index = ui_edge_index
    g['item', 'rev_rates', 'user'].edge_index = ui_edge_index.flip(0)
    return g

@pytest.fixture
def hetero_graph_50_50():
    """Graph with user_dim=50, item_dim=50 for TwoTower tests."""
    g = HeteroData()
    g['user'].x = torch.randn(10, 50)
    g['item'].x = torch.randn(20, 50)
    # Add minimal edges for GAT compatibility
    ui_edge_index = torch.tensor([[0, 1, 2], [0, 1, 2]], dtype=torch.long)
    g['user', 'rates', 'item'].edge_index = ui_edge_index
    g['item', 'rev_rates', 'user'].edge_index = ui_edge_index.flip(0)
    return g

@pytest.fixture
def sample_movies_df():
    """Sample movies DataFrame for testing feature building."""
    return pd.DataFrame({
        "movieId": [1, 2, 3],
        "title": ["Movie 1 (1995)", "Movie 2 (2000)", "Movie 3 (2010)"],
        "genres": ["Action|Adventure", "Comedy|Romance", "Drama"]
    })

@pytest.fixture
def movies_with_known_years():
    """Movies DataFrame with titles that need known_years lookup."""
    return pd.DataFrame({
        "movieId": [1, 2],
        "title": ["Ready Player One", "Moonlight"],
        "genres": ["Action", "Drama"]
    })

@pytest.fixture
def movies_edge_cases():
    """Movies DataFrame with edge cases (empty genres, out-of-bounds years)."""
    return pd.DataFrame({
        "movieId": [1, 2, 3, 4],
        "title": ["Movie 1 (1995)", "Movie 2 (2000)", "Old Movie (1890)", "Future Movie (2030)"],
        "genres": ["Action|Adventure", "", "Drama", "Sci-Fi"]
    })

@pytest.fixture
def movies_no_year():
    """Movies DataFrame with no extractable year."""
    return pd.DataFrame({
        "movieId": [1],
        "title": ["Unknown Movie"],
        "genres": ["Action"]
    })

@pytest.fixture
def movies_realistic():
    """Realistic MovieLens-style movies DataFrame."""
    return pd.DataFrame({
        "movieId": [1, 2, 3],
        "title": [
            "Toy Story (1995)",
            "Jumanji (1995)",
            "Ready Player One"
        ],
        "genres": [
            "Adventure|Animation|Children|Comedy|Fantasy",
            "Adventure|Children|Fantasy",
            "Action|Adventure|Sci-Fi"
        ]
    })