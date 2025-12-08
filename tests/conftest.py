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


class MockModel:
    """Mock model for testing evaluator."""
    def __init__(self, user_embeddings, item_embeddings):
        """
        Args:
            user_embeddings: Tensor of shape [num_users, hidden_dim]
            item_embeddings: Tensor of shape [num_items, hidden_dim]
        """
        self.user_embeddings = user_embeddings
        self.item_embeddings = item_embeddings
        self.hidden_dim = user_embeddings.size(1)
        self.training = False
    
    def eval(self):
        self.training = False
    
    def to(self, device):
        self.user_embeddings = self.user_embeddings.to(device)
        self.item_embeddings = self.item_embeddings.to(device)
        return self
    
    def encode(self, x_dict, edge_index_dict):
        """Return pre-computed embeddings."""
        return {
            "user": self.user_embeddings,
            "item": self.item_embeddings
        }
    
    def score(self, user_emb, item_emb):
        """Compute dot product scores."""
        return (user_emb * item_emb).sum(dim=1)


@pytest.fixture
def simple_test_data():
    """Simple test data with 3 users, 5 items, and known test edges."""
    data = HeteroData()
    data['user'].x = torch.randn(3, 16)
    data['item'].x = torch.randn(5, 32)
    
    # Training edges (for message passing)
    train_edges = torch.tensor([[0, 0, 1, 1], [0, 1, 0, 2]], dtype=torch.long)
    data['user', 'rates', 'item'].edge_index = train_edges
    
    # Test edges (ground truth for evaluation)
    test_edges = torch.tensor([[0, 1, 2], [2, 3, 4]], dtype=torch.long)
    data['user', 'rates', 'item'].edge_label_index = test_edges
    
    return data


@pytest.fixture
def mock_model_simple(simple_test_data):
    """Mock model with deterministic embeddings for simple test."""
    num_users, num_items = 3, 5
    hidden_dim = 8
    
    # Create embeddings that give predictable scores
    # User 0 should score highest with item 2
    # User 1 should score highest with item 3
    # User 2 should score highest with item 4
    user_emb = torch.zeros(num_users, hidden_dim)
    item_emb = torch.zeros(num_items, hidden_dim)
    
    # Set up so user-item pairs have high scores
    user_emb[0, :4] = 1.0  # User 0
    item_emb[2, :4] = 1.0  # Item 2 (test edge for user 0)
    
    user_emb[1, :4] = 1.0  # User 1
    item_emb[3, :4] = 1.0  # Item 3 (test edge for user 1)
    
    user_emb[2, :4] = 1.0  # User 2
    item_emb[4, :4] = 1.0  # Item 4 (test edge for user 2)
    
    return MockModel(user_emb, item_emb)


# Helper functions for model tests
def get_in_dims(graph):
    """Helper to extract input dimensions from a graph."""
    return {node_type: graph[node_type].x.size(-1) for node_type in graph.node_types}


def create_eval_test_data(num_users=2, num_items=5, train_edges=None, test_edges=None):
    """Helper to create test data for evaluator tests."""
    data = HeteroData()
    data['user'].x = torch.randn(num_users, 16)
    data['item'].x = torch.randn(num_items, 32)
    
    if train_edges is None:
        train_edges = torch.tensor([[0], [0]], dtype=torch.long)
    data['user', 'rates', 'item'].edge_index = train_edges
    
    if test_edges is not None:
        data['user', 'rates', 'item'].edge_label_index = test_edges
    
    return data


def create_mock_model_with_scores(num_users, num_items, hidden_dim, user_item_scores):
    """
    Helper to create a MockModel with specific user-item score patterns.
    
    Args:
        num_users: Number of users
        num_items: Number of items
        hidden_dim: Hidden dimension size
        user_item_scores: Dict mapping (user_idx, item_idx) -> score_value
    """
    user_emb = torch.zeros(num_users, hidden_dim)
    item_emb = torch.zeros(num_items, hidden_dim)
    
    for (u_idx, i_idx), score in user_item_scores.items():
        user_emb[u_idx, :4] = score
        item_emb[i_idx, :4] = score
    
    return MockModel(user_emb, item_emb)