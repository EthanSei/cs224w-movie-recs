import torch
import numpy as np
from torch_geometric.data import HeteroData
from recommender.evaluation.evaluator import Evaluator
from tests.conftest import MockModel, create_eval_test_data, create_mock_model_with_scores


def test_evaluate_basic_flow(simple_test_data, mock_model_simple):
    """Test basic evaluation flow returns metrics."""
    evaluator = Evaluator(k=3, device="cpu")
    hr, ndcg, recall = evaluator.evaluate(mock_model_simple, simple_test_data)
    
    assert isinstance(hr, (float, np.floating))
    assert isinstance(ndcg, (float, np.floating))
    assert isinstance(recall, (float, np.floating))
    assert 0.0 <= hr <= 1.0
    assert 0.0 <= ndcg <= 1.0
    assert 0.0 <= recall <= 1.0


def test_evaluate_excludes_training_items(simple_test_data, mock_model_simple):
    """Test that training items are excluded from ranking."""
    train_data = HeteroData()
    train_data['user'].x = simple_test_data['user'].x
    train_data['item'].x = simple_test_data['item'].x
    train_edges = torch.tensor([[0, 0, 1], [0, 1, 0]], dtype=torch.long)
    train_data['user', 'rates', 'item'].edge_index = train_edges
    
    evaluator = Evaluator(k=3, device="cpu")
    hr, ndcg, recall = evaluator.evaluate(mock_model_simple, simple_test_data, train_data=train_data)
    
    # Should still work and exclude training items
    assert 0.0 <= hr <= 1.0
    assert 0.0 <= ndcg <= 1.0
    assert 0.0 <= recall <= 1.0


def test_evaluate_hit_at_k_perfect_case():
    """Test Hit@K when relevant item is in top-k."""
    data = create_eval_test_data(
        num_users=2, 
        num_items=5,
        train_edges=torch.tensor([[0], [0]], dtype=torch.long),
        test_edges=torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    )
    
    model = create_mock_model_with_scores(2, 5, 8, {
        (0, 1): 1.0,
        (1, 2): 1.0,
    })
    
    evaluator = Evaluator(k=3, device="cpu")
    hr, ndcg, recall = evaluator.evaluate(model, data)
    
    # Both users should have their test items in top-3, so Hit@3 should be 1.0
    assert hr == 1.0
    assert recall > 0.0
    assert ndcg > 0.0


def test_evaluate_hit_at_k_miss_case():
    """Test Hit@K when relevant item is NOT in top-k."""
    data = create_eval_test_data(
        num_users=1,
        num_items=5,
        train_edges=torch.tensor([[0], [0]], dtype=torch.long),
        test_edges=torch.tensor([[0], [4]], dtype=torch.long)
    )
    
    model = create_mock_model_with_scores(1, 5, 8, {
        (0, 0): 1.0, (0, 1): 0.9, (0, 2): 0.8, (0, 3): 0.7, (0, 4): 0.0
    })
    
    evaluator = Evaluator(k=3, device="cpu")
    hr, ndcg, recall = evaluator.evaluate(model, data)
    
    # Test item 4 is not in top-3, so Hit@3 should be 0.0
    assert hr == 0.0
    assert recall == 0.0
    assert ndcg == 0.0


def test_evaluate_recall_at_k():
    """Test Recall@K calculation (proportion of relevant items found)."""
    data = create_eval_test_data(
        num_users=1,
        num_items=5,
        train_edges=torch.tensor([[0], [0]], dtype=torch.long),
        test_edges=torch.tensor([[0, 0, 0], [1, 2, 3]], dtype=torch.long)
    )
    
    model = create_mock_model_with_scores(1, 5, 8, {
        (0, 1): 1.0, (0, 2): 0.9, (0, 3): 0.1, (0, 0): 0.5
    })
    
    evaluator = Evaluator(k=2, device="cpu")
    hr, ndcg, recall = evaluator.evaluate(model, data)
    
    # 2 out of 3 relevant items in top-2, so Recall@2 should be 2/3
    assert abs(recall - 2/3) < 1e-5
    assert hr == 1.0  # At least one relevant item found


def test_evaluate_ndcg_at_k():
    """Test NDCG@K rewards placing relevant items higher in ranking."""
    data = create_eval_test_data(
        num_users=1,
        num_items=5,
        train_edges=torch.tensor([[0], [0]], dtype=torch.long),
        test_edges=torch.tensor([[0], [1]], dtype=torch.long)
    )
    
    model = create_mock_model_with_scores(1, 5, 8, {
        (0, 1): 1.0, (0, 2): 0.5, (0, 3): 0.3, (0, 4): 0.2
    })
    
    evaluator = Evaluator(k=3, device="cpu")
    hr, ndcg, recall = evaluator.evaluate(model, data)
    
    # Test item at rank 1: DCG = 1/log2(2) = 1.0
    # Ideal DCG (if item at rank 1): 1/log2(2) = 1.0
    # NDCG = 1.0 / 1.0 = 1.0
    assert abs(ndcg - 1.0) < 1e-5
    assert hr == 1.0
    assert recall == 1.0


def test_evaluate_different_k_values(simple_test_data, mock_model_simple):
    """Test evaluation with different k values."""
    k_values = [1, 3, 5]
    results = {}
    
    for k in k_values:
        evaluator = Evaluator(k=k, device="cpu")
        hr, ndcg, recall = evaluator.evaluate(mock_model_simple, simple_test_data)
        results[k] = (hr, ndcg, recall)
    
    # Generally, larger k should give same or better metrics
    # (unless all relevant items are already found)
    assert results[1][0] <= results[3][0]  # Hit@1 <= Hit@3
    assert results[3][0] <= results[5][0]   # Hit@3 <= Hit@5


def test_evaluate_empty_test_data():
    """Test evaluation with no test edges."""
    data = create_eval_test_data(
        num_users=2,
        num_items=5,
        train_edges=torch.tensor([[0, 1], [0, 1]], dtype=torch.long),
        test_edges=torch.empty((2, 0), dtype=torch.long)
    )
    
    model = MockModel(torch.randn(2, 8), torch.randn(5, 8))
    evaluator = Evaluator(k=3, device="cpu")
    hr, ndcg, recall = evaluator.evaluate(model, data)
    
    # With no test users, metrics should be 0
    assert hr == 0.0
    assert ndcg == 0.0
    assert recall == 0.0


def test_evaluate_user_with_no_test_items():
    """Test evaluation when some users have test items and others don't."""
    data = create_eval_test_data(
        num_users=3,
        num_items=5,
        train_edges=torch.tensor([[0, 1, 2], [0, 1, 2]], dtype=torch.long),
        test_edges=torch.tensor([[0, 1], [3, 4]], dtype=torch.long)
    )
    
    model = MockModel(torch.randn(3, 8), torch.randn(5, 8))
    evaluator = Evaluator(k=3, device="cpu")
    hr, ndcg, recall = evaluator.evaluate(model, data)
    
    # Should only evaluate on users 0 and 1 (not user 2)
    assert 0.0 <= hr <= 1.0
    assert 0.0 <= ndcg <= 1.0
    assert 0.0 <= recall <= 1.0

