# tests/losses/bpr_tests.py

import torch
import pytest
from recommender.losses.bpr import BPRLoss
import torch.nn.functional as F

@pytest.mark.parametrize("reduction", ["mean", "sum"])
def test_bpr_loss(reduction):
    loss_fn = BPRLoss(reduction=reduction)
    torch.manual_seed(42)
    pos_scores = torch.randn(16)
    neg_scores = torch.randn(16)
    out = loss_fn(pos_scores, neg_scores)
    assert out.item() > 0.0
    assert out.shape == torch.Size([])

@pytest.mark.parametrize("reduction", ["mean", "sum"])
def test_numerical_stability(reduction):
    loss_fn = BPRLoss(reduction=reduction)
    torch.manual_seed(42)
    pos_scores = torch.tensor([10.0, 10.0, 0.0])
    neg_scores = torch.tensor([0.0, 0.0, 10.0])
    out = loss_fn(pos_scores, neg_scores)
    manual = -F.logsigmoid(pos_scores - neg_scores)
    if reduction == "mean":
        manual = manual.mean()
    elif reduction == "sum":
        manual = manual.sum()
    assert torch.allclose(out, manual, atol=1e-8, rtol=1e-8)