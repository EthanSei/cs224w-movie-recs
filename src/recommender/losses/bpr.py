import torch
import torch.nn.functional as F

class BPRLoss(torch.nn.Module):
    def __init__(self, reduction: str = "mean"):
        super(BPRLoss, self).__init__()
        if reduction not in ["mean", "sum"]:
            raise ValueError(f"Invalid reduction: {reduction}")
        self.reduction = reduction

    def forward(self, 
        pos_scores: torch.Tensor, # shape: (batch_size,)
        neg_scores: torch.Tensor # shape: (batch_size,)
    ):
        # numerically stable version of -logsigmoid(pos_scores - neg_scores)
        loss = F.softplus(neg_scores - pos_scores)
        if self.reduction == "mean":
            loss_val = loss.mean()
        elif self.reduction == "sum":
            loss_val = loss.sum()
        return loss_val