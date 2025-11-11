import torch
import torch.nn.functional as F

class BPRLoss:
    def __init__(self, weight: float = 1.0, reduction: str = "mean"):
        self.weight = weight
        self.reduction = reduction

    def loss_fn(self, outputs: dict, batch):
        pos = outputs["pos_score"]
        neg = outputs["neg_score"]

        if neg.dim() == 2:
            loss = -F.logsigmoid((pos.unsqueeze(1) - neg)).mean(dim=1)
        else:
            loss = -F.logsigmoid(pos - neg)

        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()
        else:
            raise ValueError(f"Invalid reduction: {self.reduction}")

        return loss * self.weight