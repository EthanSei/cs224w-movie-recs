import torch

class TypeProjector(torch.nn.Module):
    def __init__(self, in_dims: dict[str, int], out_dim: int):
        super(TypeProjector, self).__init__()
        self.projector = torch.nn.ModuleDict({
            k: torch.nn.Linear(v, out_dim) for k, v in in_dims.items()
        })

    def forward(self, x_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {k: self.projector[k](v) for k, v in x_dict.items()}


class DotProductHead(torch.nn.Module):
    """
    Simple dot product head for recommendation systems.
    Computes the dot product between user and item embeddings.
    """
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return (x * y).sum(dim=1)


class WeightedDotProductHead(torch.nn.Module):
    """
    Weighted dot product head for recommendation systems.
    Computes the weighted dot product between two embeddings.
    Adds learnable parameters but may overfit on smaller datasets.
    """
    def __init__(self, x_dim: int, y_dim: int):
        super(WeightedDotProductHead, self).__init__()
        self.weight = torch.nn.Linear(x_dim, y_dim)
    
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        xW = self.weight(x)
        return (xW * y).sum(dim=1)