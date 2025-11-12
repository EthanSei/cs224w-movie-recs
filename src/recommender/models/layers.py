import torch

class TypeProjector(torch.nn.Module):
    def __init__(self, in_dims: dict[str, int], out_dim: int):
        super(TypeProjector, self).__init__()
        self.projector = torch.nn.ModuleDict({
            k: torch.nn.Linear(v, out_dim) for k, v in in_dims.items()
        })

    def forward(self, x_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {k: self.projector[k](v) for k, v in x_dict.items()}