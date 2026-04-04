import torch


def swish_activation(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)
