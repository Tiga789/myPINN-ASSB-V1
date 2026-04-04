import math
from typing import Iterable, List, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


torch.set_default_dtype(torch.float64)


def default_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def to_tensor(value, device: torch.device | None = None) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        tensor = value.to(dtype=torch.float64)
        if device is not None:
            tensor = tensor.to(device)
        return tensor
    tensor = torch.as_tensor(value, dtype=torch.float64)
    if device is not None:
        tensor = tensor.to(device)
    return tensor


def ensure_2d(value, device: torch.device | None = None) -> torch.Tensor:
    tensor = to_tensor(value, device=device)
    if tensor.ndim == 0:
        tensor = tensor.reshape(1, 1)
    elif tensor.ndim == 1:
        tensor = tensor.reshape(-1, 1)
    return tensor


def scalar_like(value: float, ref: torch.Tensor) -> torch.Tensor:
    return torch.full_like(ref, float(value), dtype=torch.float64)


def zeros_like(ref: torch.Tensor) -> torch.Tensor:
    return torch.zeros_like(ref, dtype=torch.float64)


def ones_like(ref: torch.Tensor) -> torch.Tensor:
    return torch.ones_like(ref, dtype=torch.float64)


def clip(value, vmin, vmax):
    tensor = to_tensor(value)
    return torch.clamp(tensor, min=float(vmin), max=float(vmax))


def safe_mean_square(term: torch.Tensor) -> torch.Tensor:
    if term.numel() == 0:
        return torch.zeros((), dtype=torch.float64, device=term.device)
    return torch.mean(torch.square(term))


def grad(outputs: torch.Tensor, inputs: torch.Tensor, create_graph: bool = True) -> torch.Tensor:
    result = torch.autograd.grad(
        outputs,
        inputs,
        grad_outputs=torch.ones_like(outputs),
        create_graph=create_graph,
        retain_graph=True,
        allow_unused=True,
    )[0]
    if result is None:
        result = torch.zeros_like(inputs)
    return result


def polyval(coeffs: Sequence[float], x) -> torch.Tensor:
    xt = to_tensor(x)
    out = torch.zeros_like(xt, dtype=torch.float64)
    for coeff in coeffs:
        out = out * xt + float(coeff)
    return out


def tensor_to_numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def detach_list(values: Sequence[torch.Tensor]) -> List[np.ndarray]:
    return [tensor_to_numpy(v) for v in values]


def kaiming_init(module: nn.Module) -> None:
    if isinstance(module, nn.Linear):
        nn.init.kaiming_normal_(module.weight, nonlinearity="linear")
        if module.bias is not None:
            nn.init.kaiming_normal_(module.bias.unsqueeze(0), nonlinearity="linear")


class ActivationModule(nn.Module):
    def __init__(self, name: str):
        super().__init__()
        self.name = name.lower()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.name == "swish":
            return x * torch.sigmoid(x)
        if self.name == "sigmoid":
            return torch.sigmoid(x)
        if self.name == "tanh":
            return torch.tanh(x)
        if self.name == "elu":
            return F.elu(x)
        if self.name == "selu":
            return F.selu(x)
        if self.name == "gelu":
            return F.gelu(x)
        raise ValueError(f"Unsupported activation: {self.name}")


def build_mlp(
    in_features: int,
    hidden_units: Sequence[int] | None,
    activation: str,
    out_features: int | None = None,
) -> nn.Sequential:
    hidden_units = list(hidden_units or [])
    layers: list[nn.Module] = []
    current = in_features
    for width in hidden_units:
        layers.append(nn.Linear(current, int(width)))
        layers.append(ActivationModule(activation))
        current = int(width)
    if out_features is not None:
        layers.append(nn.Linear(current, int(out_features)))
    seq = nn.Sequential(*layers)
    seq.apply(kaiming_init)
    return seq


class ResidualBlock(nn.Module):
    def __init__(self, width: int, n_layers: int, activation: str):
        super().__init__()
        blocks: list[nn.Module] = []
        for _ in range(int(n_layers)):
            blocks.append(nn.Linear(int(width), int(width)))
            blocks.append(ActivationModule(activation))
        self.block = nn.Sequential(*blocks)
        self.activation = ActivationModule(activation)
        self.apply(kaiming_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x + self.block(x))


class PreReshape(nn.Module):
    """Match feature width before residual blocks."""

    def __init__(self, in_features: int, out_features: int, activation: str):
        super().__init__()
        if int(in_features) == int(out_features):
            self.layer = nn.Identity()
        else:
            self.layer = nn.Sequential(
                nn.Linear(int(in_features), int(out_features)),
                ActivationModule(activation),
            )
            self.layer.apply(kaiming_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer(x)


class GradientPathBlock(nn.Module):
    def __init__(self, in_features: int, n_blocks: int, n_units: int, activation: str):
        super().__init__()
        self.n_blocks = int(n_blocks)
        self.U = nn.Sequential(
            nn.Linear(int(in_features), int(n_units)),
            ActivationModule(activation),
        )
        self.V = nn.Sequential(
            nn.Linear(int(in_features), int(n_units)),
            ActivationModule(activation),
        )
        self.H0 = nn.Sequential(
            nn.Linear(int(in_features), int(n_units)),
            ActivationModule(activation),
        )
        self.Z = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(int(n_units), int(n_units)),
                    ActivationModule(activation),
                )
                for _ in range(max(int(n_blocks) - 1, 0))
            ]
        )
        self.apply(kaiming_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = self.U(x)
        v = self.V(x)
        h = self.H0(x)
        for z_layer in self.Z:
            z = z_layer(h)
            h = (1.0 - z) * u + z * v
        return h
