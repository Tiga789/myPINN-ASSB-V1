from __future__ import annotations

import math

import torch
import torch.nn as nn


def logit_from_fraction_ASSBfinal2(x: float, eps: float = 1e-6) -> float:
    x_clip = min(max(float(x), eps), 1.0 - eps)
    return math.log(x_clip / (1.0 - x_clip))


class RadiusFeatures_ASSBfinal2(nn.Module):
    def __init__(self, n_modes: int = 6):
        super().__init__()
        self.n_modes = int(max(0, n_modes))

    @property
    def out_dim(self) -> int:
        return 1 + 2 * self.n_modes

    def forward(self, rho: torch.Tensor) -> torch.Tensor:
        rho = rho.reshape(-1, 1)
        feats = [rho]
        if self.n_modes > 0:
            for i in range(self.n_modes):
                freq = float(2**i) * math.pi
                feats.append(torch.sin(freq * rho))
                feats.append(torch.cos(freq * rho))
        return torch.cat(feats, dim=1)
