from __future__ import annotations

from typing import Iterable

import torch
import torch.nn as nn

from _rescale_ASSBfinal2 import RadiusFeatures_ASSBfinal2, logit_from_fraction_ASSBfinal2


def _make_mlp(in_dim: int, hidden_dim: int, num_layers: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    last = in_dim
    for _ in range(int(max(1, num_layers))):
        layers.append(nn.Linear(last, hidden_dim))
        layers.append(nn.SiLU())
        last = hidden_dim
    layers.append(nn.Linear(last, 1))
    return nn.Sequential(*layers)


class StepLatentFieldModel_ASSBfinal2(nn.Module):
    """
    Physics-only surrogate for a *fixed* current profile.

    The key lesson from ASSBfinal1 is that the teacher problem is a *discrete marching*
    problem, not a random continuous-collocation problem. This model therefore keeps one
    trainable latent code per *discrete time node* (except the known initial node), and a
    shared radial decoder. The initial condition is enforced exactly by construction.
    """

    def __init__(
        self,
        n_time_nodes: int,
        latent_dim: int,
        hidden_dim: int,
        num_layers: int,
        r_fourier_modes: int,
        x_a0: float,
        x_c0: float,
        csanmax: float,
        cscamax: float,
        dtype: torch.dtype = torch.float64,
    ) -> None:
        super().__init__()
        if n_time_nodes < 2:
            raise ValueError("Need at least two time nodes.")
        self.n_time_nodes = int(n_time_nodes)
        self.n_trainable_time_nodes = int(n_time_nodes - 1)  # t=0 is exact and fixed.
        self.latent_dim = int(latent_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.csanmax = float(csanmax)
        self.cscamax = float(cscamax)
        self.base_logit_a = float(logit_from_fraction_ASSBfinal2(x_a0))
        self.base_logit_c = float(logit_from_fraction_ASSBfinal2(x_c0))

        self.time_embedding = nn.Embedding(self.n_trainable_time_nodes, self.latent_dim)
        self.radius_features = RadiusFeatures_ASSBfinal2(n_modes=r_fourier_modes)
        in_dim = self.latent_dim + self.radius_features.out_dim
        self.decoder_a = _make_mlp(in_dim, self.hidden_dim, self.num_layers)
        self.decoder_c = _make_mlp(in_dim, self.hidden_dim, self.num_layers)

        self._dtype = dtype
        self.to(dtype=dtype)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.zeros_(self.time_embedding.weight)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        # Start close to the exact initial condition for every time node.
        final_a = self.decoder_a[-1]
        final_c = self.decoder_c[-1]
        assert isinstance(final_a, nn.Linear)
        assert isinstance(final_c, nn.Linear)
        nn.init.zeros_(final_a.weight)
        nn.init.zeros_(final_c.weight)
        final_a.bias.data.fill_(0.0)
        final_c.bias.data.fill_(0.0)

    def _decode(self, time_idx: torch.Tensor, rho: torch.Tensor, decoder: nn.Sequential, base_logit: float, csmax: float, x0: float) -> torch.Tensor:
        time_idx = time_idx.reshape(-1).long()
        rho = rho.reshape(-1).to(dtype=self._dtype, device=time_idx.device)
        out = torch.empty((time_idx.numel(),), dtype=self._dtype, device=time_idx.device)

        zero_mask = time_idx == 0
        if torch.any(zero_mask):
            out[zero_mask] = float(x0) * float(csmax)

        nonzero_mask = ~zero_mask
        if torch.any(nonzero_mask):
            emb = self.time_embedding(time_idx[nonzero_mask] - 1)
            r_feat = self.radius_features(rho[nonzero_mask])
            raw = decoder(torch.cat([emb, r_feat], dim=1)).reshape(-1)
            logits = raw + float(base_logit)
            theta = torch.sigmoid(logits)
            out[nonzero_mask] = theta * float(csmax)
        return out

    def predict_cs_a_flat(self, time_idx: torch.Tensor, rho: torch.Tensor, x_a0: float) -> torch.Tensor:
        return self._decode(time_idx, rho, self.decoder_a, self.base_logit_a, self.csanmax, x_a0)

    def predict_cs_c_flat(self, time_idx: torch.Tensor, rho: torch.Tensor, x_c0: float) -> torch.Tensor:
        return self._decode(time_idx, rho, self.decoder_c, self.base_logit_c, self.cscamax, x_c0)

    def predict_profiles(self, time_idx: torch.Tensor, rho_grid: torch.Tensor, electrode: str, x0: float) -> torch.Tensor:
        time_idx = time_idx.reshape(-1).long()
        rho_grid = rho_grid.reshape(-1)
        n_t = time_idx.numel()
        n_r = rho_grid.numel()
        idx_flat = time_idx.repeat_interleave(n_r)
        rho_flat = rho_grid.repeat(n_t)
        if electrode == "a":
            values = self.predict_cs_a_flat(idx_flat, rho_flat, x0)
        elif electrode == "c":
            values = self.predict_cs_c_flat(idx_flat, rho_flat, x0)
        else:
            raise ValueError(electrode)
        return values.reshape(n_t, n_r)

    def count_trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
