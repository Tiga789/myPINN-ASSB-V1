"""Torch-based OCP helper functions for the _experimental_1 workflow.

This file is independent from the original repository and keeps the same
polynomial OCP definitions used in the earlier experimental scripts.
"""
from __future__ import annotations

from typing import Iterable

import torch

UOCP_A_COEFFS = [
    1878.6244900261463,
    -4981.580023016213,
    516.2941996957871,
    6452.38177755237,
    -436.0524457974526,
    1264.0514576769442,
    -20918.656956191975,
    12954.334261316431,
    28871.72866007402,
    -37943.83286204571,
    34.11141793217983,
    29363.16490602074,
    -25774.496334571464,
    11073.868226559767,
    -2702.638445370805,
    375.62895901410747,
    -28.064663950113868,
    1.1265244540945243,
]

UOCP_C_COEFFS = [
    -43309.69063512314,
    122888.63938515769,
    -69735.99554716503,
    -59749.183217994185,
    25744.002733171154,
    15730.398058573825,
    54021.915506318735,
    -44566.03206954511,
    64.32177924593454,
    -7780.173422833786,
    1117.4042221859695,
    7387.492376558274,
    -7237.289515884936,
    -705.4465901574707,
    17170.20236584321,
    -42.60228181558803,
    -23266.56994359366,
    10810.92851132453,
    2545.4065429021307,
    1.6554268823619098,
    751.3515882152476,
    -4447.12851190078,
    3727.268889820381,
    -1331.1791971457515,
    227.4712483170547,
    -17.646894926746256,
    0.8568207255402533,
    -2.34505930698951,
    5.059010555584711,
]


def _as_tensor(x: torch.Tensor | float, device: torch.device | None = None) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x
    return torch.as_tensor(x, dtype=torch.float64, device=device)


def _polyval_torch(coeffs: Iterable[float], x: torch.Tensor) -> torch.Tensor:
    out = torch.zeros_like(x, dtype=torch.float64)
    for coeff in coeffs:
        out = out * x + torch.as_tensor(coeff, dtype=torch.float64, device=x.device)
    return out


def uocp_a_fun_x_experimental_1(x: torch.Tensor | float) -> torch.Tensor:
    x_t = torch.clamp(_as_tensor(x), 0.0, 1.0).to(dtype=torch.float64)
    return _polyval_torch(UOCP_A_COEFFS, x_t)


def uocp_c_fun_x_experimental_1(x: torch.Tensor | float) -> torch.Tensor:
    x_t = torch.clamp(_as_tensor(x), 0.0, 1.0).to(dtype=torch.float64)
    return _polyval_torch(UOCP_C_COEFFS, x_t)
