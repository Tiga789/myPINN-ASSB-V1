"""Torch-based electrochemical helper functions for the _experimental_1 workflow."""
from __future__ import annotations

import torch

from uocp_cs_experimental_1 import (
    uocp_a_fun_x_experimental_1,
    uocp_c_fun_x_experimental_1,
)


def _as_tensor(x: torch.Tensor | float, device: torch.device | None = None) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.to(dtype=torch.float64)
    return torch.as_tensor(x, dtype=torch.float64, device=device)


def uocp_a_fun_experimental_1(cs_a: torch.Tensor | float, csanmax: torch.Tensor | float) -> torch.Tensor:
    cs_a_t = _as_tensor(cs_a)
    csanmax_t = _as_tensor(csanmax, device=cs_a_t.device)
    return uocp_a_fun_x_experimental_1(cs_a_t / csanmax_t)


def uocp_c_fun_experimental_1(cs_c: torch.Tensor | float, cscamax: torch.Tensor | float) -> torch.Tensor:
    cs_c_t = _as_tensor(cs_c)
    cscamax_t = _as_tensor(cscamax, device=cs_c_t.device)
    return uocp_c_fun_x_experimental_1(cs_c_t / cscamax_t)


def i0_a_degradation_param_fun_experimental_1(
    cs_a: torch.Tensor | float,
    ce: torch.Tensor | float,
    T: torch.Tensor | float,
    alpha: torch.Tensor | float,
    csanmax: torch.Tensor | float,
    R: torch.Tensor | float,
    degradation_param: torch.Tensor | float,
) -> torch.Tensor:
    cs_a_t = _as_tensor(cs_a)
    device = cs_a_t.device
    ce_t = _as_tensor(ce, device=device)
    T_t = _as_tensor(T, device=device)
    alpha_t = _as_tensor(alpha, device=device)
    csanmax_t = _as_tensor(csanmax, device=device)
    R_t = _as_tensor(R, device=device)
    degradation_t = _as_tensor(degradation_param, device=device)
    return (
        torch.as_tensor(2.5, dtype=torch.float64, device=device)
        * torch.as_tensor(0.27, dtype=torch.float64, device=device)
        * torch.exp((torch.as_tensor(-30.0e6, dtype=torch.float64, device=device) / R_t) * (torch.as_tensor(1.0, dtype=torch.float64, device=device) / T_t - torch.as_tensor(1.0 / 303.15, dtype=torch.float64, device=device)))
        * torch.clamp(ce_t, min=0.0) ** alpha_t
        * torch.clamp(csanmax_t - cs_a_t, min=0.0) ** alpha_t
        * torch.clamp(cs_a_t, min=0.0) ** (torch.as_tensor(1.0, dtype=torch.float64, device=device) - alpha_t)
        * degradation_t
    )


def i0_c_fun_experimental_1(
    cs_c: torch.Tensor | float,
    ce: torch.Tensor | float,
    T: torch.Tensor | float,
    alpha: torch.Tensor | float,
    cscamax: torch.Tensor | float,
    R: torch.Tensor | float,
) -> torch.Tensor:
    cs_c_t = _as_tensor(cs_c)
    device = cs_c_t.device
    ce_t = _as_tensor(ce, device=device)
    T_t = _as_tensor(T, device=device)
    alpha_t = _as_tensor(alpha, device=device)
    cscamax_t = _as_tensor(cscamax, device=device)
    R_t = _as_tensor(R, device=device)
    x = torch.clamp(cs_c_t / cscamax_t, 0.0, 1.0)
    poly = (
        torch.as_tensor(1.650452829641290e01, dtype=torch.float64, device=device) * x**5
        - torch.as_tensor(7.523567141488800e01, dtype=torch.float64, device=device) * x**4
        + torch.as_tensor(1.240524690073040e02, dtype=torch.float64, device=device) * x**3
        - torch.as_tensor(9.416571081287610e01, dtype=torch.float64, device=device) * x**2
        + torch.as_tensor(3.249768821737960e01, dtype=torch.float64, device=device) * x
        - torch.as_tensor(3.585290065824760e00, dtype=torch.float64, device=device)
    )
    return (
        torch.as_tensor(9.0, dtype=torch.float64, device=device)
        * poly
        * torch.clamp(ce_t / torch.as_tensor(1.2, dtype=torch.float64, device=device), min=0.0) ** alpha_t
        * torch.exp((torch.as_tensor(-30.0e6, dtype=torch.float64, device=device) / R_t) * (torch.as_tensor(1.0, dtype=torch.float64, device=device) / T_t - torch.as_tensor(1.0 / 303.15, dtype=torch.float64, device=device)))
    )


def ds_a_fun_experimental_1(T: torch.Tensor | float, R: torch.Tensor | float) -> torch.Tensor:
    T_t = _as_tensor(T)
    device = T_t.device
    R_t = _as_tensor(R, device=device)
    return torch.as_tensor(3.0e-14, dtype=torch.float64, device=device) * torch.exp((torch.as_tensor(-30.0e6, dtype=torch.float64, device=device) / R_t) * (torch.as_tensor(1.0, dtype=torch.float64, device=device) / T_t - torch.as_tensor(1.0 / 303.15, dtype=torch.float64, device=device)))


def ds_c_degradation_param_fun_experimental_1(
    cs_c: torch.Tensor | float,
    T: torch.Tensor | float,
    R: torch.Tensor | float,
    cscamax: torch.Tensor | float,
    degradation_param: torch.Tensor | float,
) -> torch.Tensor:
    cs_c_t = _as_tensor(cs_c)
    device = cs_c_t.device
    T_t = _as_tensor(T, device=device)
    R_t = _as_tensor(R, device=device)
    cscamax_t = _as_tensor(cscamax, device=device)
    degradation_t = _as_tensor(degradation_param, device=device)
    x = torch.clamp(cs_c_t / cscamax_t, 0.0, 1.0)
    power = (
        -torch.as_tensor(2.509010843479270e02, dtype=torch.float64, device=device) * x**10
        + torch.as_tensor(2.391026725259970e03, dtype=torch.float64, device=device) * x**9
        - torch.as_tensor(4.868420267611360e03, dtype=torch.float64, device=device) * x**8
        - torch.as_tensor(8.331104102921070e01, dtype=torch.float64, device=device) * x**7
        + torch.as_tensor(1.057636028329000e04, dtype=torch.float64, device=device) * x**6
        - torch.as_tensor(1.268324548348120e04, dtype=torch.float64, device=device) * x**5
        + torch.as_tensor(5.016272167775530e03, dtype=torch.float64, device=device) * x**4
        + torch.as_tensor(9.824896659649480e02, dtype=torch.float64, device=device) * x**3
        - torch.as_tensor(1.502439339070900e03, dtype=torch.float64, device=device) * x**2
        + torch.as_tensor(4.723709304247700e02, dtype=torch.float64, device=device) * x
        - torch.as_tensor(6.526092046397090e01, dtype=torch.float64, device=device)
    )
    return (
        torch.as_tensor(1.5, dtype=torch.float64, device=device)
        * (torch.as_tensor(1.5, dtype=torch.float64, device=device) * torch.pow(torch.as_tensor(10.0, dtype=torch.float64, device=device), power))
        * torch.exp((torch.as_tensor(-30.0e6, dtype=torch.float64, device=device) / R_t) * (torch.as_tensor(1.0, dtype=torch.float64, device=device) / T_t - torch.as_tensor(1.0 / 303.15, dtype=torch.float64, device=device)))
        * degradation_t
    )


def grad_ds_c_cs_c_experimental_1(
    cs_c: torch.Tensor | float,
    T: torch.Tensor | float,
    R: torch.Tensor | float,
    cscamax: torch.Tensor | float,
    degradation_param: torch.Tensor | float,
) -> torch.Tensor:
    cs_c_t = _as_tensor(cs_c)
    device = cs_c_t.device
    T_t = _as_tensor(T, device=device)
    R_t = _as_tensor(R, device=device)
    cscamax_t = _as_tensor(cscamax, device=device)
    degradation_t = _as_tensor(degradation_param, device=device)
    return (
        torch.as_tensor(2.25, dtype=torch.float64, device=device)
        * torch.pow(
            torch.as_tensor(10.0, dtype=torch.float64, device=device),
            -torch.as_tensor(250.901084347927, dtype=torch.float64, device=device) * cs_c_t**10 / cscamax_t**10
            + torch.as_tensor(2391.02672525997, dtype=torch.float64, device=device) * cs_c_t**9 / cscamax_t**9
            - torch.as_tensor(4868.42026761136, dtype=torch.float64, device=device) * cs_c_t**8 / cscamax_t**8
            - torch.as_tensor(83.3110410292107, dtype=torch.float64, device=device) * cs_c_t**7 / cscamax_t**7
            + torch.as_tensor(10576.36028329, dtype=torch.float64, device=device) * cs_c_t**6 / cscamax_t**6
            - torch.as_tensor(12683.2454834812, dtype=torch.float64, device=device) * cs_c_t**5 / cscamax_t**5
            + torch.as_tensor(5016.27216777553, dtype=torch.float64, device=device) * cs_c_t**4 / cscamax_t**4
            + torch.as_tensor(982.489665964948, dtype=torch.float64, device=device) * cs_c_t**3 / cscamax_t**3
            - torch.as_tensor(1502.4393390709, dtype=torch.float64, device=device) * cs_c_t**2 / cscamax_t**2
            + torch.as_tensor(472.37093042477, dtype=torch.float64, device=device) * cs_c_t / cscamax_t
            - torch.as_tensor(65.2609204639709, dtype=torch.float64, device=device)
        )
        * (
            -torch.as_tensor(5777.21096635578, dtype=torch.float64, device=device) * cs_c_t**9 / cscamax_t**10
            + torch.as_tensor(49549.8824508058, dtype=torch.float64, device=device) * cs_c_t**8 / cscamax_t**9
            - torch.as_tensor(89679.615477056, dtype=torch.float64, device=device) * cs_c_t**7 / cscamax_t**8
            - torch.as_tensor(1342.81532808973, dtype=torch.float64, device=device) * cs_c_t**6 / cscamax_t**7
            + torch.as_tensor(146117.817158627, dtype=torch.float64, device=device) * cs_c_t**5 / cscamax_t**6
            - torch.as_tensor(146021.259905239, dtype=torch.float64, device=device) * cs_c_t**4 / cscamax_t**5
            + torch.as_tensor(46201.5740636835, dtype=torch.float64, device=device) * cs_c_t**3 / cscamax_t**4
            + torch.as_tensor(6786.79817661477, dtype=torch.float64, device=device) * cs_c_t**2 / cscamax_t**3
            - torch.as_tensor(6918.98885054496, dtype=torch.float64, device=device) * cs_c_t / cscamax_t**2
            + torch.as_tensor(1087.6742627598, dtype=torch.float64, device=device) / cscamax_t
        )
        * torch.exp(torch.as_tensor(-30000000.0, dtype=torch.float64, device=device) * (torch.as_tensor(-0.0032986970146792, dtype=torch.float64, device=device) + torch.as_tensor(1.0, dtype=torch.float64, device=device) / T_t) / R_t)
        * degradation_t
    )


def phie_linearized_experimental_1(i0_a: torch.Tensor | float, j_a: torch.Tensor | float, R: torch.Tensor | float, T: torch.Tensor | float, Uocp_a0: torch.Tensor | float) -> torch.Tensor:
    i0_a_t = _as_tensor(i0_a)
    device = i0_a_t.device
    j_a_t = _as_tensor(j_a, device=device)
    R_t = _as_tensor(R, device=device)
    T_t = _as_tensor(T, device=device)
    U_t = _as_tensor(Uocp_a0, device=device)
    return -j_a_t * (R_t * T_t / torch.clamp(i0_a_t, min=1e-30)) - U_t


def phis_c_linearized_experimental_1(i0_c: torch.Tensor | float, j_c: torch.Tensor | float, R: torch.Tensor | float, T: torch.Tensor | float, Uocp_c0: torch.Tensor | float, phie0: torch.Tensor | float) -> torch.Tensor:
    i0_c_t = _as_tensor(i0_c)
    device = i0_c_t.device
    j_c_t = _as_tensor(j_c, device=device)
    R_t = _as_tensor(R, device=device)
    T_t = _as_tensor(T, device=device)
    U_t = _as_tensor(Uocp_c0, device=device)
    phie_t = _as_tensor(phie0, device=device)
    return j_c_t * (R_t * T_t / torch.clamp(i0_c_t, min=1e-30)) + U_t + phie_t
