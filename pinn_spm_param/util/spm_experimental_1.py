"""Torch-based parameter block for the _experimental_1 workflow."""
from __future__ import annotations

import torch

from thermo_experimental_1 import (
    ds_a_fun_experimental_1,
    ds_c_degradation_param_fun_experimental_1,
    i0_a_degradation_param_fun_experimental_1,
    i0_c_fun_experimental_1,
    uocp_a_fun_experimental_1,
    uocp_c_fun_experimental_1,
)


def make_params_experimental_1(device: str | None = None, dtype: torch.dtype = torch.float64) -> dict:
    resolved_device = torch.device(device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu"))
    params: dict[str, object] = {}
    params["device"] = resolved_device
    params["dtype"] = dtype

    def t(value: float) -> torch.Tensor:
        return torch.as_tensor(value, dtype=dtype, device=resolved_device)

    params["F"] = t(96485.3321e3)
    params["R"] = t(8.3145e3)
    params["T"] = t(303.15)
    params["A_a"] = t(1.4e-3)
    params["A_c"] = t(1.4e-3)
    params["alpha_a"] = t(0.5)
    params["alpha_c"] = t(0.5)
    params["Rs_a"] = t(8e-6 / 2.0)
    params["Rs_c"] = t(3.6e-6 / 2.0)
    params["csanmax"] = t(30.53)
    params["cscamax"] = t(49.6)
    params["eps_s_a"] = t(0.5430727763)
    params["eps_s_c"] = t(0.47662)
    params["L_a"] = t(44e-6)
    params["L_c"] = t(42e-6)
    params["ce0"] = t(1.2)

    params["default_deg_i0_a"] = t(0.5000000000001683)
    params["default_deg_ds_c"] = t(1.0000000000000002)
    params["default_x_a0"] = t(0.00100000253293656)
    params["default_x_c0"] = t(0.4634028432597517)

    params["Uocp_a"] = uocp_a_fun_experimental_1
    params["Uocp_c"] = uocp_c_fun_experimental_1
    params["i0_a"] = i0_a_degradation_param_fun_experimental_1
    params["i0_c"] = i0_c_fun_experimental_1
    params["D_s_a"] = ds_a_fun_experimental_1
    params["D_s_c"] = ds_c_degradation_param_fun_experimental_1
    return params


def current_to_flux_experimental_1(current_a: torch.Tensor | float, params: dict) -> tuple[torch.Tensor, torch.Tensor]:
    device = params["device"]
    dtype = params["dtype"]
    current_t = torch.as_tensor(current_a, dtype=dtype, device=device)
    j_a = (
        -(current_t / params["A_a"])
        * params["Rs_a"]
        / (torch.as_tensor(3.0, dtype=dtype, device=device) * params["eps_s_a"] * params["F"] * params["L_a"])
    )
    j_c = (
        (current_t / params["A_c"])
        * params["Rs_c"]
        / (torch.as_tensor(3.0, dtype=dtype, device=device) * params["eps_s_c"] * params["F"] * params["L_c"])
    )
    return j_a, j_c
