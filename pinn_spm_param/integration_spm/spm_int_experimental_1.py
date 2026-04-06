"""Torch-based current-driven SPM integrator for the _experimental_1 workflow.

The physical state variables remain the same as the original PINNSTRIPES SPM:
cs_a(r, t), cs_c(r, t), phie(t), phis_c(t).
"""
from __future__ import annotations

import torch

from spm_experimental_1 import current_to_flux_experimental_1
from thermo_experimental_1 import (
    grad_ds_c_cs_c_experimental_1,
    phie_linearized_experimental_1,
    phis_c_linearized_experimental_1,
)


def _gradient_1d(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    grad = torch.zeros_like(y)
    if y.numel() < 2:
        return grad
    grad[0] = (y[1] - y[0]) / torch.clamp(x[1] - x[0], min=1e-30)
    grad[-1] = (y[-1] - y[-2]) / torch.clamp(x[-1] - x[-2], min=1e-30)
    if y.numel() > 2:
        grad[1:-1] = (y[2:] - y[:-2]) / torch.clamp(x[2:] - x[:-2], min=1e-30)
    return grad


def tridiag_experimental_1(ds: torch.Tensor, dt: torch.Tensor, dr: torch.Tensor) -> torch.Tensor:
    ds_t = ds.to(dtype=torch.float64)
    dt_t = torch.as_tensor(dt, dtype=torch.float64, device=ds_t.device)
    dr_t = torch.as_tensor(dr, dtype=torch.float64, device=ds_t.device)
    n_r = ds_t.numel()
    a = torch.as_tensor(1.0, dtype=torch.float64, device=ds_t.device) + torch.as_tensor(2.0, dtype=torch.float64, device=ds_t.device) * ds_t * dt_t / (dr_t**2)
    b = -ds_t * dt_t / (dr_t**2)
    mat = torch.zeros((n_r, n_r), dtype=torch.float64, device=ds_t.device)
    idx = torch.arange(n_r, device=ds_t.device)
    mat[idx, idx] = a
    if n_r > 1:
        mat[idx[1:], idx[:-1]] = b[1:]
        mat[idx[:-1], idx[1:]] = b[:-1]
    mat[0, :] = torch.as_tensor(0.0, dtype=torch.float64, device=ds_t.device)
    mat[-1, :] = torch.as_tensor(0.0, dtype=torch.float64, device=ds_t.device)
    mat[0, 0] = -torch.as_tensor(1.0, dtype=torch.float64, device=ds_t.device) / dr_t
    mat[0, 1] = torch.as_tensor(1.0, dtype=torch.float64, device=ds_t.device) / dr_t
    mat[-1, -2] = -torch.as_tensor(1.0, dtype=torch.float64, device=ds_t.device) / dr_t
    mat[-1, -1] = torch.as_tensor(1.0, dtype=torch.float64, device=ds_t.device) / dr_t
    return mat


def rhs_experimental_1(
    dt: torch.Tensor,
    r: torch.Tensor,
    ddr_cs: torch.Tensor,
    ds: torch.Tensor,
    ddDs_cs: torch.Tensor,
    cs: torch.Tensor,
    bound_grad: torch.Tensor,
) -> torch.Tensor:
    dt_t = torch.as_tensor(dt, dtype=torch.float64, device=r.device)
    safe_r = torch.clamp(r, min=1e-12)
    rhs_col = dt_t * (torch.as_tensor(2.0, dtype=torch.float64, device=r.device) / safe_r) * ddr_cs * ds
    rhs_col = rhs_col + dt_t * ddr_cs**2 * ddDs_cs
    rhs_col = rhs_col + cs
    rhs_col[0] = torch.as_tensor(0.0, dtype=torch.float64, device=r.device)
    rhs_col[-1] = torch.as_tensor(bound_grad, dtype=torch.float64, device=r.device)
    return rhs_col


def simulate_current_profile_experimental_1(
    time_s,
    current_a,
    params: dict,
    deg_i0_a=None,
    deg_ds_c=None,
    x_a0=None,
    x_c0=None,
    n_r: int = 12,
) -> dict:
    device = params["device"]
    dtype = params["dtype"]
    time_t = torch.as_tensor(time_s, dtype=dtype, device=device)
    current_t = torch.as_tensor(current_a, dtype=dtype, device=device)
    if time_t.ndim != 1 or current_t.ndim != 1 or time_t.numel() != current_t.numel():
        raise ValueError("time_s and current_a must be one-dimensional arrays with the same length.")
    if time_t.numel() < 2:
        raise ValueError("Need at least two time points.")
    if bool(torch.any(time_t[1:] < time_t[:-1]).item()):
        raise ValueError("time_s must be nondecreasing.")

    deg_i0_a_t = torch.as_tensor(params["default_deg_i0_a"] if deg_i0_a is None else deg_i0_a, dtype=dtype, device=device)
    deg_ds_c_t = torch.as_tensor(params["default_deg_ds_c"] if deg_ds_c is None else deg_ds_c, dtype=dtype, device=device)
    x_a0_t = torch.as_tensor(params["default_x_a0"] if x_a0 is None else x_a0, dtype=dtype, device=device)
    x_c0_t = torch.as_tensor(params["default_x_c0"] if x_c0 is None else x_c0, dtype=dtype, device=device)

    n_t = int(time_t.numel())
    dt = time_t[1:] - time_t[:-1]

    r_a = torch.linspace(torch.as_tensor(0.0, dtype=dtype, device=device), params["Rs_a"], n_r, device=device, dtype=dtype)
    r_c = torch.linspace(torch.as_tensor(0.0, dtype=dtype, device=device), params["Rs_c"], n_r, device=device, dtype=dtype)
    dR_a = params["Rs_a"] / torch.as_tensor(n_r - 1, dtype=dtype, device=device)
    dR_c = params["Rs_c"] / torch.as_tensor(n_r - 1, dtype=dtype, device=device)

    cs_a = torch.zeros((n_t, n_r), dtype=dtype, device=device)
    cs_c = torch.zeros((n_t, n_r), dtype=dtype, device=device)
    phie = torch.zeros(n_t, dtype=dtype, device=device)
    phis_c = torch.zeros(n_t, dtype=dtype, device=device)

    with torch.no_grad():
        cs_a[0, :] = x_a0_t * params["csanmax"]
        cs_c[0, :] = x_c0_t * params["cscamax"]
        ce = params["ce0"]
        ds_a_const = params["D_s_a"](params["T"], params["R"])

        j_a0, j_c0 = current_to_flux_experimental_1(current_t[0], params)
        cse_a0 = cs_a[0, -1]
        cse_c0 = cs_c[0, -1]
        i0_a0 = params["i0_a"](cse_a0, ce, params["T"], params["alpha_a"], params["csanmax"], params["R"], deg_i0_a_t)
        i0_c0 = params["i0_c"](cse_c0, ce, params["T"], params["alpha_c"], params["cscamax"], params["R"])
        Uocp_a0 = params["Uocp_a"](cse_a0, params["csanmax"])
        Uocp_c0 = params["Uocp_c"](cse_c0, params["cscamax"])
        phie[0] = phie_linearized_experimental_1(i0_a0, j_a0, params["R"], params["T"], Uocp_a0)
        phis_c[0] = phis_c_linearized_experimental_1(i0_c0, j_c0, params["R"], params["T"], Uocp_c0, phie[0])

        for i_t in range(1, n_t):
            j_a, j_c = current_to_flux_experimental_1(current_t[i_t - 1], params)
            cse_a = cs_a[i_t - 1, -1]
            cse_c = cs_c[i_t - 1, -1]
            i0_a = params["i0_a"](cse_a, ce, params["T"], params["alpha_a"], params["csanmax"], params["R"], deg_i0_a_t)
            i0_c = params["i0_c"](cse_c, ce, params["T"], params["alpha_c"], params["cscamax"], params["R"])
            Uocp_a = params["Uocp_a"](cse_a, params["csanmax"])
            Uocp_c = params["Uocp_c"](cse_c, params["cscamax"])
            phie[i_t] = phie_linearized_experimental_1(i0_a, j_a, params["R"], params["T"], Uocp_a)
            phis_c[i_t] = phis_c_linearized_experimental_1(i0_c, j_c, params["R"], params["T"], Uocp_c, phie[i_t])

            if dt[i_t - 1] <= 0:
                cs_a[i_t, :] = cs_a[i_t - 1, :]
                cs_c[i_t, :] = cs_c[i_t - 1, :]
                continue

            Ds_a = torch.full((n_r,), ds_a_const.item(), dtype=dtype, device=device)
            grad_cs_a = _gradient_1d(cs_a[i_t - 1, :], r_a)
            A_a = tridiag_experimental_1(Ds_a, dt[i_t - 1], dR_a)
            B_a = rhs_experimental_1(
                dt[i_t - 1],
                r_a,
                grad_cs_a,
                Ds_a,
                torch.zeros(n_r, dtype=dtype, device=device),
                cs_a[i_t - 1, :],
                -j_a / torch.clamp(Ds_a[-1], min=1e-30),
            )
            cs_a[i_t, :] = torch.clamp(torch.linalg.solve(A_a, B_a), min=0.0, max=params["csanmax"].item())

            Ds_c = params["D_s_c"](cs_c[i_t - 1, :], params["T"], params["R"], params["cscamax"], deg_ds_c_t)
            grad_Ds_c = grad_ds_c_cs_c_experimental_1(cs_c[i_t - 1, :], params["T"], params["R"], params["cscamax"], deg_ds_c_t)
            grad_cs_c = _gradient_1d(cs_c[i_t - 1, :], r_c)
            A_c = tridiag_experimental_1(Ds_c, dt[i_t - 1], dR_c)
            B_c = rhs_experimental_1(
                dt[i_t - 1],
                r_c,
                grad_cs_c,
                Ds_c,
                grad_Ds_c,
                cs_c[i_t - 1, :],
                -j_c / torch.clamp(Ds_c[-1], min=1e-30),
            )
            cs_c[i_t, :] = torch.clamp(torch.linalg.solve(A_c, B_c), min=0.0, max=params["cscamax"].item())

    return {
        "t": time_t.detach().cpu().numpy(),
        "r_a": r_a.detach().cpu().numpy(),
        "r_c": r_c.detach().cpu().numpy(),
        "cs_a": cs_a.detach().cpu().numpy(),
        "cs_c": cs_c.detach().cpu().numpy(),
        "phie": phie.detach().cpu().numpy(),
        "phis_c": phis_c.detach().cpu().numpy(),
        "voltage": phis_c.detach().cpu().numpy(),
        "device": str(device),
    }
