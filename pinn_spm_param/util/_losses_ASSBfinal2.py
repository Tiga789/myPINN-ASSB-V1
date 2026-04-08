from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch


def gradient_batch_1d_ASSBfinal2(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    grad = torch.zeros_like(y)
    x1 = torch.as_tensor(x, dtype=y.dtype, device=y.device).reshape(-1)
    n = y.shape[1]
    if n < 2:
        return grad
    dx0 = torch.clamp(x1[1] - x1[0], min=1e-30)
    dxl = torch.clamp(x1[-1] - x1[-2], min=1e-30)
    grad[:, 0] = (y[:, 1] - y[:, 0]) / dx0
    grad[:, -1] = (y[:, -1] - y[:, -2]) / dxl
    if n > 2:
        denom = torch.clamp(x1[2:] - x1[:-2], min=1e-30).reshape(1, -1)
        grad[:, 1:-1] = (y[:, 2:] - y[:, :-2]) / denom
    return grad


@dataclass
class LossBreakdown_ASSBfinal2:
    total: torch.Tensor
    step_a: torch.Tensor
    step_c: torch.Tensor
    bound_a: torch.Tensor
    bound_c: torch.Tensor

    def as_float_dict(self) -> dict[str, float]:
        return {
            "total": float(self.total.detach().cpu()),
            "step_a": float(self.step_a.detach().cpu()),
            "step_c": float(self.step_c.detach().cpu()),
            "bound_a": float(self.bound_a.detach().cpu()),
            "bound_c": float(self.bound_c.detach().cpu()),
        }


def _normalize_residual_ASSBfinal2(res: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    denom = torch.clamp(torch.as_tensor(scale, dtype=res.dtype, device=res.device), min=1e-12)
    return res / denom


def compute_batch_loss_ASSBfinal2(
    model,
    step_idx: torch.Tensor,
    ctx: dict[str, Any],
    cfg: dict[str, Any],
) -> LossBreakdown_ASSBfinal2:
    device = ctx["device"]
    dtype = ctx["dtype"]
    step_idx = step_idx.reshape(-1).long().to(device)
    if step_idx.numel() == 0:
        z = torch.zeros((), dtype=dtype, device=device)
        return LossBreakdown_ASSBfinal2(z, z, z, z, z)

    next_idx = step_idx + 1
    dt = ctx["dt_s"][step_idx].reshape(-1, 1)
    j_a = ctx["j_a_prof"][step_idx].reshape(-1, 1)
    j_c = ctx["j_c_prof"][step_idx].reshape(-1, 1)

    rho_a = (ctx["r_a"] / torch.clamp(ctx["params"]["Rs_a"], min=1e-30)).to(dtype=dtype)
    rho_c = (ctx["r_c"] / torch.clamp(ctx["params"]["Rs_c"], min=1e-30)).to(dtype=dtype)

    cs_a_prev = model.predict_profiles(step_idx, rho_a, "a", float(ctx["x_a0"]))
    cs_a_next = model.predict_profiles(next_idx, rho_a, "a", float(ctx["x_a0"]))
    cs_c_prev = model.predict_profiles(step_idx, rho_c, "c", float(ctx["x_c0"]))
    cs_c_next = model.predict_profiles(next_idx, rho_c, "c", float(ctx["x_c0"]))

    r_a = ctx["r_a"]
    r_c = ctx["r_c"]
    dR_a = ctx["dR_a"]
    dR_c = ctx["dR_c"]
    params = ctx["params"]

    grad_cs_a_prev = gradient_batch_1d_ASSBfinal2(cs_a_prev, r_a)
    ds_a_prev = params["D_s_a"](params["T"], params["R"]) * torch.ones_like(cs_a_prev)
    rhs_a = dt * (2.0 / torch.clamp(r_a.reshape(1, -1), min=1e-12)) * grad_cs_a_prev * ds_a_prev + cs_a_prev
    coeff_a = ds_a_prev * dt / (dR_a**2)
    res_a_int = (
        (1.0 + 2.0 * coeff_a[:, 1:-1]) * cs_a_next[:, 1:-1]
        - coeff_a[:, 1:-1] * cs_a_next[:, :-2]
        - coeff_a[:, 1:-1] * cs_a_next[:, 2:]
        - rhs_a[:, 1:-1]
    )
    bound_a_center = (cs_a_next[:, 1] - cs_a_next[:, 0]) / dR_a
    bound_a_surface = (cs_a_next[:, -1] - cs_a_next[:, -2]) / dR_a + j_a.reshape(-1) / torch.clamp(ds_a_prev[:, -1], min=1e-30)

    grad_cs_c_prev = gradient_batch_1d_ASSBfinal2(cs_c_prev, r_c)
    ds_c_prev = params["D_s_c"](
        cs_c_prev,
        params["T"],
        params["R"],
        params["cscamax"],
        ctx["deg_ds_c"],
    )
    ddDs_c_prev = ctx["grad_ds_c_cs_c_experimental_1"](
        cs_c_prev,
        params["T"],
        params["R"],
        params["cscamax"],
        ctx["deg_ds_c"],
    )
    rhs_c = (
        dt * (2.0 / torch.clamp(r_c.reshape(1, -1), min=1e-12)) * grad_cs_c_prev * ds_c_prev
        + dt * (grad_cs_c_prev**2) * ddDs_c_prev
        + cs_c_prev
    )
    coeff_c = ds_c_prev * dt / (dR_c**2)
    res_c_int = (
        (1.0 + 2.0 * coeff_c[:, 1:-1]) * cs_c_next[:, 1:-1]
        - coeff_c[:, 1:-1] * cs_c_next[:, :-2]
        - coeff_c[:, 1:-1] * cs_c_next[:, 2:]
        - rhs_c[:, 1:-1]
    )
    bound_c_center = (cs_c_next[:, 1] - cs_c_next[:, 0]) / dR_c
    bound_c_surface = (cs_c_next[:, -1] - cs_c_next[:, -2]) / dR_c + j_c.reshape(-1) / torch.clamp(ds_c_prev[:, -1], min=1e-30)

    cs_a_scale = torch.maximum(torch.abs(ctx["cs_a0"]), torch.as_tensor(1e-3, dtype=dtype, device=device))
    cs_c_scale = torch.maximum(torch.abs(ctx["cs_c0"]), torch.as_tensor(1e-3, dtype=dtype, device=device))
    grad_a_scale = cs_a_scale / torch.clamp(params["Rs_a"], min=1e-30)
    grad_c_scale = cs_c_scale / torch.clamp(params["Rs_c"], min=1e-30)

    step_a = torch.mean(_normalize_residual_ASSBfinal2(res_a_int, cs_a_scale) ** 2)
    step_c = torch.mean(_normalize_residual_ASSBfinal2(res_c_int, cs_c_scale) ** 2)
    bound_a = torch.mean(_normalize_residual_ASSBfinal2(torch.cat([bound_a_center, bound_a_surface], dim=0), grad_a_scale) ** 2)
    bound_c = torch.mean(_normalize_residual_ASSBfinal2(torch.cat([bound_c_center, bound_c_surface], dim=0), grad_c_scale) ** 2)

    total = (
        float(cfg["STEP_WEIGHT_A"]) * step_a
        + float(cfg["STEP_WEIGHT_C"]) * step_c
        + float(cfg["BOUND_WEIGHT_A"]) * bound_a
        + float(cfg["BOUND_WEIGHT_C"]) * bound_c
    )
    return LossBreakdown_ASSBfinal2(total, step_a, step_c, bound_a, bound_c)


@torch.no_grad()
def predict_all_concentrations_ASSBfinal2(model, ctx: dict[str, Any], time_chunk: int = 2048) -> tuple[np.ndarray, np.ndarray]:
    device = ctx["device"]
    n_t = ctx["n_t"]
    idx_all = torch.arange(n_t, device=device, dtype=torch.long)
    rho_a = (ctx["r_a"] / torch.clamp(ctx["params"]["Rs_a"], min=1e-30)).to(dtype=ctx["dtype"])
    rho_c = (ctx["r_c"] / torch.clamp(ctx["params"]["Rs_c"], min=1e-30)).to(dtype=ctx["dtype"])
    out_a = []
    out_c = []
    for start in range(0, n_t, int(time_chunk)):
        end = min(start + int(time_chunk), n_t)
        idx = idx_all[start:end]
        out_a.append(model.predict_profiles(idx, rho_a, "a", float(ctx["x_a0"])) .detach().cpu())
        out_c.append(model.predict_profiles(idx, rho_c, "c", float(ctx["x_c0"])) .detach().cpu())
    return torch.cat(out_a, dim=0).numpy(), torch.cat(out_c, dim=0).numpy()


@torch.no_grad()
def derive_potentials_from_concentrations_ASSBfinal2(cs_a: np.ndarray, cs_c: np.ndarray, ctx: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    device = ctx["device"]
    dtype = ctx["dtype"]
    params = ctx["params"]
    ce0 = params["ce0"]
    deg_i0_a = ctx["deg_i0_a"]
    time_n = cs_a.shape[0]
    cs_a_t = torch.as_tensor(cs_a, dtype=dtype, device=device)
    cs_c_t = torch.as_tensor(cs_c, dtype=dtype, device=device)
    current = ctx["current_a"]
    j_a_prof = ctx["j_a_prof"]
    j_c_prof = ctx["j_c_prof"]
    phie = torch.zeros((time_n,), dtype=dtype, device=device)
    phis = torch.zeros((time_n,), dtype=dtype, device=device)

    cse_a0 = cs_a_t[0, -1]
    cse_c0 = cs_c_t[0, -1]
    i0_a0 = params["i0_a"](cse_a0, ce0, params["T"], params["alpha_a"], params["csanmax"], params["R"], deg_i0_a)
    i0_c0 = params["i0_c"](cse_c0, ce0, params["T"], params["alpha_c"], params["cscamax"], params["R"])
    Uocp_a0 = params["Uocp_a"](cse_a0, params["csanmax"])
    Uocp_c0 = params["Uocp_c"](cse_c0, params["cscamax"])
    phie[0] = ctx["phie_linearized_experimental_1"](i0_a0, j_a_prof[0], params["R"], params["T"], Uocp_a0)
    phis[0] = ctx["phis_c_linearized_experimental_1"](i0_c0, j_c_prof[0], params["R"], params["T"], Uocp_c0, phie[0])

    if time_n > 1:
        cse_a_prev = cs_a_t[:-1, -1]
        cse_c_prev = cs_c_t[:-1, -1]
        i0_a_prev = params["i0_a"](cse_a_prev, ce0, params["T"], params["alpha_a"], params["csanmax"], params["R"], deg_i0_a)
        i0_c_prev = params["i0_c"](cse_c_prev, ce0, params["T"], params["alpha_c"], params["cscamax"], params["R"])
        Uocp_a_prev = params["Uocp_a"](cse_a_prev, params["csanmax"])
        Uocp_c_prev = params["Uocp_c"](cse_c_prev, params["cscamax"])
        phie[1:] = ctx["phie_linearized_experimental_1"](i0_a_prev, j_a_prof[:-1], params["R"], params["T"], Uocp_a_prev)
        phis[1:] = ctx["phis_c_linearized_experimental_1"](i0_c_prev, j_c_prof[:-1], params["R"], params["T"], Uocp_c_prev, phie[1:])

    return phie.detach().cpu().numpy(), phis.detach().cpu().numpy()
