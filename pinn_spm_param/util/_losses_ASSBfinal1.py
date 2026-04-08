from __future__ import annotations

import numpy as np
import torch

from current_profile_ASSBfinal1 import select_step_indices_ASSBfinal1
from thermo_experimental_1 import (
    grad_ds_c_cs_c_experimental_1,
    phie_linearized_experimental_1,
    phis_c_linearized_experimental_1,
)
from torch_utils import ensure_2d, safe_mean_square, to_tensor


torch.set_default_dtype(torch.float64)


def _stack_term_list(term_groups):
    return [ensure_2d(term[0]) for term in term_groups]


def _sum_sq(term_groups):
    if len(term_groups) == 0:
        return torch.zeros((), dtype=torch.float64)
    total = torch.zeros((), dtype=torch.float64, device=ensure_2d(term_groups[0]).device)
    for term in term_groups:
        total = total + safe_mean_square(ensure_2d(term))
    return total


def _zero(self):
    return torch.zeros((1, 1), dtype=torch.float64, device=self.device)


def _gradient_batch_ASSBfinal1(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    grad = torch.zeros_like(y)
    x1 = torch.as_tensor(x, dtype=torch.float64, device=y.device).reshape(-1)
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


def _tridiag_batched_ASSBfinal1(ds: torch.Tensor, dt: torch.Tensor, dr: torch.Tensor) -> torch.Tensor:
    ds_t = torch.as_tensor(ds, dtype=torch.float64)
    dt_t = torch.as_tensor(dt, dtype=torch.float64, device=ds_t.device).reshape(-1, 1)
    dr_t = torch.as_tensor(dr, dtype=torch.float64, device=ds_t.device)
    batch, n_r = ds_t.shape
    a = torch.as_tensor(1.0, dtype=torch.float64, device=ds_t.device) + torch.as_tensor(2.0, dtype=torch.float64, device=ds_t.device) * ds_t * dt_t / (dr_t**2)
    b = -ds_t * dt_t / (dr_t**2)
    mat = torch.zeros((batch, n_r, n_r), dtype=torch.float64, device=ds_t.device)
    idx = torch.arange(n_r, device=ds_t.device)
    mat[:, idx, idx] = a
    if n_r > 1:
        mat[:, idx[1:], idx[:-1]] = b[:, 1:]
        mat[:, idx[:-1], idx[1:]] = b[:, :-1]
    mat[:, 0, :] = torch.as_tensor(0.0, dtype=torch.float64, device=ds_t.device)
    mat[:, -1, :] = torch.as_tensor(0.0, dtype=torch.float64, device=ds_t.device)
    mat[:, 0, 0] = -torch.as_tensor(1.0, dtype=torch.float64, device=ds_t.device) / dr_t
    mat[:, 0, 1] = torch.as_tensor(1.0, dtype=torch.float64, device=ds_t.device) / dr_t
    mat[:, -1, -2] = -torch.as_tensor(1.0, dtype=torch.float64, device=ds_t.device) / dr_t
    mat[:, -1, -1] = torch.as_tensor(1.0, dtype=torch.float64, device=ds_t.device) / dr_t
    return mat


def _rhs_batched_ASSBfinal1(
    dt: torch.Tensor,
    r: torch.Tensor,
    ddr_cs: torch.Tensor,
    ds: torch.Tensor,
    ddDs_cs: torch.Tensor,
    cs: torch.Tensor,
    bound_grad: torch.Tensor,
) -> torch.Tensor:
    dt_t = torch.as_tensor(dt, dtype=torch.float64, device=cs.device).reshape(-1, 1)
    r_t = torch.as_tensor(r, dtype=torch.float64, device=cs.device).reshape(1, -1)
    safe_r = torch.clamp(r_t, min=1e-12)
    rhs = dt_t * (torch.as_tensor(2.0, dtype=torch.float64, device=cs.device) / safe_r) * ddr_cs * ds
    rhs = rhs + dt_t * (ddr_cs**2) * ddDs_cs
    rhs = rhs + cs
    rhs[:, 0] = torch.as_tensor(0.0, dtype=torch.float64, device=cs.device)
    rhs[:, -1] = torch.as_tensor(bound_grad, dtype=torch.float64, device=cs.device).reshape(-1)
    return rhs


def _select_step_batch_ASSBfinal1(self, t_query: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    t_nodes = torch.as_tensor(self.params["profile_time_s"], dtype=torch.float64, device=self.device).reshape(-1)
    k = select_step_indices_ASSBfinal1(ensure_2d(t_query, device=self.device), t_nodes)
    k = torch.unique(k, sorted=False)
    t_prev = t_nodes[k].reshape(-1, 1)
    t_next = t_nodes[k + 1].reshape(-1, 1)
    dt = (t_next - t_prev).reshape(-1, 1)
    j_a = torch.as_tensor(self.params["profile_j_a"], dtype=torch.float64, device=self.device).reshape(-1)[k].reshape(-1, 1)
    j_c = torch.as_tensor(self.params["profile_j_c"], dtype=torch.float64, device=self.device).reshape(-1)[k].reshape(-1, 1)
    positive = (dt.reshape(-1) > 0.0)
    if bool(torch.any(positive).item()):
        t_prev = t_prev[positive]
        t_next = t_next[positive]
        dt = dt[positive]
        j_a = j_a[positive]
        j_c = j_c[positive]
    return t_prev, t_next, dt, j_a, j_c


def _eval_branch_on_grid_ASSBfinal1(self, t_nodes, r_grid, deg_i0_a, deg_ds_c, out_index: int, rescale_name: str):
    t_nodes = ensure_2d(t_nodes, device=self.device)
    deg_i0_a = ensure_2d(deg_i0_a, device=self.device)
    deg_ds_c = ensure_2d(deg_ds_c, device=self.device)
    r_grid = torch.as_tensor(r_grid, dtype=torch.float64, device=self.device).reshape(-1)
    batch = t_nodes.shape[0]
    n_r = r_grid.numel()

    t_flat = t_nodes.repeat_interleave(n_r, dim=0)
    r_flat = r_grid.reshape(1, -1).repeat(batch, 1).reshape(-1, 1)
    deg_i0_flat = deg_i0_a.repeat_interleave(n_r, dim=0)
    deg_ds_flat = deg_ds_c.repeat_interleave(n_r, dim=0)

    outputs = self.model(
        [
            t_flat / self.params["rescale_T"],
            r_flat / self.params["rescale_R"],
            self.rescale_param(deg_i0_flat, self.ind_deg_i0_a),
            self.rescale_param(deg_ds_flat, self.ind_deg_ds_c),
        ],
        training=True,
    )
    raw = outputs[out_index]
    if rescale_name == "cs_a":
        val = self.rescaleCs_a(raw, t_flat, r_flat, deg_i0_flat, deg_ds_flat, clip=False)
    elif rescale_name == "cs_c":
        val = self.rescaleCs_c(raw, t_flat, r_flat, deg_i0_flat, deg_ds_flat, clip=False)
    else:
        raise ValueError(rescale_name)
    return val.reshape(batch, n_r)


def _eval_potential_ASSBfinal1(self, t_nodes, deg_i0_a, deg_ds_c, out_index: int, which: str):
    t_nodes = ensure_2d(t_nodes, device=self.device)
    deg_i0_a = ensure_2d(deg_i0_a, device=self.device)
    deg_ds_c = ensure_2d(deg_ds_c, device=self.device)
    r_surf = self.params["Rs_a"] * torch.ones_like(t_nodes, dtype=torch.float64, device=self.device)
    outputs = self.model(
        [
            t_nodes / self.params["rescale_T"],
            r_surf / self.params["rescale_R"],
            self.rescale_param(deg_i0_a, self.ind_deg_i0_a),
            self.rescale_param(deg_ds_c, self.ind_deg_ds_c),
        ],
        training=True,
    )
    raw = outputs[out_index]
    if which == "phie":
        return self.rescalePhie(raw, t_nodes, deg_i0_a, deg_ds_c)
    if which == "phis_c":
        return self.rescalePhis_c(raw, t_nodes, deg_i0_a, deg_ds_c)
    raise ValueError(which)


def loss_fn(interiorTerms, boundaryTerms, dataTerms, regularizationTerms, alpha):
    int_loss = _sum_sq(interiorTerms)
    bound_loss = _sum_sq(boundaryTerms)
    data_loss = _sum_sq(dataTerms)
    reg_loss = _sum_sq(regularizationTerms)
    global_loss = (
        float(alpha[0]) * int_loss
        + float(alpha[1]) * bound_loss
        + float(alpha[2]) * data_loss
        + float(alpha[3]) * reg_loss
    )
    return (
        global_loss,
        float(alpha[0]) * int_loss,
        float(alpha[1]) * bound_loss,
        float(alpha[2]) * data_loss,
        float(alpha[3]) * reg_loss,
    )


def loss_fn_lbfgs(interiorTerms, boundaryTerms, dataTerms, regularizationTerms, alpha):
    return loss_fn(interiorTerms, boundaryTerms, dataTerms, regularizationTerms, alpha)


def loss_fn_lbfgs_SA(
    interiorTerms,
    boundaryTerms,
    dataTerms,
    regularizationTerms,
    int_col_weights,
    bound_col_weights,
    data_col_weights,
    reg_col_weights,
    alpha,
):
    return loss_fn(interiorTerms, boundaryTerms, dataTerms, regularizationTerms, alpha)


def loss_fn_lbfgs_annealing(
    interiorTerms,
    boundaryTerms,
    dataTerms,
    regularizationTerms,
    int_loss_weights,
    bound_loss_weights,
    data_loss_weights,
    reg_loss_weights,
    alpha,
):
    return loss_fn(interiorTerms, boundaryTerms, dataTerms, regularizationTerms, alpha)


def loss_fn_dynamicAttention_tensor(
    interiorTerms,
    boundaryTerms,
    dataTerms,
    regularizationTerms,
    int_col_weights,
    bound_col_weights,
    data_col_weights,
    reg_col_weights,
    alpha,
):
    weighted = loss_fn(interiorTerms, boundaryTerms, dataTerms, regularizationTerms, alpha)
    unweighted = loss_fn(interiorTerms, boundaryTerms, dataTerms, regularizationTerms, [1.0, 1.0, 1.0, 1.0])
    return (
        weighted[0],
        unweighted[0],
        weighted[1],
        weighted[2],
        weighted[3],
        weighted[4],
    )


def loss_fn_annealing(
    interiorTerms,
    boundaryTerms,
    dataTerms,
    regularizationTerms,
    int_loss_terms,
    bound_loss_terms,
    data_loss_terms,
    reg_loss_terms,
    int_loss_weights,
    bound_loss_weights,
    data_loss_weights,
    reg_loss_weights,
    alpha,
):
    return loss_fn(interiorTerms, boundaryTerms, dataTerms, regularizationTerms, alpha)


def setResidualRescaling(self, weights):
    cs_a_exc = max(float(to_tensor(self.params["cs_a_excursion_est"]).detach().cpu().reshape(-1)[0]), 1e-8)
    cs_c_exc = max(float(to_tensor(self.params["cs_c_excursion_est"]).detach().cpu().reshape(-1)[0]), 1e-8)
    phie_exc = max(float(to_tensor(self.params["phie_excursion_est"]).detach().cpu().reshape(-1)[0]), 1e-8)
    phis_exc = max(float(to_tensor(self.params["phis_c_excursion_est"]).detach().cpu().reshape(-1)[0]), 1e-8)
    Rs_a = max(float(to_tensor(self.params["Rs_a"]).detach().cpu().reshape(-1)[0]), 1e-12)
    Rs_c = max(float(to_tensor(self.params["Rs_c"]).detach().cpu().reshape(-1)[0]), 1e-12)

    if self.activeInt:
        if weights is None:
            w_phie_int = np.float64(1.0)
            w_phis_c_int = np.float64(1.0)
            w_cs_a_int = np.float64(20.0)
            w_cs_c_int = np.float64(20.0)
        else:
            w_phie_int = weights["phie_int"]
            w_phis_c_int = weights["phis_c_int"]
            w_cs_a_int = weights["cs_a_int"]
            w_cs_c_int = weights["cs_c_int"]
        self.interiorTerms_rescale_unweighted = [
            np.float64(1.0 / phie_exc),
            np.float64(1.0 / phis_exc),
            np.float64(1.0 / cs_a_exc),
            np.float64(1.0 / cs_c_exc),
        ]
        self.interiorTerms_rescale = [
            w_phie_int * self.interiorTerms_rescale_unweighted[0],
            w_phis_c_int * self.interiorTerms_rescale_unweighted[1],
            w_cs_a_int * self.interiorTerms_rescale_unweighted[2],
            w_cs_c_int * self.interiorTerms_rescale_unweighted[3],
        ]
    else:
        self.interiorTerms_rescale_unweighted = [np.float64(0.0)]
        self.interiorTerms_rescale = [np.float64(0.0)]

    if self.activeBound:
        if weights is None:
            w_cs_a_rmin_bound = np.float64(1.0)
            w_cs_c_rmin_bound = np.float64(1.0)
            w_cs_a_rmax_bound = np.float64(5.0)
            w_cs_c_rmax_bound = np.float64(5.0)
        else:
            w_cs_a_rmin_bound = weights["cs_a_rmin_bound"]
            w_cs_c_rmin_bound = weights["cs_c_rmin_bound"]
            w_cs_a_rmax_bound = weights["cs_a_rmax_bound"]
            w_cs_c_rmax_bound = weights["cs_c_rmax_bound"]
        self.boundaryTerms_rescale_unweighted = [
            np.float64(Rs_a / cs_a_exc),
            np.float64(Rs_c / cs_c_exc),
            np.float64(Rs_a / cs_a_exc),
            np.float64(Rs_c / cs_c_exc),
        ]
        self.boundaryTerms_rescale = [
            w_cs_a_rmin_bound * self.boundaryTerms_rescale_unweighted[0],
            w_cs_c_rmin_bound * self.boundaryTerms_rescale_unweighted[1],
            w_cs_a_rmax_bound * self.boundaryTerms_rescale_unweighted[2],
            w_cs_c_rmax_bound * self.boundaryTerms_rescale_unweighted[3],
        ]
    else:
        self.boundaryTerms_rescale_unweighted = [np.float64(0.0)]
        self.boundaryTerms_rescale = [np.float64(0.0)]

    self.n_data_terms = 4
    if self.activeData:
        if weights is None:
            w_phie_dat = np.float64(1.0)
            w_phis_c_dat = np.float64(1.0)
            w_cs_a_dat = np.float64(1.0)
            w_cs_c_dat = np.float64(1.0)
        else:
            w_phie_dat = weights["phie_dat"]
            w_phis_c_dat = weights["phis_c_dat"]
            w_cs_a_dat = weights["cs_a_dat"]
            w_cs_c_dat = weights["cs_c_dat"]
        self.dataTerms_rescale_unweighted = [
            np.float64(1.0 / max(float(to_tensor(self.params["mag_phie"]).cpu().reshape(-1)[0]), 1e-8)),
            np.float64(1.0 / max(float(to_tensor(self.params["mag_phis_c"]).cpu().reshape(-1)[0]), 1e-8)),
            np.float64(1.0 / max(float(to_tensor(self.params["mag_cs_a"]).cpu().reshape(-1)[0]), 1e-8)),
            np.float64(1.0 / max(float(to_tensor(self.params["mag_cs_c"]).cpu().reshape(-1)[0]), 1e-8)),
        ]
        self.dataTerms_rescale = [
            w_phie_dat * self.dataTerms_rescale_unweighted[0],
            w_phis_c_dat * self.dataTerms_rescale_unweighted[1],
            w_cs_a_dat * self.dataTerms_rescale_unweighted[2],
            w_cs_c_dat * self.dataTerms_rescale_unweighted[3],
        ]
    else:
        self.dataTerms_rescale_unweighted = [np.float64(0.0)]
        self.dataTerms_rescale = [np.float64(0.0)]

    if self.activeReg:
        self.regTerms_rescale_unweighted = [np.float64(1.0)]
        self.regTerms_rescale = [np.float64(1.0)]
    else:
        self.regTerms_rescale_unweighted = [np.float64(0.0)]
        self.regTerms_rescale = [np.float64(0.0)]


def data_loss(self, x_batch_trainList, x_cs_batch_trainList, x_params_batch_trainList, y_batch_trainList):
    if not self.activeData:
        return [[_zero(self)]]

    resc_t = self.params["rescale_T"]
    resc_r = self.params["rescale_R"]

    surfR_a = self.params["Rs_a"] * torch.ones_like(ensure_2d(x_batch_trainList[self.ind_phie_data][:, self.ind_t], device=self.device))
    out_phie = self.model(
        [
            ensure_2d(x_batch_trainList[self.ind_phie_data][:, self.ind_t], device=self.device) / resc_t,
            surfR_a / resc_r,
            self.rescale_param(x_params_batch_trainList[self.ind_phie_data][:, self.ind_deg_i0_a], self.ind_deg_i0_a),
            self.rescale_param(x_params_batch_trainList[self.ind_phie_data][:, self.ind_deg_ds_c], self.ind_deg_ds_c),
        ],
        training=True,
    )
    phie_pred_rescaled = self.rescalePhie(
        out_phie[self.ind_phie],
        x_batch_trainList[self.ind_phie_data][:, self.ind_t],
        x_params_batch_trainList[self.ind_phie_data][:, self.ind_deg_i0_a],
        x_params_batch_trainList[self.ind_phie_data][:, self.ind_deg_ds_c],
    )

    out_phis_c = self.model(
        [
            ensure_2d(x_batch_trainList[self.ind_phis_c_data][:, self.ind_t], device=self.device) / resc_t,
            surfR_a[: ensure_2d(x_batch_trainList[self.ind_phis_c_data][:, self.ind_t], device=self.device).shape[0]] / resc_r,
            self.rescale_param(x_params_batch_trainList[self.ind_phis_c_data][:, self.ind_deg_i0_a], self.ind_deg_i0_a),
            self.rescale_param(x_params_batch_trainList[self.ind_phis_c_data][:, self.ind_deg_ds_c], self.ind_deg_ds_c),
        ],
        training=True,
    )
    phis_c_pred_rescaled = self.rescalePhis_c(
        out_phis_c[self.ind_phis_c],
        x_batch_trainList[self.ind_phis_c_data][:, self.ind_t],
        x_params_batch_trainList[self.ind_phis_c_data][:, self.ind_deg_i0_a],
        x_params_batch_trainList[self.ind_phis_c_data][:, self.ind_deg_ds_c],
    )

    cs_a_pred_non_rescaled = self.model(
        [
            ensure_2d(x_cs_batch_trainList[self.ind_cs_a_data - self.ind_cs_offset_data][:, self.ind_t], device=self.device) / resc_t,
            ensure_2d(x_cs_batch_trainList[self.ind_cs_a_data - self.ind_cs_offset_data][:, self.ind_r], device=self.device) / resc_r,
            self.rescale_param(x_params_batch_trainList[self.ind_cs_a_data][:, self.ind_deg_i0_a], self.ind_deg_i0_a),
            self.rescale_param(x_params_batch_trainList[self.ind_cs_a_data][:, self.ind_deg_ds_c], self.ind_deg_ds_c),
        ],
        training=True,
    )[self.ind_cs_a]
    cs_a_pred_rescaled = self.rescaleCs_a(
        cs_a_pred_non_rescaled,
        x_cs_batch_trainList[self.ind_cs_a_data - self.ind_cs_offset_data][:, self.ind_t],
        x_cs_batch_trainList[self.ind_cs_a_data - self.ind_cs_offset_data][:, self.ind_r],
        x_params_batch_trainList[self.ind_cs_a_data][:, self.ind_deg_i0_a],
        x_params_batch_trainList[self.ind_cs_a_data][:, self.ind_deg_ds_c],
        clip=False,
    )

    cs_c_pred_non_rescaled = self.model(
        [
            ensure_2d(x_cs_batch_trainList[self.ind_cs_c_data - self.ind_cs_offset_data][:, self.ind_t], device=self.device) / resc_t,
            ensure_2d(x_cs_batch_trainList[self.ind_cs_c_data - self.ind_cs_offset_data][:, self.ind_r], device=self.device) / resc_r,
            self.rescale_param(x_params_batch_trainList[self.ind_cs_c_data][:, self.ind_deg_i0_a], self.ind_deg_i0_a),
            self.rescale_param(x_params_batch_trainList[self.ind_cs_c_data][:, self.ind_deg_ds_c], self.ind_deg_ds_c),
        ],
        training=True,
    )[self.ind_cs_c]
    cs_c_pred_rescaled = self.rescaleCs_c(
        cs_c_pred_non_rescaled,
        x_cs_batch_trainList[self.ind_cs_c_data - self.ind_cs_offset_data][:, self.ind_t],
        x_cs_batch_trainList[self.ind_cs_c_data - self.ind_cs_offset_data][:, self.ind_r],
        x_params_batch_trainList[self.ind_cs_c_data][:, self.ind_deg_i0_a],
        x_params_batch_trainList[self.ind_cs_c_data][:, self.ind_deg_ds_c],
        clip=False,
    )

    return [
        [phie_pred_rescaled - ensure_2d(y_batch_trainList[self.ind_phie_data], device=self.device)],
        [phis_c_pred_rescaled - ensure_2d(y_batch_trainList[self.ind_phis_c_data], device=self.device)],
        [cs_a_pred_rescaled - ensure_2d(y_batch_trainList[self.ind_cs_a_data], device=self.device)],
        [cs_c_pred_rescaled - ensure_2d(y_batch_trainList[self.ind_cs_c_data], device=self.device)],
    ]


def interior_loss(self, int_col_pts=None, int_col_params=None, tmax=None):
    if not self.activeInt:
        return [[_zero(self)]]

    t, _, _, _, _, deg_i0_a, deg_ds_c = self._prepare_interior_batch(int_col_pts, int_col_params, tmax)
    t_prev, t_next, dt, j_a, j_c = _select_step_batch_ASSBfinal1(self, t)
    if t_prev.numel() == 0:
        return [[_zero(self)], [_zero(self)], [_zero(self)], [_zero(self)]]

    deg_i0_a = ensure_2d(deg_i0_a, device=self.device)[: t_prev.shape[0]]
    deg_ds_c = ensure_2d(deg_ds_c, device=self.device)[: t_prev.shape[0]]

    r_a_grid = self.params["profile_r_a_grid"]
    r_c_grid = self.params["profile_r_c_grid"]
    dR_a = self.params["profile_dR_a"]
    dR_c = self.params["profile_dR_c"]

    cs_a_prev = _eval_branch_on_grid_ASSBfinal1(self, t_prev, r_a_grid, deg_i0_a, deg_ds_c, self.ind_cs_a, "cs_a")
    cs_a_next = _eval_branch_on_grid_ASSBfinal1(self, t_next, r_a_grid, deg_i0_a, deg_ds_c, self.ind_cs_a, "cs_a")
    cs_c_prev = _eval_branch_on_grid_ASSBfinal1(self, t_prev, r_c_grid, deg_i0_a, deg_ds_c, self.ind_cs_c, "cs_c")
    cs_c_next = _eval_branch_on_grid_ASSBfinal1(self, t_next, r_c_grid, deg_i0_a, deg_ds_c, self.ind_cs_c, "cs_c")

    ce = self.params["ce0"] * torch.ones_like(t_prev, dtype=torch.float64)
    cse_a_prev = cs_a_prev[:, -1].reshape(-1, 1)
    cse_c_prev = cs_c_prev[:, -1].reshape(-1, 1)

    i0_a_prev = self.params["i0_a"](
        cse_a_prev,
        ce,
        self.params["T"],
        self.params["alpha_a"],
        self.params["csanmax"],
        self.params["R"],
        deg_i0_a,
    )
    i0_c_prev = self.params["i0_c"](
        cse_c_prev,
        ce,
        self.params["T"],
        self.params["alpha_c"],
        self.params["cscamax"],
        self.params["R"],
    )
    Uocp_a_prev = self.params["Uocp_a"](cse_a_prev, self.params["csanmax"])
    Uocp_c_prev = self.params["Uocp_c"](cse_c_prev, self.params["cscamax"])

    phie_pred_next = _eval_potential_ASSBfinal1(self, t_next, deg_i0_a, deg_ds_c, self.ind_phie, "phie")
    phie_rhs_next = phie_linearized_experimental_1(i0_a_prev, j_a, self.params["R"], self.params["T"], Uocp_a_prev)

    phis_c_pred_next = _eval_potential_ASSBfinal1(self, t_next, deg_i0_a, deg_ds_c, self.ind_phis_c, "phis_c")
    phis_c_rhs_next = phis_c_linearized_experimental_1(i0_c_prev, j_c, self.params["R"], self.params["T"], Uocp_c_prev, phie_rhs_next)

    ds_a_prev = self.params["D_s_a"](self.params["T"], self.params["R"]) * torch.ones_like(cs_a_prev)
    grad_cs_a_prev = _gradient_batch_ASSBfinal1(cs_a_prev, r_a_grid)
    A_a = _tridiag_batched_ASSBfinal1(ds_a_prev, dt, dR_a)
    B_a = _rhs_batched_ASSBfinal1(
        dt,
        r_a_grid,
        grad_cs_a_prev,
        ds_a_prev,
        torch.zeros_like(cs_a_prev),
        cs_a_prev,
        -j_a / torch.clamp(ds_a_prev[:, -1].reshape(-1, 1), min=1e-30),
    )
    res_cs_a = torch.bmm(A_a, cs_a_next.unsqueeze(-1)).squeeze(-1) - B_a

    ds_c_prev = self.params["D_s_c"](
        cs_c_prev,
        self.params["T"],
        self.params["R"],
        self.params["cscamax"],
        deg_ds_c,
    )
    grad_Ds_c_prev = grad_ds_c_cs_c_experimental_1(
        cs_c_prev,
        self.params["T"],
        self.params["R"],
        self.params["cscamax"],
        deg_ds_c,
    )
    grad_cs_c_prev = _gradient_batch_ASSBfinal1(cs_c_prev, r_c_grid)
    A_c = _tridiag_batched_ASSBfinal1(ds_c_prev, dt, dR_c)
    B_c = _rhs_batched_ASSBfinal1(
        dt,
        r_c_grid,
        grad_cs_c_prev,
        ds_c_prev,
        grad_Ds_c_prev,
        cs_c_prev,
        -j_c / torch.clamp(ds_c_prev[:, -1].reshape(-1, 1), min=1e-30),
    )
    res_cs_c = torch.bmm(A_c, cs_c_next.unsqueeze(-1)).squeeze(-1) - B_c

    return [
        [phie_pred_next - phie_rhs_next],
        [phis_c_pred_next - phis_c_rhs_next],
        [res_cs_a],
        [res_cs_c],
    ]


def boundary_loss(self, bound_col_pts=None, bound_col_params=None, tmax=None):
    if not self.activeBound:
        return [[_zero(self)]]

    t_bound, _, _, _, deg_i0_a_bound, deg_ds_c_bound = self._prepare_boundary_batch(bound_col_pts, bound_col_params, tmax)
    t_prev, t_next, dt, j_a, j_c = _select_step_batch_ASSBfinal1(self, t_bound)
    if t_prev.numel() == 0:
        return [[_zero(self)], [_zero(self)], [_zero(self)], [_zero(self)]]

    deg_i0_a_bound = ensure_2d(deg_i0_a_bound, device=self.device)[: t_prev.shape[0]]
    deg_ds_c_bound = ensure_2d(deg_ds_c_bound, device=self.device)[: t_prev.shape[0]]

    r_a_grid = self.params["profile_r_a_grid"]
    r_c_grid = self.params["profile_r_c_grid"]

    cs_a_prev = _eval_branch_on_grid_ASSBfinal1(self, t_prev, r_a_grid, deg_i0_a_bound, deg_ds_c_bound, self.ind_cs_a, "cs_a")
    cs_a_next = _eval_branch_on_grid_ASSBfinal1(self, t_next, r_a_grid, deg_i0_a_bound, deg_ds_c_bound, self.ind_cs_a, "cs_a")
    cs_c_prev = _eval_branch_on_grid_ASSBfinal1(self, t_prev, r_c_grid, deg_i0_a_bound, deg_ds_c_bound, self.ind_cs_c, "cs_c")
    cs_c_next = _eval_branch_on_grid_ASSBfinal1(self, t_next, r_c_grid, deg_i0_a_bound, deg_ds_c_bound, self.ind_cs_c, "cs_c")

    grad_cs_a_next = _gradient_batch_ASSBfinal1(cs_a_next, r_a_grid)
    grad_cs_c_next = _gradient_batch_ASSBfinal1(cs_c_next, r_c_grid)
    ds_a_prev = self.params["D_s_a"](self.params["T"], self.params["R"]) * torch.ones_like(cs_a_prev[:, -1].reshape(-1, 1))
    ds_c_prev_surf = self.params["D_s_c"](
        cs_c_prev[:, -1].reshape(-1, 1),
        self.params["T"],
        self.params["R"],
        self.params["cscamax"],
        deg_ds_c_bound,
    )

    return [
        [grad_cs_a_next[:, 0].reshape(-1, 1)],
        [grad_cs_c_next[:, 0].reshape(-1, 1)],
        [grad_cs_a_next[:, -1].reshape(-1, 1) + j_a / torch.clamp(ds_a_prev, min=1e-30)],
        [grad_cs_c_next[:, -1].reshape(-1, 1) + j_c / torch.clamp(ds_c_prev_surf, min=1e-30)],
    ]


def regularization_loss(self, reg_col_pts=None, tmax=None):
    if not self.activeReg:
        return [[_zero(self)]]
    return [[_zero(self)]]


def _apply_scales(terms, scales):
    scaled = []
    for term, scale in zip(terms, scales):
        scaled.append(ensure_2d(term[0]) * float(scale))
    return scaled


def get_unweighted_loss(
    self,
    int_col_pts,
    int_col_params,
    bound_col_pts,
    bound_col_params,
    reg_col_pts,
    reg_col_params,
    x_trainList,
    x_params_trainList,
    y_trainList,
    n_batch=1,
    tmax=None,
):
    losses = []
    for i_batch in range(int(n_batch)):
        batch = self._assemble_batch(i_batch, use_lbfgs=True)
        interiorTerms = self.interior_loss(batch["int_col_pts"], batch["int_col_params"], tmax)
        boundaryTerms = self.boundary_loss(batch["bound_col_pts"], batch["bound_col_params"], tmax)
        dataTerms = self.data_loss(batch["x_batch_trainList"], batch["x_cs_batch_trainList"], batch["x_params_batch_trainList"], batch["y_batch_trainList"])
        regTerms = self.regularization_loss(batch["reg_col_pts"], tmax)
        interiorTerms_rescaled = _apply_scales(interiorTerms, self.interiorTerms_rescale_unweighted)
        boundaryTerms_rescaled = _apply_scales(boundaryTerms, self.boundaryTerms_rescale_unweighted)
        dataTerms_rescaled = _apply_scales(dataTerms, self.dataTerms_rescale_unweighted)
        regTerms_rescaled = _apply_scales(regTerms, self.regTerms_rescale_unweighted)
        losses.append(loss_fn(interiorTerms_rescaled, boundaryTerms_rescaled, dataTerms_rescaled, regTerms_rescaled, self.alpha_unweighted)[0])
    return torch.mean(torch.stack(losses))


def get_loss_and_flat_grad(*args, **kwargs):
    raise NotImplementedError("TensorFlow-style flattened LBFGS interface was replaced by torch.optim.LBFGS in the PyTorch patch.")


def get_loss_and_flat_grad_SA(*args, **kwargs):
    raise NotImplementedError("Dynamic attention is not ported in this first PyTorch patch.")


def get_loss_and_flat_grad_annealing(*args, **kwargs):
    raise NotImplementedError("Annealing weights are not ported in this first PyTorch patch.")
