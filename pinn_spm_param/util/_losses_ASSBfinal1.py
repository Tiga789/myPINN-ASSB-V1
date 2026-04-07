from __future__ import annotations

import numpy as np
import torch

from current_profile_ASSBfinal1 import sample_profile_zoh_left_ASSBfinal1
from thermo_experimental_1 import (
    phie_linearized_experimental_1,
    phis_c_linearized_experimental_1,
)
from torch_utils import ensure_2d, grad, safe_mean_square, to_tensor


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



def _profile_flux_of_t_ASSBfinal1(self, t_query, which: str) -> torch.Tensor:
    t_ref = ensure_2d(self.params["profile_time_s"], device=self.device).reshape(-1)
    if which == "a":
        y_ref = ensure_2d(self.params["profile_j_a"], device=self.device).reshape(-1)
    elif which == "c":
        y_ref = ensure_2d(self.params["profile_j_c"], device=self.device).reshape(-1)
    else:
        raise ValueError(which)
    return sample_profile_zoh_left_ASSBfinal1(ensure_2d(t_query, device=self.device), t_ref, y_ref)



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
    cs_a0 = float(to_tensor(self.params["cs_a0"]).detach().cpu().reshape(-1)[0])
    cs_c0 = float(to_tensor(self.params["cs_c0"]).detach().cpu().reshape(-1)[0])
    Ds_a = float(self.params["D_s_a"](self.params["T"], self.params["R"]).detach().cpu().reshape(-1)[0])
    Ds_c = float(
        self.params["D_s_c"](
            self.params["cs_c0"],
            self.params["T"],
            self.params["R"],
            self.params["cscamax"],
            self.params["deg_ds_c_ref"],
        )
        .detach()
        .cpu()
        .reshape(-1)[0]
    )
    j_a_mag = max(float(to_tensor(self.params["j_a_abs_max"]).detach().cpu().reshape(-1)[0]), 1e-30)
    j_c_mag = max(float(to_tensor(self.params["j_c_abs_max"]).detach().cpu().reshape(-1)[0]), 1e-30)
    C = max(abs(float(to_tensor(self.params["C"]).detach().cpu().reshape(-1)[0])), 1e-30)

    self.phie_transp_resc = np.float64(1.0) / j_a_mag
    self.phis_c_transp_resc = np.float64(1.0) / j_c_mag
    self.cs_a_transp_resc = (np.float64(3600.0) / np.float64(C)) / max(float(to_tensor(self.params["csanmax"]).cpu().reshape(-1)[0]), 1e-30)
    self.cs_c_transp_resc = (np.float64(3600.0) / np.float64(C)) / max(float(to_tensor(self.params["cscamax"]).cpu().reshape(-1)[0]), 1e-30)

    if self.activeInt:
        if weights is None:
            w_phie_int = np.float64(1.0)
            w_phis_c_int = np.float64(1.0)
            w_cs_a_int = np.float64(50.0)
            w_cs_c_int = np.float64(50.0)
        else:
            w_phie_int = weights["phie_int"]
            w_phis_c_int = weights["phis_c_int"]
            w_cs_a_int = weights["cs_a_int"]
            w_cs_c_int = weights["cs_c_int"]
        self.interiorTerms_rescale_unweighted = [
            abs(self.phie_transp_resc),
            abs(self.phis_c_transp_resc),
            abs(self.cs_a_transp_resc),
            abs(self.cs_c_transp_resc),
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

    self.cs_a_bound_resc = Ds_a / j_a_mag
    self.cs_c_bound_resc = Ds_c / j_c_mag
    self.cs_a_bound_j_resc = Ds_a / j_a_mag
    self.cs_c_bound_j_resc = Ds_c / j_c_mag
    if self.activeBound:
        if weights is None:
            w_cs_a_rmin_bound = np.float64(1.0)
            w_cs_c_rmin_bound = np.float64(1.0)
            w_cs_a_rmax_bound = np.float64(10.0)
            w_cs_c_rmax_bound = np.float64(10.0)
        else:
            w_cs_a_rmin_bound = weights["cs_a_rmin_bound"]
            w_cs_c_rmin_bound = weights["cs_c_rmin_bound"]
            w_cs_a_rmax_bound = weights["cs_a_rmax_bound"]
            w_cs_c_rmax_bound = weights["cs_c_rmax_bound"]
        self.boundaryTerms_rescale_unweighted = [
            abs(self.cs_a_bound_resc),
            abs(self.cs_c_bound_resc),
            abs(self.cs_a_bound_j_resc),
            abs(self.cs_c_bound_j_resc),
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
            np.float64(1.0) / np.float64(float(to_tensor(self.params["mag_phie"]).cpu().reshape(-1)[0])),
            np.float64(1.0) / np.float64(float(to_tensor(self.params["mag_phis_c"]).cpu().reshape(-1)[0])),
            np.float64(1.0) / np.float64(float(to_tensor(self.params["mag_cs_a"]).cpu().reshape(-1)[0])),
            np.float64(1.0) / np.float64(float(to_tensor(self.params["mag_cs_c"]).cpu().reshape(-1)[0])),
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
    phie_pred_non_rescaled = out_phie[self.ind_phie]
    phie_pred_rescaled = self.rescalePhie(
        phie_pred_non_rescaled,
        x_batch_trainList[self.ind_phie_data][:, self.ind_t],
        x_params_batch_trainList[self.ind_phie_data][:, self.ind_deg_i0_a],
        x_params_batch_trainList[self.ind_phie_data][:, self.ind_deg_ds_c],
    )

    surfR_a = self.params["Rs_a"] * torch.ones_like(ensure_2d(x_batch_trainList[self.ind_phis_c_data][:, self.ind_t], device=self.device))
    out_phis_c = self.model(
        [
            ensure_2d(x_batch_trainList[self.ind_phis_c_data][:, self.ind_t], device=self.device) / resc_t,
            surfR_a / resc_r,
            self.rescale_param(x_params_batch_trainList[self.ind_phis_c_data][:, self.ind_deg_i0_a], self.ind_deg_i0_a),
            self.rescale_param(x_params_batch_trainList[self.ind_phis_c_data][:, self.ind_deg_ds_c], self.ind_deg_ds_c),
        ],
        training=True,
    )
    phis_c_pred_non_rescaled = out_phis_c[self.ind_phis_c]
    phis_c_pred_rescaled = self.rescalePhis_c(
        phis_c_pred_non_rescaled,
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

    t, r_a, r_c, rSurf_a, rSurf_c, deg_i0_a, deg_ds_c = self._prepare_interior_batch(int_col_pts, int_col_params, tmax)
    resc_t = self.params["rescale_T"]
    resc_r = self.params["rescale_R"]
    ce = self.params["ce0"] * torch.ones_like(t, dtype=torch.float64)

    r_a = r_a.clone().detach().requires_grad_(True)
    r_c = r_c.clone().detach().requires_grad_(True)
    t = t.clone().detach().requires_grad_(True)

    output_a = self.model(
        [
            t / resc_t,
            r_a / resc_r,
            self.rescale_param(deg_i0_a, self.ind_deg_i0_a),
            self.rescale_param(deg_ds_c, self.ind_deg_ds_c),
        ],
        training=True,
    )
    output_c = self.model(
        [
            t / resc_t,
            r_c / resc_r,
            self.rescale_param(deg_i0_a, self.ind_deg_i0_a),
            self.rescale_param(deg_ds_c, self.ind_deg_ds_c),
        ],
        training=True,
    )
    output_surf_a = self.model(
        [
            t / resc_t,
            rSurf_a / resc_r,
            self.rescale_param(deg_i0_a, self.ind_deg_i0_a),
            self.rescale_param(deg_ds_c, self.ind_deg_ds_c),
        ],
        training=True,
    )
    output_surf_c = self.model(
        [
            t / resc_t,
            rSurf_c / resc_r,
            self.rescale_param(deg_i0_a, self.ind_deg_i0_a),
            self.rescale_param(deg_ds_c, self.ind_deg_ds_c),
        ],
        training=True,
    )

    phie = self.rescalePhie(output_a[self.ind_phie], t, deg_i0_a, deg_ds_c)
    phis_c = self.rescalePhis_c(output_c[self.ind_phis_c], t, deg_i0_a, deg_ds_c)
    cs_a = self.rescaleCs_a(output_a[self.ind_cs_a], t, r_a, deg_i0_a, deg_ds_c, clip=False)
    cs_c = self.rescaleCs_c(output_c[self.ind_cs_c], t, r_c, deg_i0_a, deg_ds_c, clip=False)
    cse_a = self.rescaleCs_a(output_surf_a[self.ind_cs_a], t, rSurf_a, deg_i0_a, deg_ds_c, clip=False)
    cse_c = self.rescaleCs_c(output_surf_c[self.ind_cs_c], t, rSurf_c, deg_i0_a, deg_ds_c, clip=False)

    i0_a = self.params["i0_a"](
        cse_a,
        ce,
        self.params["T"],
        self.params["alpha_a"],
        self.params["csanmax"],
        self.params["R"],
        deg_i0_a,
    )
    i0_c = self.params["i0_c"](
        cse_c,
        ce,
        self.params["T"],
        self.params["alpha_c"],
        self.params["cscamax"],
        self.params["R"],
    )
    uocp_a = self.params["Uocp_a"](cse_a, self.params["csanmax"])
    uocp_c = self.params["Uocp_c"](cse_c, self.params["cscamax"])
    j_a_rhs = _profile_flux_of_t_ASSBfinal1(self, t, "a")
    j_c_rhs = _profile_flux_of_t_ASSBfinal1(self, t, "c")

    phie_rhs = phie_linearized_experimental_1(i0_a, j_a_rhs, self.params["R"], self.params["T"], uocp_a)
    phis_c_rhs = phis_c_linearized_experimental_1(i0_c, j_c_rhs, self.params["R"], self.params["T"], uocp_c, phie)

    cs_a_r = grad(cs_a, r_a)
    ds_a = self.params["D_s_a"](self.params["T"], self.params["R"]) + np.float64(0.0) * r_a
    cs_c_r = grad(cs_c, r_c)
    ds_c = self.params["D_s_c"](
        cs_c,
        self.params["T"],
        self.params["R"],
        self.params["cscamax"],
        deg_ds_c,
    ) + np.float64(0.0) * r_c

    cs_a_t = grad(cs_a, t)
    cs_a_r_r = grad(cs_a_r, r_a)
    ds_a_r = grad(ds_a, r_a)

    cs_c_t = grad(cs_c, t)
    cs_c_r_r = grad(cs_c_r, r_c)
    ds_c_r = grad(ds_c, r_c)

    return [
        [phie - phie_rhs],
        [phis_c - phis_c_rhs],
        [cs_a_t - cs_a_r_r * ds_a - np.float64(2.0) * ds_a * cs_a_r / r_a - ds_a_r * cs_a_r],
        [cs_c_t - cs_c_r_r * ds_c - np.float64(2.0) * ds_c * cs_c_r / r_c - ds_c_r * cs_c_r],
    ]



def boundary_loss(self, bound_col_pts=None, bound_col_params=None, tmax=None):
    if not self.activeBound:
        return [[_zero(self)]]

    t_bound, r_0_bound, r_max_a_bound, r_max_c_bound, deg_i0_a_bound, deg_ds_c_bound = self._prepare_boundary_batch(bound_col_pts, bound_col_params, tmax)

    resc_t = self.params["rescale_T"]
    resc_r = self.params["rescale_R"]

    r_0_bound = r_0_bound.clone().detach().requires_grad_(True)
    r_max_a_bound = r_max_a_bound.clone().detach().requires_grad_(True)
    r_max_c_bound = r_max_c_bound.clone().detach().requires_grad_(True)

    output_r0_a_bound = self.model(
        [
            t_bound / resc_t,
            r_0_bound / resc_r,
            self.rescale_param(deg_i0_a_bound, self.ind_deg_i0_a),
            self.rescale_param(deg_ds_c_bound, self.ind_deg_ds_c),
        ],
        training=True,
    )
    output_r0_c_bound = self.model(
        [
            t_bound / resc_t,
            r_0_bound / resc_r,
            self.rescale_param(deg_i0_a_bound, self.ind_deg_i0_a),
            self.rescale_param(deg_ds_c_bound, self.ind_deg_ds_c),
        ],
        training=True,
    )
    output_rmax_a_bound = self.model(
        [
            t_bound / resc_t,
            r_max_a_bound / resc_r,
            self.rescale_param(deg_i0_a_bound, self.ind_deg_i0_a),
            self.rescale_param(deg_ds_c_bound, self.ind_deg_ds_c),
        ],
        training=True,
    )
    output_rmax_c_bound = self.model(
        [
            t_bound / resc_t,
            r_max_c_bound / resc_r,
            self.rescale_param(deg_i0_a_bound, self.ind_deg_i0_a),
            self.rescale_param(deg_ds_c_bound, self.ind_deg_ds_c),
        ],
        training=True,
    )

    cs_r0_a_bound = self.rescaleCs_a(output_r0_a_bound[self.ind_cs_a], t_bound, r_0_bound, deg_i0_a_bound, deg_ds_c_bound, clip=False)
    cs_r0_c_bound = self.rescaleCs_c(output_r0_c_bound[self.ind_cs_c], t_bound, r_0_bound, deg_i0_a_bound, deg_ds_c_bound, clip=False)
    cs_rmax_a_bound = self.rescaleCs_a(output_rmax_a_bound[self.ind_cs_a], t_bound, r_max_a_bound, deg_i0_a_bound, deg_ds_c_bound, clip=False)
    cs_rmax_c_bound = self.rescaleCs_c(output_rmax_c_bound[self.ind_cs_c], t_bound, r_max_c_bound, deg_i0_a_bound, deg_ds_c_bound, clip=False)

    ds_rmax_a_bound = self.params["D_s_a"](self.params["T"], self.params["R"])
    ds_rmax_c_bound = self.params["D_s_c"](
        cs_rmax_c_bound,
        self.params["T"],
        self.params["R"],
        self.params["cscamax"],
        deg_ds_c_bound,
    )

    j_a = _profile_flux_of_t_ASSBfinal1(self, t_bound, "a")
    j_c = _profile_flux_of_t_ASSBfinal1(self, t_bound, "c")

    cs_r0_a_bound_r = grad(cs_r0_a_bound, r_0_bound)
    cs_r0_c_bound_r = grad(cs_r0_c_bound, r_0_bound)
    cs_rmax_a_bound_r = grad(cs_rmax_a_bound, r_max_a_bound)
    cs_rmax_c_bound_r = grad(cs_rmax_c_bound, r_max_c_bound)

    time_gate = np.float64(1.0) - torch.exp(-t_bound / self.hard_IC_timescale)
    return [
        [cs_r0_a_bound_r],
        [cs_r0_c_bound_r],
        [time_gate * (cs_rmax_a_bound_r + j_a / ds_rmax_a_bound)],
        [time_gate * (cs_rmax_c_bound_r + j_c / ds_rmax_c_bound)],
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
