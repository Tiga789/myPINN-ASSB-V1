import numpy as np
import torch

from torch_utils import clip, to_tensor
from uocp_cs import uocp_a_fun_x, uocp_c_fun_x


torch.set_default_dtype(torch.float64)


def uocp_a_simp(cs_a, csanmax):
    x = clip(to_tensor(cs_a) / float(csanmax), 0.0, 1.0)
    return np.float64(0.2) - np.float64(0.2) * x


def uocp_a_fun(cs_a, csanmax):
    x = clip(to_tensor(cs_a) / float(csanmax), 0.0, 1.0)
    return uocp_a_fun_x(x)


def uocp_c_fun(cs_c, cscamax):
    x = clip(to_tensor(cs_c) / float(cscamax), 0.0, 1.0)
    return uocp_c_fun_x(x)


def uocp_c_simp(cs_c, cscamax):
    x = clip(to_tensor(cs_c) / float(cscamax), 0.0, 1.0)
    return np.float64(5.0) - np.float64(1.4) * x


def i0_a_fun(cs_a_max, ce, T, alpha, csanmax, R):
    cs_a_max = to_tensor(cs_a_max)
    ce = to_tensor(ce, device=cs_a_max.device)
    T = to_tensor(T, device=cs_a_max.device)
    return (
        np.float64(2.5)
        * np.float64(0.27)
        * torch.exp(
            np.float64(-30.0e6 / float(R))
            * (np.float64(1.0) / T - np.float64(1.0 / 303.15))
        )
        * torch.clamp(ce / np.float64(1.2), min=np.float64(0.0)) ** float(alpha)
        * torch.clamp(csanmax - cs_a_max, min=np.float64(0.0)) ** float(alpha)
        * torch.clamp(cs_a_max, min=np.float64(0.0)) ** (np.float64(1.0) - float(alpha))
    )


def i0_a_degradation_param_fun(cs_a_max, ce, T, alpha, csanmax, R, degradation_param):
    i0_a_nodeg = i0_a_fun(cs_a_max, ce, T, alpha, csanmax, R)
    return to_tensor(degradation_param, device=i0_a_nodeg.device).reshape_as(i0_a_nodeg) * i0_a_nodeg


def i0_a_simp(cs_a_max, ce, T, alpha, csanmax, R):
    ce_t = to_tensor(ce)
    return np.float64(2.0) * torch.ones_like(ce_t, dtype=torch.float64)


def i0_a_simp_degradation_param(cs_a_max, ce, T, alpha, csanmax, R, degradation_param):
    ce_t = to_tensor(ce)
    deg = to_tensor(degradation_param, device=ce_t.device).reshape_as(ce_t)
    return np.float64(2.0) * deg * torch.ones_like(ce_t, dtype=torch.float64)


def i0_c_fun(cs_c_max, ce, T, alpha, cscamax, R):
    cs_c_max = to_tensor(cs_c_max)
    ce = to_tensor(ce, device=cs_c_max.device)
    T = to_tensor(T, device=cs_c_max.device)
    x = clip(cs_c_max / float(cscamax), 0.0, 1.0)
    poly = (
        np.float64(1.650452829641290e01) * x**5
        - np.float64(7.523567141488800e01) * x**4
        + np.float64(1.240524690073040e02) * x**3
        - np.float64(9.416571081287610e01) * x**2
        + np.float64(3.249768821737960e01) * x
        - np.float64(3.585290065824760e00)
    )
    return (
        np.float64(9.0)
        * poly
        * torch.clamp(ce / np.float64(1.2), min=np.float64(0.0)) ** float(alpha)
        * torch.exp(
            np.float64(-30.0e6 / float(R))
            * (np.float64(1.0) / T - np.float64(1.0 / 303.15))
        )
    )


def i0_c_simp(cs_c_max, ce, T, alpha, cscamax, R):
    ce_t = to_tensor(ce)
    return np.float64(3.0) * torch.ones_like(ce_t, dtype=torch.float64)


def ds_a_fun(T, R):
    Tt = to_tensor(T)
    return np.float64(3.0e-14) * torch.exp(
        np.float64(-30.0e6 / float(R))
        * (np.float64(1.0) / Tt - np.float64(1.0 / 303.15))
    )


def grad_ds_a_cs_a(T, R):
    Tt = to_tensor(T)
    return torch.zeros_like(Tt, dtype=torch.float64)


def ds_a_fun_simp(T, R):
    Tt = to_tensor(T)
    return np.float64(3.0e-14) * torch.ones_like(Tt, dtype=torch.float64)


def ds_c_fun(cs_c, T, R, cscamax):
    cs_c = to_tensor(cs_c)
    T = to_tensor(T, device=cs_c.device)
    x = clip(cs_c / float(cscamax), 0.0, 1.0)
    power = (
        -np.float64(2.509010843479270e02) * x**10
        + np.float64(2.391026725259970e03) * x**9
        - np.float64(4.868420267611360e03) * x**8
        - np.float64(8.331104102921070e01) * x**7
        + np.float64(1.057636028329000e04) * x**6
        - np.float64(1.268324548348120e04) * x**5
        + np.float64(5.016272167775530e03) * x**4
        + np.float64(9.824896659649480e02) * x**3
        - np.float64(1.502439339070900e03) * x**2
        + np.float64(4.723709304247700e02) * x
        - np.float64(6.526092046397090e01)
    )
    return (
        np.float64(1.5)
        * (np.float64(1.5) * torch.pow(np.float64(10.0), power))
        * torch.exp(
            np.float64(-30.0e6 / float(R))
            * (np.float64(1.0) / T - np.float64(1.0 / 303.15))
        )
    )


def grad_ds_c_cs_c(cs_c, T, R, cscamax):
    cs_c = to_tensor(cs_c)
    cscamax = float(cscamax)
    T = to_tensor(T, device=cs_c.device)
    base = torch.pow(
        np.float64(10.0),
        (
            -np.float64(250.901084347927) * cs_c**10 / cscamax**10
            + np.float64(2391.02672525997) * cs_c**9 / cscamax**9
            - np.float64(4868.42026761136) * cs_c**8 / cscamax**8
            - np.float64(83.3110410292107) * cs_c**7 / cscamax**7
            + np.float64(10576.36028329) * cs_c**6 / cscamax**6
            - np.float64(12683.2454834812) * cs_c**5 / cscamax**5
            + np.float64(5016.27216777553) * cs_c**4 / cscamax**4
            + np.float64(982.489665964948) * cs_c**3 / cscamax**3
            - np.float64(1502.4393390709) * cs_c**2 / cscamax**2
            + np.float64(472.37093042477) * cs_c / cscamax
            - np.float64(65.2609204639709)
        ),
    )
    deriv = (
        -np.float64(5777.21096635578) * cs_c**9 / cscamax**10
        + np.float64(49549.8824508058) * cs_c**8 / cscamax**9
        - np.float64(89679.615477056) * cs_c**7 / cscamax**8
        - np.float64(1342.81532808973) * cs_c**6 / cscamax**7
        + np.float64(146117.817158627) * cs_c**5 / cscamax**6
        - np.float64(146021.259905239) * cs_c**4 / cscamax**5
        + np.float64(46201.5740636835) * cs_c**3 / cscamax**4
        + np.float64(6786.79817661477) * cs_c**2 / cscamax**3
        - np.float64(6918.98885054496) * cs_c / cscamax**2
        + np.float64(1087.6742627598) / cscamax
    )
    return (
        np.float64(2.25)
        * base
        * deriv
        * torch.exp(
            -np.float64(30000000.0)
            * (-np.float64(0.0032986970146792) + np.float64(1.0) / T)
            / float(R)
        )
    )


def ds_c_degradation_param_fun(cs_c, T, R, cscamax, degradation_param):
    ds_c_nodeg = ds_c_fun(cs_c, T, R, cscamax)
    return to_tensor(degradation_param, device=ds_c_nodeg.device).reshape_as(ds_c_nodeg) * ds_c_nodeg


def ds_c_fun_simp(cs_c, T, R, cscamax):
    cs_c_t = to_tensor(cs_c)
    return np.float64(3.5e-15) * torch.ones_like(cs_c_t, dtype=torch.float64)


def ds_c_fun_plot(cs_c, T, R, cscamax):
    return ds_c_fun(cs_c, T, R, cscamax)


def ds_c_fun_plot_simp(cs_c, T, R, cscamax):
    return ds_c_fun_simp(cs_c, T, R, cscamax)


def ds_c_fun_simp_degradation_param(cs_c, T, R, cscamax, degradation_param):
    cs_c_t = to_tensor(cs_c)
    deg = to_tensor(degradation_param, device=cs_c_t.device).reshape_as(cs_c_t)
    return np.float64(3.5e-15) * deg * torch.ones_like(cs_c_t, dtype=torch.float64)


def phie0_fun(i0_a, j_a, F, R, T, Uocp_a0):
    i0_a = to_tensor(i0_a)
    return -float(j_a) * (float(F) / i0_a) * (float(R) * float(T) / float(F)) - to_tensor(Uocp_a0, device=i0_a.device)


def phis_c0_fun(i0_a, j_a, F, R, T, Uocp_a0, j_c, i0_c, Uocp_c0):
    i0_c = to_tensor(i0_c)
    phie0 = phie0_fun(i0_a, j_a, F, R, T, Uocp_a0)
    return float(j_c) * (float(F) / i0_c) * (float(R) * float(T) / float(F)) + to_tensor(Uocp_c0, device=i0_c.device) + phie0


def setParams(params, deg, bat, an, ca, ic):
    params["deg_i0_a_min"] = deg.bounds[deg.ind_i0_a][0]
    params["deg_i0_a_max"] = deg.bounds[deg.ind_i0_a][1]
    params["deg_ds_c_min"] = deg.bounds[deg.ind_ds_c][0]
    params["deg_ds_c_max"] = deg.bounds[deg.ind_ds_c][1]

    params["param_eff"] = deg.eff
    params["deg_i0_a_ref"] = deg.ref_vals[deg.ind_i0_a]
    params["deg_ds_c_ref"] = deg.ref_vals[deg.ind_ds_c]
    params["deg_i0_a_min_eff"] = (
        params["deg_i0_a_ref"]
        + (params["deg_i0_a_min"] - params["deg_i0_a_ref"]) * params["param_eff"]
    )
    params["deg_i0_a_max_eff"] = (
        params["deg_i0_a_ref"]
        + (params["deg_i0_a_max"] - params["deg_i0_a_ref"]) * params["param_eff"]
    )
    params["deg_ds_c_min_eff"] = (
        params["deg_ds_c_ref"]
        + (params["deg_ds_c_min"] - params["deg_ds_c_ref"]) * params["param_eff"]
    )
    params["deg_ds_c_max_eff"] = (
        params["deg_ds_c_ref"]
        + (params["deg_ds_c_max"] - params["deg_ds_c_ref"]) * params["param_eff"]
    )

    params["tmin"] = bat.tmin
    params["tmax"] = bat.tmax
    params["rmin"] = bat.rmin

    params["A_a"] = an.A
    params["A_c"] = ca.A
    params["F"] = bat.F
    params["R"] = bat.R
    params["T"] = bat.T
    params["C"] = bat.C
    params["I_discharge"] = bat.I

    params["alpha_a"] = an.alpha
    params["alpha_c"] = ca.alpha

    params["Rs_a"] = an.D50 / np.float64(2.0)
    params["Rs_c"] = ca.D50 / np.float64(2.0)
    params["rescale_R"] = np.float64(max(params["Rs_a"], params["Rs_c"]))
    params["csanmax"] = an.csmax
    params["cscamax"] = ca.csmax
    params["rescale_T"] = np.float64(max(bat.tmax, 1e-16))

    params["mag_cs_a"] = np.float64(25)
    params["mag_cs_c"] = np.float64(32.5)
    params["mag_phis_c"] = np.float64(4.25)
    params["mag_phie"] = np.float64(0.15)
    params["mag_ce"] = np.float64(1.2)

    params["Uocp_a"] = an.uocp
    params["Uocp_c"] = ca.uocp
    params["i0_a"] = an.i0
    params["i0_c"] = ca.i0
    params["D_s_a"] = an.ds
    params["D_s_c"] = ca.ds

    params["ce0"] = ic.ce
    params["ce_a0"] = ic.ce
    params["ce_c0"] = ic.ce
    params["cs_a0"] = ic.an.cs
    params["cs_c0"] = ic.ca.cs
    params["eps_s_a"] = an.solids.eps
    params["eps_s_c"] = ca.solids.eps
    params["L_a"] = an.thickness
    params["L_c"] = ca.thickness
    j_a = (
        -(params["I_discharge"] / params["A_a"])
        * params["Rs_a"]
        / (np.float64(3.0) * params["eps_s_a"] * params["F"] * params["L_a"])
    )
    j_c = (
        (params["I_discharge"] / params["A_c"])
        * params["Rs_c"]
        / (np.float64(3.0) * params["eps_s_c"] * params["F"] * params["L_c"])
    )
    params["j_a"] = j_a
    params["j_c"] = j_c

    cse_a = ic.an.cs
    i0_a = params["i0_a"](
        cse_a,
        params["ce0"],
        params["T"],
        params["alpha_a"],
        params["csanmax"],
        params["R"],
        np.float64(1.0),
    )
    Uocp_a = params["Uocp_a"](cse_a, params["csanmax"])
    params["Uocp_a0"] = Uocp_a
    params["phie0"] = phie0_fun

    cse_c = ic.ca.cs
    i0_c = params["i0_c"](
        cse_c,
        params["ce0"],
        params["T"],
        params["alpha_c"],
        params["cscamax"],
        params["R"],
    )
    params["i0_c0"] = i0_c
    Uocp_c = params["Uocp_c"](cse_c, params["cscamax"])
    params["Uocp_c0"] = Uocp_c
    params["phis_c0"] = phis_c0_fun

    params["rescale_cs_a"] = -ic.an.cs
    params["rescale_cs_c"] = params["cscamax"] - ic.ca.cs
    params["rescale_phis_c"] = abs(np.float64(3.8) - np.float64(4.110916387038547))
    params["rescale_phie"] = abs(np.float64(-0.15) - np.float64(-0.07645356566609385))
    return params
