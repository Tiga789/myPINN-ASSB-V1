from __future__ import annotations

import numpy as np
import torch

from current_profile_ASSBfinal1 import load_current_profile_ASSBfinal1
from spm_experimental_1 import current_to_flux_experimental_1, make_params_experimental_1
from thermo_experimental_1 import (
    phie_linearized_experimental_1,
    phis_c_linearized_experimental_1,
)


print("INFO: USING TIME-VARYING ASSB FINAL1 SPM MODEL")


_RUNTIME_CONFIG_ASSBfinal1: dict[str, object] = {
    "profile_path": None,
    "profile_encoding": "auto",
    "cycle_start": None,
    "cycle_end": None,
}


def configure_runtime_ASSBfinal1(
    profile_path: str,
    profile_encoding: str = "auto",
    cycle_start: int | None = None,
    cycle_end: int | None = None,
) -> None:
    _RUNTIME_CONFIG_ASSBfinal1["profile_path"] = profile_path
    _RUNTIME_CONFIG_ASSBfinal1["profile_encoding"] = profile_encoding
    _RUNTIME_CONFIG_ASSBfinal1["cycle_start"] = cycle_start
    _RUNTIME_CONFIG_ASSBfinal1["cycle_end"] = cycle_end


def _phie0_wrapper_ASSBfinal1(i0_a, j_a, F, R, T, Uocp_a0):
    del F
    return phie_linearized_experimental_1(i0_a, j_a, R, T, Uocp_a0)


def _phis_c0_wrapper_ASSBfinal1(i0_a, j_a, F, R, T, Uocp_a0, j_c, i0_c, Uocp_c0):
    del F
    phie0 = phie_linearized_experimental_1(i0_a, j_a, R, T, Uocp_a0)
    return phis_c_linearized_experimental_1(i0_c, j_c, R, T, Uocp_c0, phie0)


def _as_float(value: torch.Tensor | float) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().reshape(-1)[0])
    return float(value)


def _estimate_cs_excursions_ASSBfinal1(params: dict, j_a_prof: torch.Tensor, j_c_prof: torch.Tensor, t_prof: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    device = t_prof.device
    dtype = t_prof.dtype
    dt = t_prof[1:] - t_prof[:-1]
    if dt.numel() == 0:
        tiny = torch.as_tensor(1e-3, dtype=dtype, device=device)
        return tiny, tiny
    # Average-particle balance: d cbar / dt = -3 j / R_s
    delta_a = -torch.as_tensor(3.0, dtype=dtype, device=device) * j_a_prof[:-1] * dt / torch.clamp(params["Rs_a"], min=1e-30)
    delta_c = -torch.as_tensor(3.0, dtype=dtype, device=device) * j_c_prof[:-1] * dt / torch.clamp(params["Rs_c"], min=1e-30)
    cum_a = torch.cumsum(delta_a, dim=0)
    cum_c = torch.cumsum(delta_c, dim=0)
    max_a = torch.max(torch.abs(cum_a)) if cum_a.numel() > 0 else torch.as_tensor(0.0, dtype=dtype, device=device)
    max_c = torch.max(torch.abs(cum_c)) if cum_c.numel() > 0 else torch.as_tensor(0.0, dtype=dtype, device=device)
    # Give the network enough room for radial deviation beyond the particle average excursion.
    exc_a = torch.maximum(torch.as_tensor(2.5, dtype=dtype, device=device) * max_a, torch.as_tensor(0.25, dtype=dtype, device=device))
    exc_c = torch.maximum(torch.as_tensor(2.5, dtype=dtype, device=device) * max_c, torch.as_tensor(0.25, dtype=dtype, device=device))
    exc_a = torch.minimum(exc_a, torch.as_tensor(float(params["csanmax"]) * 0.95, dtype=dtype, device=device))
    exc_c = torch.minimum(exc_c, torch.as_tensor(float(params["cscamax"]) * 0.95, dtype=dtype, device=device))
    return exc_a, exc_c


def makeParams() -> dict:
    profile_path = _RUNTIME_CONFIG_ASSBfinal1.get("profile_path")
    if not profile_path:
        raise ValueError(
            "PROFILE_PATH is required for spm_ASSBfinal1. Set it in the input file and launch main_ASSBfinal1.py."
        )

    profile = load_current_profile_ASSBfinal1(
        profile_path=str(profile_path),
        encoding=str(_RUNTIME_CONFIG_ASSBfinal1.get("profile_encoding", "auto")),
        cycle_start=_RUNTIME_CONFIG_ASSBfinal1.get("cycle_start"),
        cycle_end=_RUNTIME_CONFIG_ASSBfinal1.get("cycle_end"),
    )

    params = make_params_experimental_1()
    dtype = params["dtype"]
    device = params["device"]

    t_prof = torch.as_tensor(profile["time_s"], dtype=dtype, device=device)
    i_prof = torch.as_tensor(profile["current_a"], dtype=dtype, device=device)
    dt_prof = t_prof[1:] - t_prof[:-1]

    deg_i0_a_ref = params["default_deg_i0_a"]
    deg_ds_c_ref = params["default_deg_ds_c"]
    cs_a0 = params["default_x_a0"] * params["csanmax"]
    cs_c0 = params["default_x_c0"] * params["cscamax"]

    params["deg_i0_a_min"] = torch.as_tensor(0.5, dtype=dtype, device=device)
    params["deg_i0_a_max"] = torch.as_tensor(4.0, dtype=dtype, device=device)
    params["deg_ds_c_min"] = torch.as_tensor(1.0, dtype=dtype, device=device)
    params["deg_ds_c_max"] = torch.as_tensor(10.0, dtype=dtype, device=device)

    params["param_eff"] = torch.as_tensor(0.0, dtype=dtype, device=device)
    params["deg_i0_a_ref"] = deg_i0_a_ref
    params["deg_ds_c_ref"] = deg_ds_c_ref
    params["deg_i0_a_min_eff"] = deg_i0_a_ref
    params["deg_i0_a_max_eff"] = deg_i0_a_ref
    params["deg_ds_c_min_eff"] = deg_ds_c_ref
    params["deg_ds_c_max_eff"] = deg_ds_c_ref

    params["tmin"] = torch.as_tensor(0.0, dtype=dtype, device=device)
    params["tmax"] = t_prof[-1]
    params["rmin"] = torch.as_tensor(0.0, dtype=dtype, device=device)
    params["C"] = torch.as_tensor(-2.0, dtype=dtype, device=device)
    params["I_discharge"] = i_prof[0]
    params["rescale_R"] = torch.maximum(params["Rs_a"], params["Rs_c"])
    params["rescale_T"] = torch.clamp(t_prof[-1], min=torch.as_tensor(1e-16, dtype=dtype, device=device))

    params["ce0"] = torch.as_tensor(1.2, dtype=dtype, device=device)
    params["ce_a0"] = params["ce0"]
    params["ce_c0"] = params["ce0"]
    params["cs_a0"] = cs_a0
    params["cs_c0"] = cs_c0
    params["eps_s_a"] = torch.as_tensor(0.5430727763, dtype=dtype, device=device)
    params["eps_s_c"] = torch.as_tensor(0.47662, dtype=dtype, device=device)
    params["L_a"] = torch.as_tensor(44e-6, dtype=dtype, device=device)
    params["L_c"] = torch.as_tensor(42e-6, dtype=dtype, device=device)

    j_a0, j_c0 = current_to_flux_experimental_1(i_prof[0], params)
    j_a_prof, j_c_prof = current_to_flux_experimental_1(i_prof, params)
    params["j_a"] = j_a0
    params["j_c"] = j_c0
    params["j_a_abs_max"] = torch.max(torch.abs(j_a_prof))
    params["j_c_abs_max"] = torch.max(torch.abs(j_c_prof))
    params["profile_time_s"] = t_prof
    params["profile_dt_s"] = dt_prof
    params["profile_current_a"] = i_prof
    params["profile_j_a"] = j_a_prof
    params["profile_j_c"] = j_c_prof
    params["profile_interp_mode"] = "zoh_left"
    params["profile_source"] = str(profile["profile_path"])
    params["profile_csv_path"] = str(profile["csv_path"])
    params["profile_encoding"] = str(profile["encoding"])
    params["profile_cycle_start"] = int(profile["cycle_start"])
    params["profile_cycle_end"] = int(profile["cycle_end"])

    # Use the same radial discretization as the soft-label generator.
    params["solver_n_r"] = 64
    params["profile_r_a_grid"] = torch.linspace(
        torch.as_tensor(0.0, dtype=dtype, device=device),
        params["Rs_a"],
        int(params["solver_n_r"]),
        dtype=dtype,
        device=device,
    )
    params["profile_r_c_grid"] = torch.linspace(
        torch.as_tensor(0.0, dtype=dtype, device=device),
        params["Rs_c"],
        int(params["solver_n_r"]),
        dtype=dtype,
        device=device,
    )
    params["profile_dR_a"] = params["Rs_a"] / torch.as_tensor(int(params["solver_n_r"]) - 1, dtype=dtype, device=device)
    params["profile_dR_c"] = params["Rs_c"] / torch.as_tensor(int(params["solver_n_r"]) - 1, dtype=dtype, device=device)

    i0_a0 = params["i0_a"](
        cs_a0,
        params["ce0"],
        params["T"],
        params["alpha_a"],
        params["csanmax"],
        params["R"],
        deg_i0_a_ref,
    )
    Uocp_a0 = params["Uocp_a"](cs_a0, params["csanmax"])
    params["Uocp_a0"] = Uocp_a0
    params["phie0"] = _phie0_wrapper_ASSBfinal1

    i0_c0 = params["i0_c"](
        cs_c0,
        params["ce0"],
        params["T"],
        params["alpha_c"],
        params["cscamax"],
        params["R"],
    )
    Uocp_c0 = params["Uocp_c"](cs_c0, params["cscamax"])
    params["i0_c0"] = i0_c0
    params["Uocp_c0"] = Uocp_c0
    params["phis_c0"] = _phis_c0_wrapper_ASSBfinal1

    cs_a_exc, cs_c_exc = _estimate_cs_excursions_ASSBfinal1(params, j_a_prof, j_c_prof, t_prof)
    params["cs_a_excursion_est"] = cs_a_exc
    params["cs_c_excursion_est"] = cs_c_exc

    phie_scale = torch.maximum(
        torch.abs((params["R"] * params["T"] / torch.clamp(i0_a0, min=1e-30)) * params["j_a_abs_max"]),
        torch.as_tensor(0.05, dtype=dtype, device=device),
    )
    phis_scale = torch.maximum(
        torch.abs((params["R"] * params["T"] / torch.clamp(i0_c0, min=1e-30)) * params["j_c_abs_max"]),
        torch.as_tensor(0.05, dtype=dtype, device=device),
    )
    params["phie_excursion_est"] = torch.as_tensor(1.5, dtype=dtype, device=device) * phie_scale
    params["phis_c_excursion_est"] = torch.as_tensor(1.5, dtype=dtype, device=device) * phis_scale

    params["mag_cs_a"] = torch.maximum(params["cs_a_excursion_est"], torch.as_tensor(1e-3, dtype=dtype, device=device))
    params["mag_cs_c"] = torch.maximum(params["cs_c_excursion_est"], torch.as_tensor(1e-3, dtype=dtype, device=device))
    params["mag_phie"] = torch.maximum(params["phie_excursion_est"], torch.as_tensor(1e-3, dtype=dtype, device=device))
    params["mag_phis_c"] = torch.maximum(params["phis_c_excursion_est"], torch.as_tensor(1e-3, dtype=dtype, device=device))
    params["mag_ce"] = torch.as_tensor(1.2, dtype=dtype, device=device)

    params["rescale_cs_a"] = params["cs_a_excursion_est"]
    params["rescale_cs_c"] = params["cs_c_excursion_est"]
    params["rescale_phis_c"] = params["phis_c_excursion_est"]
    params["rescale_phie"] = params["phie_excursion_est"]

    print(
        "INFO: ASSBfinal1 profile loaded | "
        f"rows={t_prof.numel()} t_end={_as_float(t_prof[-1]):.6f}s I0={_as_float(i_prof[0]):.8f}A "
        f"cycles={profile['cycle_start']}-{profile['cycle_end']} encoding={profile['encoding']} "
        f"n_r={int(params['solver_n_r'])}"
    )
    return params
