from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import torch

from current_profile_ASSBfinal2 import load_current_profile_ASSBfinal2


def configure_repo_imports_ASSBfinal2(repo_root: str | None) -> None:
    current_dir = Path(__file__).resolve().parent
    candidates: list[Path] = []
    if repo_root:
        rr = Path(repo_root)
        candidates.extend(
            [
                rr / "pinn_spm_param" / "integration_spm",
                rr / "pinn_spm_param" / "util",
                rr / "integration_spm",
                rr / "util",
            ]
        )
    candidates.extend(
        [
            current_dir,
            current_dir.parent / "integration_spm",
            current_dir.parent / "util",
            Path.cwd() / "pinn_spm_param" / "integration_spm",
            Path.cwd() / "pinn_spm_param" / "util",
            Path.cwd() / "integration_spm",
            Path.cwd() / "util",
        ]
    )
    for c in candidates:
        if c.exists() and c.is_dir() and str(c) not in sys.path:
            sys.path.insert(0, str(c))


def resolve_device_ASSBfinal2(device: str | None) -> torch.device:
    if device is None or str(device).lower() == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(str(device))


def resolve_dtype_ASSBfinal2(dtype: str | None) -> torch.dtype:
    text = "float64" if dtype is None else str(dtype).strip().lower()
    if text in {"float64", "double", "fp64"}:
        return torch.float64
    if text in {"float32", "float", "single", "fp32"}:
        return torch.float32
    raise ValueError(f"Unsupported DTYPE: {dtype}")


def _as_float(value: Any) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().reshape(-1)[0])
    return float(value)


def make_context_ASSBfinal2(
    repo_root: str | None,
    profile_path: str,
    profile_encoding: str = "auto",
    cycle_start: int | None = None,
    cycle_end: int | None = None,
    n_r: int = 64,
    device: str | None = "auto",
    dtype: str | None = "float64",
) -> dict[str, Any]:
    configure_repo_imports_ASSBfinal2(repo_root)

    from spm_experimental_1 import current_to_flux_experimental_1, make_params_experimental_1
    from thermo_experimental_1 import (
        grad_ds_c_cs_c_experimental_1,
        phie_linearized_experimental_1,
        phis_c_linearized_experimental_1,
    )

    resolved_device = resolve_device_ASSBfinal2(device)
    resolved_dtype = resolve_dtype_ASSBfinal2(dtype)

    profile = load_current_profile_ASSBfinal2(
        profile_path=profile_path,
        encoding=profile_encoding,
        cycle_start=cycle_start,
        cycle_end=cycle_end,
    )

    params = make_params_experimental_1(device=str(resolved_device), dtype=resolved_dtype)

    t = torch.as_tensor(profile["time_s"], dtype=resolved_dtype, device=resolved_device)
    current = torch.as_tensor(profile["current_a"], dtype=resolved_dtype, device=resolved_device)
    dt = t[1:] - t[:-1]
    cycle = torch.as_tensor(profile["cycle"], dtype=torch.long, device=resolved_device)

    r_a = torch.linspace(torch.as_tensor(0.0, dtype=resolved_dtype, device=resolved_device), params["Rs_a"], n_r, dtype=resolved_dtype, device=resolved_device)
    r_c = torch.linspace(torch.as_tensor(0.0, dtype=resolved_dtype, device=resolved_device), params["Rs_c"], n_r, dtype=resolved_dtype, device=resolved_device)
    dR_a = params["Rs_a"] / torch.as_tensor(n_r - 1, dtype=resolved_dtype, device=resolved_device)
    dR_c = params["Rs_c"] / torch.as_tensor(n_r - 1, dtype=resolved_dtype, device=resolved_device)

    j_a_prof, j_c_prof = current_to_flux_experimental_1(current, params)

    x_a0 = params["default_x_a0"]
    x_c0 = params["default_x_c0"]
    cs_a0 = x_a0 * params["csanmax"]
    cs_c0 = x_c0 * params["cscamax"]

    context: dict[str, Any] = {
        "repo_root": str(repo_root) if repo_root is not None else None,
        "profile_path": str(profile["profile_path"]),
        "profile_csv_path": str(profile["csv_path"]),
        "profile_encoding": str(profile["encoding"]),
        "cycle_start": int(profile["cycle_start"]),
        "cycle_end": int(profile["cycle_end"]),
        "device": resolved_device,
        "dtype": resolved_dtype,
        "params": params,
        "time_s": t,
        "current_a": current,
        "dt_s": dt,
        "cycle": cycle,
        "n_t": int(t.numel()),
        "n_steps": int(max(t.numel() - 1, 0)),
        "n_r": int(n_r),
        "r_a": r_a,
        "r_c": r_c,
        "dR_a": dR_a,
        "dR_c": dR_c,
        "j_a_prof": j_a_prof,
        "j_c_prof": j_c_prof,
        "deg_i0_a": params["default_deg_i0_a"],
        "deg_ds_c": params["default_deg_ds_c"],
        "cs_a0": cs_a0,
        "cs_c0": cs_c0,
        "x_a0": x_a0,
        "x_c0": x_c0,
        "current_to_flux_experimental_1": current_to_flux_experimental_1,
        "grad_ds_c_cs_c_experimental_1": grad_ds_c_cs_c_experimental_1,
        "phie_linearized_experimental_1": phie_linearized_experimental_1,
        "phis_c_linearized_experimental_1": phis_c_linearized_experimental_1,
    }

    print(
        "INFO: ASSBfinal2 profile loaded | "
        f"rows={context['n_t']} t_end={_as_float(t[-1]):.6f}s I0={_as_float(current[0]):.8f}A "
        f"cycles={profile['cycle_start']}-{profile['cycle_end']} encoding={profile['encoding']} n_r={n_r}"
    )
    return context


def metadata_from_context_ASSBfinal2(context: dict[str, Any]) -> dict[str, Any]:
    params = context["params"]
    return {
        "repo_root": context.get("repo_root"),
        "profile_path": context["profile_path"],
        "profile_csv_path": context["profile_csv_path"],
        "profile_encoding": context["profile_encoding"],
        "cycle_start": context["cycle_start"],
        "cycle_end": context["cycle_end"],
        "n_t": context["n_t"],
        "n_steps": context["n_steps"],
        "n_r": context["n_r"],
        "dtype": str(context["dtype"]).replace("torch.", ""),
        "device": str(context["device"]),
        "default_deg_i0_a": _as_float(context["deg_i0_a"]),
        "default_deg_ds_c": _as_float(context["deg_ds_c"]),
        "default_x_a0": _as_float(context["x_a0"]),
        "default_x_c0": _as_float(context["x_c0"]),
        "Rs_a": _as_float(params["Rs_a"]),
        "Rs_c": _as_float(params["Rs_c"]),
        "csanmax": _as_float(params["csanmax"]),
        "cscamax": _as_float(params["cscamax"]),
        "T": _as_float(params["T"]),
        "R": _as_float(params["R"]),
        "alpha_a": _as_float(params["alpha_a"]),
        "alpha_c": _as_float(params["alpha_c"]),
    }


def save_metadata_json_ASSBfinal2(path: str | Path, payload: dict[str, Any]) -> None:
    Path(path).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
