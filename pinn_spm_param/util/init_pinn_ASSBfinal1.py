from __future__ import annotations

import sys
from typing import Any

sys.path.append("util")

import init_pinn as _base_init_pinn
import spm_ASSBfinal1 as _spm_ASSBfinal1
from myNN_ASSBfinal1 import myNN



def _normalize_optional_path_ASSBfinal1(value: Any):
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.upper() in {"NONE", "NULL", ""}:
            return None
        return stripped
    return value



def _maybe_int_ASSBfinal1(value: Any):
    if value is None:
        return None
    if isinstance(value, str) and value.strip().upper() in {"NONE", "NULL", ""}:
        return None
    return int(value)



def initialize_params_from_inpt(inpt):
    out = _base_init_pinn.initialize_params_from_inpt(inpt)
    out["PROFILE_PATH"] = _normalize_optional_path_ASSBfinal1(inpt.get("PROFILE_PATH"))
    out["PROFILE_ENCODING"] = str(inpt.get("PROFILE_ENCODING", "auto")).strip() or "auto"
    out["PROFILE_CYCLE_START"] = _maybe_int_ASSBfinal1(inpt.get("PROFILE_CYCLE_START"))
    out["PROFILE_CYCLE_END"] = _maybe_int_ASSBfinal1(inpt.get("PROFILE_CYCLE_END"))
    return out



def initialize_params(args):
    inpt = _base_init_pinn.parse_input_file(args.input_file)
    return initialize_params_from_inpt(inpt)



def initialize_nn(args, input_params):
    profile_path = input_params.get("PROFILE_PATH")
    if not profile_path:
        raise ValueError(
            "PROFILE_PATH is required in the input file for the ASSBfinal1 physics-consistent workflow."
        )

    _spm_ASSBfinal1.configure_runtime_ASSBfinal1(
        profile_path=profile_path,
        profile_encoding=input_params.get("PROFILE_ENCODING", "auto"),
        cycle_start=input_params.get("PROFILE_CYCLE_START"),
        cycle_end=input_params.get("PROFILE_CYCLE_END"),
    )

    old_myNN = getattr(_base_init_pinn, "myNN", None)
    old_spm = sys.modules.get("spm")
    _base_init_pinn.myNN = myNN
    sys.modules["spm"] = _spm_ASSBfinal1
    try:
        return _base_init_pinn.initialize_nn(args=args, input_params=input_params)
    finally:
        if old_myNN is not None:
            _base_init_pinn.myNN = old_myNN
        if old_spm is not None:
            sys.modules["spm"] = old_spm
        else:
            sys.modules.pop("spm", None)
