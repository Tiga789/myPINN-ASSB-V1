from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from myNN_ASSBfinal2 import StepLatentFieldModel_ASSBfinal2
from spm_ASSBfinal2 import make_context_ASSBfinal2, metadata_from_context_ASSBfinal2


def _parse_scalar(text: str) -> Any:
    s = text.strip()
    if s == "":
        return ""
    if s.upper() in {"NONE", "NULL"}:
        return None
    if s.upper() in {"TRUE", "FALSE"}:
        return s.upper() == "TRUE"
    try:
        if any(ch in s for ch in [".", "e", "E"]):
            return float(s)
        return int(s)
    except ValueError:
        return s


DEFAULT_CFG_ASSBfinal2: dict[str, Any] = {
    "ID": 9401,
    "REPO_ROOT": None,
    "PROFILE_PATH": None,
    "PROFILE_ENCODING": "auto",
    "PROFILE_CYCLE_START": None,
    "PROFILE_CYCLE_END": None,
    "N_R": 64,
    "DTYPE": "float64",
    "DEVICE": "auto",
    "EPOCHS": 200,
    "BATCH_SIZE_STEPS": 2048,
    "SHUFFLE_STEPS": False,
    "LEARNING_RATE": 1e-3,
    "LEARNING_RATE_FINAL": 1e-4,
    "WEIGHT_DECAY": 0.0,
    "LATENT_DIM": 12,
    "HIDDEN_DIM": 128,
    "NUM_LAYERS": 3,
    "R_FOURIER_MODES": 6,
    "STEP_WEIGHT_A": 1.0,
    "STEP_WEIGHT_C": 1.0,
    "BOUND_WEIGHT_A": 0.1,
    "BOUND_WEIGHT_C": 0.1,
    "LOG_EVERY": 1,
    "SAVE_EVERY": 10,
    "TIME_CHUNK_EVAL": 2048,
    "GRAD_CLIP_NORM": 1.0,
}


def parse_input_file_ASSBfinal2(path: str) -> dict[str, Any]:
    cfg = dict(DEFAULT_CFG_ASSBfinal2)
    text = Path(path).read_text(encoding="utf-8")
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("!") or line.startswith("#"):
            continue
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        cfg[key.strip()] = _parse_scalar(value)
    if not cfg.get("PROFILE_PATH"):
        raise ValueError("PROFILE_PATH is required in input_physics_only_ASSBfinal2.")
    return cfg


def initialize_ASSBfinal2(input_file: str) -> tuple[dict[str, Any], dict[str, Any], StepLatentFieldModel_ASSBfinal2]:
    cfg = parse_input_file_ASSBfinal2(input_file)
    context = make_context_ASSBfinal2(
        repo_root=cfg.get("REPO_ROOT"),
        profile_path=str(cfg["PROFILE_PATH"]),
        profile_encoding=str(cfg.get("PROFILE_ENCODING", "auto")),
        cycle_start=cfg.get("PROFILE_CYCLE_START"),
        cycle_end=cfg.get("PROFILE_CYCLE_END"),
        n_r=int(cfg.get("N_R", 64)),
        device=str(cfg.get("DEVICE", "auto")),
        dtype=str(cfg.get("DTYPE", "float64")),
    )
    model = StepLatentFieldModel_ASSBfinal2(
        n_time_nodes=context["n_t"],
        latent_dim=int(cfg["LATENT_DIM"]),
        hidden_dim=int(cfg["HIDDEN_DIM"]),
        num_layers=int(cfg["NUM_LAYERS"]),
        r_fourier_modes=int(cfg["R_FOURIER_MODES"]),
        x_a0=float(context["x_a0"]),
        x_c0=float(context["x_c0"]),
        csanmax=float(context["params"]["csanmax"]),
        cscamax=float(context["params"]["cscamax"]),
        dtype=context["dtype"],
    ).to(device=context["device"], dtype=context["dtype"])
    return cfg, context, model


def checkpoint_payload_ASSBfinal2(cfg: dict[str, Any], context: dict[str, Any], model, epoch: int, best_loss: float, history: list[dict[str, float]]) -> dict[str, Any]:
    return {
        "epoch": int(epoch),
        "best_loss": float(best_loss),
        "config": cfg,
        "context_meta": metadata_from_context_ASSBfinal2(context),
        "model_state_dict": model.state_dict(),
        "history": history,
    }


def save_config_and_meta_ASSBfinal2(model_dir: Path, cfg: dict[str, Any], context: dict[str, Any]) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "config_ASSBfinal2.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")
    (model_dir / "context_meta_ASSBfinal2.json").write_text(json.dumps(metadata_from_context_ASSBfinal2(context), indent=2, ensure_ascii=False), encoding="utf-8")
