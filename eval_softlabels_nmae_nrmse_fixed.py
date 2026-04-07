import argparse
import csv
import json
import math
import os
import sys
from typing import Dict, List

import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate trained PINN on soft-label test splits with physical rescaling applied.")
    p.add_argument("--repo-root", required=True)
    p.add_argument("--model-dir", required=True)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--weight-name", default=None)
    p.add_argument("--norm", default="range", choices=["range", "mean_abs", "std"])
    p.add_argument("--output-prefix", default="eval_softlabels_fixed")
    p.add_argument("--batch-size", type=int, default=65536)
    return p.parse_args()


def resolve_checkpoint(model_dir: str, weight_name: str | None) -> str:
    if weight_name:
        ckpt = os.path.join(model_dir, weight_name)
        if not os.path.isfile(ckpt):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
        return ckpt
    for name in ["best.pt", "best.weights.h5", "last.pt", "last.weights.h5"]:
        cand = os.path.join(model_dir, name)
        if os.path.isfile(cand):
            return cand
    raise FileNotFoundError(f"No checkpoint found in {model_dir}")


TARGET_FILES = {
    "phie": "data_phie.npz",
    "phis_c": "data_phis_c.npz",
    "cs_a": "data_cs_a.npz",
    "cs_c": "data_cs_c.npz",
}
TARGET_INDEX = {"phie": 0, "phis_c": 1, "cs_a": 2, "cs_c": 3}


def build_model(repo_root: str, model_dir: str, ckpt_path: str):
    util_dir = os.path.join(repo_root, "pinn_spm_param", "util")
    if util_dir not in sys.path:
        sys.path.insert(0, util_dir)

    from spm import makeParams  # type: ignore
    from init_pinn import initialize_nn_from_params_config, safe_load  # type: ignore

    config_path = os.path.join(model_dir, "config.json")
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"config.json not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    params = makeParams()
    nn = initialize_nn_from_params_config(params, config)
    nn = safe_load(nn, ckpt_path)
    nn.model.eval()
    return nn, config


def to_2d_np(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    return x


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot <= 0.0:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def norm_denom(y_true: np.ndarray, mode: str) -> float:
    y = y_true.reshape(-1)
    if mode == "range":
        denom = float(np.max(y) - np.min(y))
    elif mode == "mean_abs":
        denom = float(np.mean(np.abs(y)))
    elif mode == "std":
        denom = float(np.std(y))
    else:
        raise ValueError(mode)
    if not math.isfinite(denom) or abs(denom) < 1e-18:
        denom = 1.0
    return denom


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, mode: str) -> Dict[str, float]:
    err = y_pred.reshape(-1) - y_true.reshape(-1)
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    denom = norm_denom(y_true, mode)
    return {
        "n_samples": int(y_true.size),
        "denom": denom,
        "mae": mae,
        "rmse": rmse,
        "nmae": mae / denom,
        "nrmse": rmse / denom,
        "r2": r2_score(y_true, y_pred),
    }


@torch.no_grad()
def predict_target(nn, target: str, x: np.ndarray, x_params: np.ndarray, batch_size: int) -> np.ndarray:
    x = to_2d_np(x)
    x_params = to_2d_np(x_params)
    if x_params.shape[1] != 2:
        raise ValueError(f"x_params must have 2 columns, got {x_params.shape}")

    resc_t = float(nn.params["rescale_T"])
    resc_r = float(nn.params["rescale_R"])

    preds = []
    for start in range(0, len(x), batch_size):
        end = min(start + batch_size, len(x))
        xb = x[start:end]
        pb = x_params[start:end]

        t = torch.as_tensor(xb[:, 0:1], dtype=torch.float64, device=nn.device)
        if xb.shape[1] >= 2:
            r = torch.as_tensor(xb[:, 1:2], dtype=torch.float64, device=nn.device)
        else:
            r = torch.zeros_like(t)
        deg_i0_a = torch.as_tensor(pb[:, 0:1], dtype=torch.float64, device=nn.device)
        deg_ds_c = torch.as_tensor(pb[:, 1:2], dtype=torch.float64, device=nn.device)

        deg_i0_a_scaled = nn.rescale_param(deg_i0_a, nn.ind_deg_i0_a)
        deg_ds_c_scaled = nn.rescale_param(deg_ds_c, nn.ind_deg_ds_c)

        if target in ("phie", "phis_c"):
            r_use = float(nn.params["Rs_a"]) * torch.ones_like(t)
        else:
            r_use = r

        outputs = nn.model([
            t / resc_t,
            r_use / resc_r,
            deg_i0_a_scaled,
            deg_ds_c_scaled,
        ], training=False)
        raw = outputs[TARGET_INDEX[target]]

        if target == "phie":
            pred = nn.rescalePhie(raw, t, deg_i0_a, deg_ds_c)
        elif target == "phis_c":
            pred = nn.rescalePhis_c(raw, t, deg_i0_a, deg_ds_c)
        elif target == "cs_a":
            pred = nn.rescaleCs_a(raw, t, r, deg_i0_a, deg_ds_c, clip=False)
        elif target == "cs_c":
            pred = nn.rescaleCs_c(raw, t, r, deg_i0_a, deg_ds_c, clip=False)
        else:
            raise ValueError(target)

        preds.append(pred.detach().cpu().numpy().reshape(-1, 1))

    return np.vstack(preds)


def main() -> None:
    args = parse_args()
    model_dir = os.path.abspath(args.model_dir)
    data_dir = os.path.abspath(args.data_dir)
    ckpt = resolve_checkpoint(model_dir, args.weight_name)

    nn, config = build_model(args.repo_root, model_dir, ckpt)

    summary: Dict[str, object] = {
        "repo_root": os.path.abspath(args.repo_root),
        "model_dir": model_dir,
        "data_dir": data_dir,
        "checkpoint": ckpt,
        "norm_mode": args.norm,
        "device": str(nn.device),
        "config_file": os.path.join(model_dir, "config.json"),
        "metrics": {},
    }

    rows: List[Dict[str, object]] = []
    for target, fname in TARGET_FILES.items():
        path = os.path.join(data_dir, fname)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Missing dataset file: {path}")
        z = np.load(path)
        x_test = z["x_test"]
        y_test = z["y_test"]
        x_params_test = z["x_params_test"]
        y_pred = predict_target(nn, target, x_test, x_params_test, args.batch_size)
        metrics = compute_metrics(y_test, y_pred, args.norm)
        summary["metrics"][target] = metrics
        rows.append({"target": target, **metrics})

    csv_path = os.path.join(model_dir, f"{args.output_prefix}_metrics.csv")
    json_path = os.path.join(model_dir, f"{args.output_prefix}_metrics.json")

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["target", "n_samples", "denom", "mae", "rmse", "nmae", "nrmse", "r2"])
        writer.writeheader()
        writer.writerows(rows)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"Checkpoint: {ckpt}")
    print(f"Normalization: {args.norm}")
    print(f"Device: {nn.device}")
    print("=" * 72)
    for row in rows:
        print(
            f"{row['target']:<7} "
            f"NMAE={100.0 * row['nmae']:.4f}% "
            f"NRMSE={100.0 * row['nrmse']:.4f}% "
            f"R2={row['r2']:.6f} "
            f"MAE={row['mae']:.6e} RMSE={row['rmse']:.6e}"
        )
    print("=" * 72)
    print(f"Saved CSV : {csv_path}")
    print(f"Saved JSON: {json_path}")


if __name__ == "__main__":
    main()
