import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Plot truth-vs-prediction comparisons for one cycle using ASSBfinal1 model."
    )
    p.add_argument("--repo-root", required=True, help="Project root, e.g. C:/.../myPINN-ASSB-V1")
    p.add_argument("--model-dir", required=True, help="ModelFin_9301 folder")
    p.add_argument("--solution-path", required=True, help="Path to solution.npz")
    p.add_argument("--profile-path", required=True, help="Original CSV or ZIP used for profile generation")
    p.add_argument("--cycle", type=int, default=5, help="Cycle number to plot")
    p.add_argument("--profile-encoding", default="auto")
    p.add_argument("--profile-cycle-start", type=int, default=None)
    p.add_argument("--profile-cycle-end", type=int, default=None)
    p.add_argument("--weight-name", default=None)
    p.add_argument("--batch-size", type=int, default=65536)
    p.add_argument("--surface-time-step", type=int, default=1,
                   help="Temporal downsample for surface plotting only, >=1")
    p.add_argument("--time-unit", choices=["s", "min", "h"], default="min")
    p.add_argument("--backend", default="auto", choices=["auto", "QtAgg", "TkAgg", "Agg"])
    p.add_argument("--no-show", action="store_true")
    p.add_argument("--output-dir", default=None,
                   help="Directory to save figures; defaults to <model-dir>/cycle_<N>_plots")
    return p.parse_args()


def resolve_checkpoint(model_dir: str, weight_name: Optional[str]) -> str:
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


def build_model(repo_root: str, model_dir: str, ckpt_path: str, profile_path: str,
                profile_encoding: str, profile_cycle_start: Optional[int],
                profile_cycle_end: Optional[int]):
    util_dir = os.path.join(repo_root, "pinn_spm_param", "util")
    if util_dir not in sys.path:
        sys.path.insert(0, util_dir)

    import init_pinn as _base_init_pinn  # type: ignore
    import spm_ASSBfinal1 as _spm_ASSBfinal1  # type: ignore
    from myNN_ASSBfinal1 import myNN  # type: ignore

    config_path = os.path.join(model_dir, "config.json")
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"config.json not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    _spm_ASSBfinal1.configure_runtime_ASSBfinal1(
        profile_path=profile_path,
        profile_encoding=profile_encoding,
        cycle_start=profile_cycle_start,
        cycle_end=profile_cycle_end,
    )

    old_myNN = getattr(_base_init_pinn, "myNN", None)
    old_spm = sys.modules.get("spm")
    _base_init_pinn.myNN = myNN
    sys.modules["spm"] = _spm_ASSBfinal1
    try:
        params = _spm_ASSBfinal1.makeParams()
        nn = _base_init_pinn.initialize_nn_from_params_config(params, config)
        nn = _base_init_pinn.safe_load(nn, ckpt_path)
        nn.model.eval()
    finally:
        if old_myNN is not None:
            _base_init_pinn.myNN = old_myNN
        if old_spm is not None:
            sys.modules["spm"] = old_spm
        else:
            sys.modules.pop("spm", None)
    return nn


TARGET_INDEX = {"phie": 0, "phis_c": 1, "cs_a": 2, "cs_c": 3}


def to_2d_np(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    return x


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


def _load_cycle_mask_and_local_time(repo_root: str, profile_path: str, encoding: str,
                                   cycle: int) -> Tuple[np.ndarray, np.ndarray]:
    util_dir = os.path.join(repo_root, "pinn_spm_param", "util")
    if util_dir not in sys.path:
        sys.path.insert(0, util_dir)
    from current_profile_ASSBfinal1 import (  # type: ignore
        resolve_csv_from_input_ASSBfinal1,
        read_csv_auto_ASSBfinal1,
        detect_columns_ASSBfinal1,
        series_to_seconds_ASSBfinal1,
    )

    csv_path, tmpdir = resolve_csv_from_input_ASSBfinal1(Path(profile_path))
    try:
        df, _used_encoding = read_csv_auto_ASSBfinal1(csv_path, encoding=encoding)
        cols = detect_columns_ASSBfinal1(df)
        cycle_numeric = np.asarray(np.round(np.asarray(np.nan_to_num(np.array(df[cols["cycle"]], dtype=float), nan=np.nan))), dtype=float)
        # Robust re-read for the cycle column
        cycle_series = np.asarray(np.nan_to_num(np.asarray(np.array(df[cols["cycle"]], dtype=float), dtype=float), nan=np.nan), dtype=float)
        cycle_mask = np.isfinite(cycle_series) & (cycle_series.astype(int) == int(cycle))
        if not np.any(cycle_mask):
            unique_cycles = np.unique(cycle_series[np.isfinite(cycle_series)].astype(int))
            raise ValueError(f"Cycle {cycle} not found in profile. Available cycles head/tail: {unique_cycles[:5]} ... {unique_cycles[-5:]}")
        total_time_s = series_to_seconds_ASSBfinal1(df[cols["total_time"]])
        total_time_s = total_time_s - float(total_time_s[0])
        local_time_s = total_time_s[cycle_mask] - float(total_time_s[cycle_mask][0])
        return cycle_mask, local_time_s
    finally:
        if tmpdir is not None:
            tmpdir.cleanup()


def _ensure_time_major(arr: np.ndarray, n_t: int, n_r: int, name: str) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float64)
    if arr.shape == (n_t, n_r):
        return arr
    if arr.shape == (n_r, n_t):
        return arr.T
    raise ValueError(f"Unexpected shape for {name}: {arr.shape}, expected ({n_t},{n_r}) or ({n_r},{n_t})")


def _metric_summary(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    yt = np.asarray(y_true, dtype=np.float64).reshape(-1)
    yp = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    err = yp - yt
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    denom = float(np.max(yt) - np.min(yt))
    if not np.isfinite(denom) or abs(denom) < 1e-18:
        denom = 1.0
    ss_res = float(np.sum((yt - yp) ** 2))
    ss_tot = float(np.sum((yt - np.mean(yt)) ** 2))
    r2 = float("nan") if ss_tot <= 0.0 else (1.0 - ss_res / ss_tot)
    return {
        "mae": mae,
        "rmse": rmse,
        "nmae": mae / denom,
        "nrmse": rmse / denom,
        "r2": r2,
    }


def _time_scale_and_label(unit: str) -> Tuple[float, str]:
    if unit == "s":
        return 1.0, "Time (s)"
    if unit == "min":
        return 60.0, "Time (min)"
    if unit == "h":
        return 3600.0, "Time (h)"
    raise ValueError(unit)


def main() -> None:
    args = parse_args()

    if args.backend != "auto":
        import matplotlib
        matplotlib.use(args.backend)
    else:
        import matplotlib
        if not args.no_show:
            for cand in ("QtAgg", "TkAgg"):
                try:
                    matplotlib.use(cand)
                    break
                except Exception:
                    continue
        else:
            matplotlib.use("Agg")

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    repo_root = os.path.abspath(args.repo_root)
    model_dir = os.path.abspath(args.model_dir)
    solution_path = os.path.abspath(args.solution_path)
    profile_path = os.path.abspath(args.profile_path)
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = os.path.join(model_dir, f"cycle_{args.cycle}_plots")
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    ckpt = resolve_checkpoint(model_dir, args.weight_name)
    nn = build_model(
        repo_root=repo_root,
        model_dir=model_dir,
        ckpt_path=ckpt,
        profile_path=profile_path,
        profile_encoding=args.profile_encoding,
        profile_cycle_start=args.profile_cycle_start,
        profile_cycle_end=args.profile_cycle_end,
    )

    solution = np.load(solution_path)
    required = ["t", "r_a", "r_c", "phie", "phis_c", "cs_a", "cs_c"]
    for key in required:
        if key not in solution.files:
            raise KeyError(f"Missing '{key}' in {solution_path}. Found: {solution.files}")

    t_all = np.asarray(solution["t"], dtype=np.float64).reshape(-1)
    phie_all = np.asarray(solution["phie"], dtype=np.float64).reshape(-1)
    phis_c_all = np.asarray(solution["phis_c"], dtype=np.float64).reshape(-1)
    r_a = np.asarray(solution["r_a"], dtype=np.float64).reshape(-1)
    r_c = np.asarray(solution["r_c"], dtype=np.float64).reshape(-1)
    cs_a_all = _ensure_time_major(solution["cs_a"], len(t_all), len(r_a), "cs_a")
    cs_c_all = _ensure_time_major(solution["cs_c"], len(t_all), len(r_c), "cs_c")

    cycle_mask, cycle_local_time_s = _load_cycle_mask_and_local_time(
        repo_root=repo_root,
        profile_path=profile_path,
        encoding=args.profile_encoding,
        cycle=args.cycle,
    )
    if cycle_mask.shape[0] != t_all.shape[0]:
        raise ValueError(
            f"Profile rows ({cycle_mask.shape[0]}) and solution time length ({t_all.shape[0]}) do not match."
        )

    t_cycle_abs = t_all[cycle_mask]
    phie_truth = phie_all[cycle_mask]
    phis_c_truth = phis_c_all[cycle_mask]
    cs_a_truth = cs_a_all[cycle_mask, :]
    cs_c_truth = cs_c_all[cycle_mask, :]

    deg_i0_ref = float(nn.params.get("deg_i0_a_ref", 0.5))
    deg_ds_ref = float(nn.params.get("deg_ds_c_ref", 1.0))

    x_params_line = np.column_stack([
        np.full_like(t_cycle_abs, deg_i0_ref, dtype=np.float64),
        np.full_like(t_cycle_abs, deg_ds_ref, dtype=np.float64),
    ])
    phie_pred = predict_target(nn, "phie", t_cycle_abs.reshape(-1, 1), x_params_line, args.batch_size).reshape(-1)
    phis_c_pred = predict_target(nn, "phis_c", t_cycle_abs.reshape(-1, 1), x_params_line, args.batch_size).reshape(-1)

    n_t = len(t_cycle_abs)
    x_cs_a = np.column_stack([
        np.repeat(t_cycle_abs, len(r_a)),
        np.tile(r_a, n_t),
    ])
    x_params_cs_a = np.column_stack([
        np.full(x_cs_a.shape[0], deg_i0_ref, dtype=np.float64),
        np.full(x_cs_a.shape[0], deg_ds_ref, dtype=np.float64),
    ])
    cs_a_pred = predict_target(nn, "cs_a", x_cs_a, x_params_cs_a, args.batch_size).reshape(n_t, len(r_a))

    x_cs_c = np.column_stack([
        np.repeat(t_cycle_abs, len(r_c)),
        np.tile(r_c, n_t),
    ])
    x_params_cs_c = np.column_stack([
        np.full(x_cs_c.shape[0], deg_i0_ref, dtype=np.float64),
        np.full(x_cs_c.shape[0], deg_ds_ref, dtype=np.float64),
    ])
    cs_c_pred = predict_target(nn, "cs_c", x_cs_c, x_params_cs_c, args.batch_size).reshape(n_t, len(r_c))

    time_div, time_label = _time_scale_and_label(args.time_unit)
    time_plot = cycle_local_time_s / time_div

    phie_metrics = _metric_summary(phie_truth, phie_pred)
    phis_c_metrics = _metric_summary(phis_c_truth, phis_c_pred)
    cs_a_metrics = _metric_summary(cs_a_truth, cs_a_pred)
    cs_c_metrics = _metric_summary(cs_c_truth, cs_c_pred)

    # Line figure for phie / phis_c
    fig1, axes = plt.subplots(2, 1, figsize=(10.5, 7.8), sharex=True)
    axes[0].plot(time_plot, phie_truth, label="Truth")
    axes[0].plot(time_plot, phie_pred, linestyle="--", label="Prediction")
    axes[0].set_ylabel("phie")
    axes[0].set_title(
        f"Cycle {args.cycle} | phie | NMAE={phie_metrics['nmae']*100:.2f}%  "
        f"NRMSE={phie_metrics['nrmse']*100:.2f}%  R2={phie_metrics['r2']:.4f}"
    )
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(time_plot, phis_c_truth, label="Truth")
    axes[1].plot(time_plot, phis_c_pred, linestyle="--", label="Prediction")
    axes[1].set_ylabel("phis_c")
    axes[1].set_xlabel(time_label)
    axes[1].set_title(
        f"Cycle {args.cycle} | phis_c | NMAE={phis_c_metrics['nmae']*100:.2f}%  "
        f"NRMSE={phis_c_metrics['nrmse']*100:.2f}%  R2={phis_c_metrics['r2']:.4f}"
    )
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    fig1.tight_layout()
    fig1_path = os.path.join(output_dir, f"cycle_{args.cycle}_phie_phis_c_truth_vs_pred.png")
    fig1.savefig(fig1_path, dpi=220, bbox_inches="tight")

    # Prepare surface grids
    step = max(int(args.surface_time_step), 1)
    idx = np.arange(0, n_t, step)
    time_surf = time_plot[idx]
    cs_a_truth_s = cs_a_truth[idx, :]
    cs_a_pred_s = cs_a_pred[idx, :]
    cs_c_truth_s = cs_c_truth[idx, :]
    cs_c_pred_s = cs_c_pred[idx, :]

    TA, RA = np.meshgrid(time_surf, r_a, indexing="ij")
    TC, RC = np.meshgrid(time_surf, r_c, indexing="ij")

    # cs_a surfaces
    fig2 = plt.figure(figsize=(14, 5.8))
    ax21 = fig2.add_subplot(1, 2, 1, projection="3d")
    surf21 = ax21.plot_surface(TA, RA, cs_a_truth_s, cmap="viridis", linewidth=0, antialiased=True)
    ax21.set_title(f"Cycle {args.cycle} | cs_a Truth")
    ax21.set_xlabel(time_label)
    ax21.set_ylabel("r_a")
    ax21.set_zlabel("cs_a")
    fig2.colorbar(surf21, ax=ax21, shrink=0.7, pad=0.08)

    ax22 = fig2.add_subplot(1, 2, 2, projection="3d")
    surf22 = ax22.plot_surface(TA, RA, cs_a_pred_s, cmap="viridis", linewidth=0, antialiased=True)
    ax22.set_title(
        f"Cycle {args.cycle} | cs_a Prediction\n"
        f"NMAE={cs_a_metrics['nmae']*100:.2f}%  NRMSE={cs_a_metrics['nrmse']*100:.2f}%  R2={cs_a_metrics['r2']:.4f}"
    )
    ax22.set_xlabel(time_label)
    ax22.set_ylabel("r_a")
    ax22.set_zlabel("cs_a")
    fig2.colorbar(surf22, ax=ax22, shrink=0.7, pad=0.08)
    fig2.tight_layout()
    fig2_path = os.path.join(output_dir, f"cycle_{args.cycle}_cs_a_truth_vs_pred_surface.png")
    fig2.savefig(fig2_path, dpi=220, bbox_inches="tight")

    # cs_c surfaces
    fig3 = plt.figure(figsize=(14, 5.8))
    ax31 = fig3.add_subplot(1, 2, 1, projection="3d")
    surf31 = ax31.plot_surface(TC, RC, cs_c_truth_s, cmap="viridis", linewidth=0, antialiased=True)
    ax31.set_title(f"Cycle {args.cycle} | cs_c Truth")
    ax31.set_xlabel(time_label)
    ax31.set_ylabel("r_c")
    ax31.set_zlabel("cs_c")
    fig3.colorbar(surf31, ax=ax31, shrink=0.7, pad=0.08)

    ax32 = fig3.add_subplot(1, 2, 2, projection="3d")
    surf32 = ax32.plot_surface(TC, RC, cs_c_pred_s, cmap="viridis", linewidth=0, antialiased=True)
    ax32.set_title(
        f"Cycle {args.cycle} | cs_c Prediction\n"
        f"NMAE={cs_c_metrics['nmae']*100:.2f}%  NRMSE={cs_c_metrics['nrmse']*100:.2f}%  R2={cs_c_metrics['r2']:.4f}"
    )
    ax32.set_xlabel(time_label)
    ax32.set_ylabel("r_c")
    ax32.set_zlabel("cs_c")
    fig3.colorbar(surf32, ax=ax32, shrink=0.7, pad=0.08)
    fig3.tight_layout()
    fig3_path = os.path.join(output_dir, f"cycle_{args.cycle}_cs_c_truth_vs_pred_surface.png")
    fig3.savefig(fig3_path, dpi=220, bbox_inches="tight")

    np.savez_compressed(
        os.path.join(output_dir, f"cycle_{args.cycle}_truth_pred_arrays.npz"),
        cycle=args.cycle,
        time_abs_s=t_cycle_abs,
        time_local_s=cycle_local_time_s,
        r_a=r_a,
        r_c=r_c,
        phie_truth=phie_truth,
        phie_pred=phie_pred,
        phis_c_truth=phis_c_truth,
        phis_c_pred=phis_c_pred,
        cs_a_truth=cs_a_truth,
        cs_a_pred=cs_a_pred,
        cs_c_truth=cs_c_truth,
        cs_c_pred=cs_c_pred,
    )

    print("Saved:")
    print(fig1_path)
    print(fig2_path)
    print(fig3_path)
    print(os.path.join(output_dir, f"cycle_{args.cycle}_truth_pred_arrays.npz"))

    if args.no_show:
        plt.close("all")
    else:
        plt.show()


if __name__ == "__main__":
    main()
