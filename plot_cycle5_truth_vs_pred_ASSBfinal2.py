from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot cycle-specific truth vs prediction for ASSBfinal2")
    parser.add_argument("--repo-root", type=str, default=None)
    parser.add_argument("--model-dir", type=str, required=True)
    parser.add_argument("--solution-path", type=str, required=True)
    parser.add_argument("--profile-path", type=str, required=True)
    parser.add_argument("--cycle", type=int, default=5)
    parser.add_argument("--encoding", type=str, default="auto")
    parser.add_argument("--cycle-start", type=int, default=None)
    parser.add_argument("--cycle-end", type=int, default=None)
    parser.add_argument("--no-show", action="store_true")
    return parser.parse_args()


def n_metrics(pred: np.ndarray, true: np.ndarray) -> tuple[float, float, float]:
    err = np.asarray(pred, dtype=float) - np.asarray(true, dtype=float)
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))
    denom = float(np.max(true) - np.min(true))
    denom = denom if abs(denom) > 1e-15 else 1.0
    ss_res = float(np.sum(err**2))
    ss_tot = float(np.sum((true - np.mean(true)) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    return mae / denom * 100.0, rmse / denom * 100.0, r2


def main() -> None:
    args = parse_args()
    repo_root = Path(args.repo_root) if args.repo_root is not None else Path(__file__).resolve().parent
    sys.path.insert(0, str(repo_root / "pinn_spm_param"))
    sys.path.insert(0, str(repo_root / "pinn_spm_param" / "util"))

    from current_profile_ASSBfinal2 import load_current_profile_ASSBfinal2
    from myNN_ASSBfinal2 import StepLatentFieldModel_ASSBfinal2
    from spm_ASSBfinal2 import make_context_ASSBfinal2
    from _losses_ASSBfinal2 import derive_potentials_from_concentrations_ASSBfinal2, predict_all_concentrations_ASSBfinal2

    model_dir = Path(args.model_dir)
    ckpt = torch.load(model_dir / "best.pt", map_location="cpu")
    cfg = ckpt["config"]
    context = make_context_ASSBfinal2(
        repo_root=str(repo_root),
        profile_path=args.profile_path,
        profile_encoding=args.encoding,
        cycle_start=args.cycle_start,
        cycle_end=args.cycle_end,
        n_r=int(cfg["N_R"]),
        device=str(cfg["DEVICE"]),
        dtype=str(cfg["DTYPE"]),
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
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    cs_a_pred, cs_c_pred = predict_all_concentrations_ASSBfinal2(model, context, time_chunk=int(cfg.get("TIME_CHUNK_EVAL", 2048)))
    phie_pred, phis_pred = derive_potentials_from_concentrations_ASSBfinal2(cs_a_pred, cs_c_pred, context)

    sol = np.load(args.solution_path)
    cs_a_true = np.asarray(sol["cs_a"], dtype=float)
    cs_c_true = np.asarray(sol["cs_c"], dtype=float)
    phie_true = np.asarray(sol["phie"], dtype=float)
    phis_true = np.asarray(sol["phis_c"], dtype=float)
    t_s = np.asarray(sol["t"], dtype=float)
    r_a = np.asarray(sol["r_a"], dtype=float)
    r_c = np.asarray(sol["r_c"], dtype=float)

    profile = load_current_profile_ASSBfinal2(
        profile_path=args.profile_path,
        encoding=args.encoding,
        cycle_start=args.cycle_start,
        cycle_end=args.cycle_end,
    )
    cycle_vec = np.asarray(profile["cycle"], dtype=int)
    mask = cycle_vec == int(args.cycle)
    if not np.any(mask):
        raise ValueError(f"Cycle {args.cycle} not found in the filtered profile.")

    t_cycle_min = t_s[mask] / 60.0
    phie_true_c = phie_true[mask]
    phie_pred_c = phie_pred[mask]
    phis_true_c = phis_true[mask]
    phis_pred_c = phis_pred[mask]
    cs_a_true_c = cs_a_true[mask, :]
    cs_a_pred_c = cs_a_pred[mask, :]
    cs_c_true_c = cs_c_true[mask, :]
    cs_c_pred_c = cs_c_pred[mask, :]

    out_dir = model_dir / f"cycle_{args.cycle}_plots_ASSBfinal2"
    out_dir.mkdir(parents=True, exist_ok=True)

    nmae, nrmse, r2 = n_metrics(phie_pred_c, phie_true_c)
    nmae2, nrmse2, r22 = n_metrics(phis_pred_c, phis_true_c)
    fig1, ax = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    ax[0].plot(t_cycle_min, phie_true_c, label="Truth")
    ax[0].plot(t_cycle_min, phie_pred_c, linestyle="--", label="Prediction")
    ax[0].set_title(f"Cycle {args.cycle} | phie | NMAE={nmae:.2f}% NRMSE={nrmse:.2f}% R2={r2:.4f}")
    ax[0].set_ylabel("phie")
    ax[0].grid(True, alpha=0.3)
    ax[0].legend()
    ax[1].plot(t_cycle_min, phis_true_c, label="Truth")
    ax[1].plot(t_cycle_min, phis_pred_c, linestyle="--", label="Prediction")
    ax[1].set_title(f"Cycle {args.cycle} | phis_c | NMAE={nmae2:.2f}% NRMSE={nrmse2:.2f}% R2={r22:.4f}")
    ax[1].set_xlabel("Time (min)")
    ax[1].set_ylabel("phis_c")
    ax[1].grid(True, alpha=0.3)
    ax[1].legend()
    fig1.tight_layout()
    fig1.savefig(out_dir / f"cycle_{args.cycle}_phie_phis_c_truth_vs_pred.png", dpi=180)

    def _plot_surface_pair(title_prefix: str, time_min: np.ndarray, r_grid: np.ndarray, truth: np.ndarray, pred: np.ndarray, filename: str):
        Tm, Rm = np.meshgrid(time_min, r_grid, indexing="ij")
        nmae_s, nrmse_s, r2_s = n_metrics(pred, truth)
        fig = plt.figure(figsize=(15, 6))
        ax1 = fig.add_subplot(1, 2, 1, projection="3d")
        ax2 = fig.add_subplot(1, 2, 2, projection="3d")
        surf1 = ax1.plot_surface(Tm, Rm, truth, cmap="viridis", linewidth=0, antialiased=True)
        ax1.set_title(f"Cycle {args.cycle} | {title_prefix} Truth")
        ax1.set_xlabel("Time (min)")
        ax1.set_ylabel(f"r_{title_prefix[-1]}")
        ax1.set_zlabel(title_prefix)
        fig.colorbar(surf1, ax=ax1, shrink=0.65)
        surf2 = ax2.plot_surface(Tm, Rm, pred, cmap="viridis", linewidth=0, antialiased=True)
        ax2.set_title(f"Cycle {args.cycle} | {title_prefix} Prediction\nNMAE={nmae_s:.2f}% NRMSE={nrmse_s:.2f}% R2={r2_s:.4f}")
        ax2.set_xlabel("Time (min)")
        ax2.set_ylabel(f"r_{title_prefix[-1]}")
        ax2.set_zlabel(title_prefix)
        fig.colorbar(surf2, ax=ax2, shrink=0.65)
        fig.tight_layout()
        fig.savefig(out_dir / filename, dpi=180)
        return fig

    fig2 = _plot_surface_pair("cs_a", t_cycle_min, r_a, cs_a_true_c, cs_a_pred_c, f"cycle_{args.cycle}_cs_a_truth_vs_pred_surface.png")
    fig3 = _plot_surface_pair("cs_c", t_cycle_min, r_c, cs_c_true_c, cs_c_pred_c, f"cycle_{args.cycle}_cs_c_truth_vs_pred_surface.png")

    np.savez_compressed(
        out_dir / f"cycle_{args.cycle}_truth_pred_arrays.npz",
        t_min=t_cycle_min,
        r_a=r_a,
        r_c=r_c,
        phie_true=phie_true_c,
        phie_pred=phie_pred_c,
        phis_c_true=phis_true_c,
        phis_c_pred=phis_pred_c,
        cs_a_true=cs_a_true_c,
        cs_a_pred=cs_a_pred_c,
        cs_c_true=cs_c_true_c,
        cs_c_pred=cs_c_pred_c,
    )
    print(f"Saved plots to: {out_dir}")

    if not args.no_show:
        plt.show()
    else:
        plt.close(fig1)
        plt.close(fig2)
        plt.close(fig3)


if __name__ == "__main__":
    main()
