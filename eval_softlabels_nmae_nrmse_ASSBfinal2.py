from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate ASSBfinal2 physics-only model against soft labels")
    parser.add_argument("--repo-root", type=str, default=None, help="Repository root")
    parser.add_argument("--model-dir", type=str, required=True, help="ModelFin_<ID> folder")
    parser.add_argument("--solution-path", type=str, default=None, help="Path to solution.npz")
    parser.add_argument("--data-dir", type=str, default=None, help="Folder containing solution.npz")
    parser.add_argument("--profile-path", type=str, required=True, help="CSV or ZIP current profile")
    parser.add_argument("--encoding", type=str, default="auto")
    parser.add_argument("--cycle-start", type=int, default=None)
    parser.add_argument("--cycle-end", type=int, default=None)
    parser.add_argument("--norm", choices=["range", "std", "meanabs"], default="range")
    parser.add_argument("--output-prefix", type=str, default="eval_ASSBfinal2")
    return parser.parse_args()


def resolve_solution_path(solution_path: str | None, data_dir: str | None) -> Path:
    if solution_path:
        return Path(solution_path)
    if data_dir:
        return Path(data_dir) / "solution.npz"
    raise ValueError("Provide --solution-path or --data-dir.")


def n_metrics(pred: np.ndarray, true: np.ndarray, norm: str) -> dict[str, float]:
    pred = np.asarray(pred, dtype=float)
    true = np.asarray(true, dtype=float)
    err = pred - true
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))
    ss_res = float(np.sum(err**2))
    ss_tot = float(np.sum((true - np.mean(true)) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    if norm == "range":
        denom = float(np.max(true) - np.min(true))
    elif norm == "std":
        denom = float(np.std(true))
    else:
        denom = float(np.mean(np.abs(true)))
    denom = denom if abs(denom) > 1e-15 else 1.0
    return {
        "NMAE": mae / denom * 100.0,
        "NRMSE": rmse / denom * 100.0,
        "R2": r2,
        "MAE": mae,
        "RMSE": rmse,
    }


def main() -> None:
    args = parse_args()
    repo_root = Path(args.repo_root) if args.repo_root is not None else Path(__file__).resolve().parent
    sys.path.insert(0, str(repo_root / "pinn_spm_param"))
    sys.path.insert(0, str(repo_root / "pinn_spm_param" / "util"))

    from init_pinn_ASSBfinal2 import parse_input_file_ASSBfinal2
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

    sol = np.load(resolve_solution_path(args.solution_path, args.data_dir))
    phie_true = np.asarray(sol["phie"], dtype=float)
    phis_true = np.asarray(sol["phis_c"], dtype=float)
    cs_a_true = np.asarray(sol["cs_a"], dtype=float)
    cs_c_true = np.asarray(sol["cs_c"], dtype=float)

    if phie_true.shape[0] != phie_pred.shape[0]:
        raise ValueError("Prediction length does not match solution.npz length.")

    results = {
        "phie": n_metrics(phie_pred, phie_true, args.norm),
        "phis_c": n_metrics(phis_pred, phis_true, args.norm),
        "cs_a": n_metrics(cs_a_pred, cs_a_true, args.norm),
        "cs_c": n_metrics(cs_c_pred, cs_c_true, args.norm),
    }

    print("=" * 72)
    for name in ["phie", "phis_c", "cs_a", "cs_c"]:
        m = results[name]
        print(
            f"{name:7s} NMAE={m['NMAE']:.4f}% NRMSE={m['NRMSE']:.4f}% "
            f"R2={m['R2']:.6f} MAE={m['MAE']:.6e} RMSE={m['RMSE']:.6e}"
        )
    print("=" * 72)

    out_json = model_dir / f"{args.output_prefix}_metrics.json"
    out_csv = model_dir / f"{args.output_prefix}_metrics.csv"
    out_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    with out_csv.open("w", encoding="utf-8") as f:
        f.write("name,NMAE,NRMSE,R2,MAE,RMSE\n")
        for name, m in results.items():
            f.write(f"{name},{m['NMAE']},{m['NRMSE']},{m['R2']},{m['MAE']},{m['RMSE']}\n")
    print(f"Saved CSV : {out_csv}")
    print(f"Saved JSON: {out_json}")


if __name__ == "__main__":
    main()
