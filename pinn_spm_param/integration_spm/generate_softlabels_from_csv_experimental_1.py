from __future__ import annotations

import argparse
import json
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


ENCODING_CANDIDATES = ["utf-8", "utf-8-sig", "gbk", "gb18030", "latin1"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Use the repository's experimental current-driven SPM solver to generate "
            "soft labels (phie, phis_c, cs_a, cs_c) from a CSV/ZIP current record."
        )
    )
    parser.add_argument("--input", type=str, required=True, help="Path to input CSV or ZIP containing one CSV.")
    parser.add_argument(
        "--repo-root",
        type=str,
        default=None,
        help=(
            "Repository root. Example: C:/Users/Tiga_QJW/Desktop/myPINN-V1/myPINN-ASSB-V1. "
            "If omitted, the script will try a few relative locations automatically."
        ),
    )
    parser.add_argument("--encoding", type=str, default="auto", help="CSV encoding. Default: auto detect.")
    parser.add_argument("--cycle-start", type=int, default=None, help="Inclusive cycle start. Default: CSV minimum cycle.")
    parser.add_argument("--cycle-end", type=int, default=None, help="Inclusive cycle end. Default: CSV maximum cycle.")
    parser.add_argument("--max-rows", type=int, default=None, help="Optional debug shortcut. Only keep the first N filtered rows.")
    parser.add_argument("--output-dir", type=str, default=None, help="Default: sibling folder named softlabels_nr64 next to the input file.")
    parser.add_argument("--output-prefix", type=str, default=None, help="Output file prefix. Default: input stem + cycle range.")
    parser.add_argument("--n-r", type=int, default=64, help="Radial grid count for implicit integration. Default: 64.")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"], help="Torch device.")
    parser.add_argument("--deg-i0-a", type=float, default=None, help="Override built-in default if desired.")
    parser.add_argument("--deg-ds-c", type=float, default=None, help="Override built-in default if desired.")
    parser.add_argument("--x-a0", type=float, default=None, help="Override built-in default if desired.")
    parser.add_argument("--x-c0", type=float, default=None, help="Override built-in default if desired.")
    parser.add_argument(
        "--diag-prefix-rows",
        type=int,
        default=10000,
        help=(
            "Rows used for measured-vs-model voltage diagnosis. The script compares both n_r=12 and requested n_r "
            "on this prefix because a full-file n_r=64 comparison can be very time consuming. Default: 10000."
        ),
    )
    parser.add_argument("--skip-voltage-diagnostic", action="store_true", help="Skip diagnostic comparison against measured voltage.")
    parser.add_argument(
        "--write-solution-alias",
        action="store_true",
        help="Also save an extra file literally named solution.npz in the output folder for downstream repo preprocessing.",
    )
    return parser.parse_args()



def _normalize_col_name(name: object) -> str:
    text = str(name).strip().lower()
    for token in [" ", "（", "）", "(", ")", "_", "-", "/", "\\", "."]:
        text = text.replace(token, "")
    return text



def _first_existing(df: pd.DataFrame, candidates: Iterable[str]) -> str:
    normalized = {_normalize_col_name(col): col for col in df.columns}
    for candidate in candidates:
        key = _normalize_col_name(candidate)
        if key in normalized:
            return normalized[key]
    raise KeyError(f"Missing required column. Tried: {list(candidates)}")



def detect_columns(df: pd.DataFrame) -> dict[str, str]:
    return {
        "data_index": _first_existing(df, ["数据序号", "序号", "dataindex"]),
        "cycle": _first_existing(df, ["循环号", "循环", "cycle", "循环序号"]),
        "step": _first_existing(df, ["工步号", "步骤号", "step", "工步"]),
        "step_type": _first_existing(df, ["公布类型", "工步类型", "步骤类型", "工序类型", "steptype"]),
        "time": _first_existing(df, ["时间", "单步时间", "步时间", "time"]),
        "total_time": _first_existing(df, ["总时间", "累计时间", "累积时间", "totaltime"]),
        "current": _first_existing(df, ["电流(A)", "电流", "current(a)", "current"]),
        "voltage": _first_existing(df, ["电压(V)", "电压", "voltage(v)", "voltage"]),
        "absolute_time": _first_existing(df, ["绝对时间", "日期时间", "测试时间", "datetime", "timestamp"]),
    }



def _format_duration_seconds(seconds: float) -> str:
    total = int(round(float(seconds)))
    sign = "-" if total < 0 else ""
    total = abs(total)
    days, rem = divmod(total, 86400)
    hours, rem = divmod(rem, 3600)
    minutes, secs = divmod(rem, 60)
    if days > 0:
        return f"{sign}{days} days {hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{sign}{hours:02d}:{minutes:02d}:{secs:02d}"



def format_time_like_series_for_output(series: pd.Series, mode: str) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(series):
        return pd.to_datetime(series).dt.strftime("%Y-%m-%d %H:%M:%S")
    if pd.api.types.is_timedelta64_dtype(series):
        return series.dt.total_seconds().map(_format_duration_seconds)

    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().sum() >= max(3, int(0.8 * len(series))):
        values = numeric.ffill().bfill().to_numpy(dtype=float)
        if mode == "duration":
            seconds = values * 86400.0 if np.nanmax(values) <= 3650 else values
            return pd.Series([_format_duration_seconds(v) for v in seconds], index=series.index)
        if mode == "absolute" and np.nanmin(values) > 20000:
            dt_vals = pd.to_datetime(values, unit="D", origin="1899-12-30", errors="coerce")
            return pd.Series(dt_vals.strftime("%Y-%m-%d %H:%M:%S"), index=series.index)

    td = pd.to_timedelta(series, errors="coerce")
    if td.notna().sum() >= max(3, int(0.8 * len(series))):
        filled = td.ffill().bfill()
        return filled.dt.total_seconds().map(_format_duration_seconds)

    dt = pd.to_datetime(series, errors="coerce")
    if dt.notna().sum() >= max(3, int(0.8 * len(series))):
        filled = dt.ffill().bfill()
        return filled.dt.strftime("%Y-%m-%d %H:%M:%S")

    return series.astype(str)



def series_to_seconds(series: pd.Series) -> np.ndarray:
    if pd.api.types.is_timedelta64_dtype(series):
        return series.dt.total_seconds().to_numpy(dtype=float)
    if pd.api.types.is_datetime64_any_dtype(series):
        return (series - series.iloc[0]).dt.total_seconds().to_numpy(dtype=float)

    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().sum() >= max(3, int(0.8 * len(series))):
        values = numeric.ffill().bfill().to_numpy(dtype=float)
        if np.nanmax(values) <= 3650:
            values = values * 86400.0
        return values - values[0]

    td = pd.to_timedelta(series, errors="coerce")
    if td.notna().sum() >= max(3, int(0.8 * len(series))):
        filled = td.ffill().bfill()
        return filled.dt.total_seconds().to_numpy(dtype=float)

    dt = pd.to_datetime(series, errors="coerce")
    if dt.notna().sum() >= max(3, int(0.8 * len(series))):
        filled = dt.ffill().bfill()
        return (filled - filled.iloc[0]).dt.total_seconds().to_numpy(dtype=float)

    raise ValueError("Unable to convert the selected time column to seconds.")



def resolve_csv_from_input(input_path: Path) -> tuple[Path, tempfile.TemporaryDirectory | None]:
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    if input_path.suffix.lower() == ".csv":
        return input_path, None
    if input_path.suffix.lower() != ".zip":
        raise ValueError("--input must point to a CSV or a ZIP containing one CSV.")

    tmpdir = tempfile.TemporaryDirectory(prefix="spm_csv_")
    with zipfile.ZipFile(input_path, "r") as zf:
        csv_members = [m for m in zf.namelist() if m.lower().endswith(".csv")]
        if len(csv_members) != 1:
            raise ValueError(f"ZIP must contain exactly one CSV. Found: {csv_members}")
        zf.extract(csv_members[0], path=tmpdir.name)
        return Path(tmpdir.name) / csv_members[0], tmpdir



def read_csv_auto(csv_path: Path, encoding: str) -> tuple[pd.DataFrame, str]:
    attempted: list[str] = []
    candidates = ENCODING_CANDIDATES if encoding == "auto" else [encoding]
    last_error: Exception | None = None
    for enc in candidates:
        try:
            return pd.read_csv(csv_path, encoding=enc), enc
        except Exception as exc:  # noqa: BLE001
            attempted.append(enc)
            last_error = exc
    raise RuntimeError(f"Failed to read CSV with encodings {attempted}") from last_error



def configure_repo_imports(repo_root: str | None) -> None:
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
            current_dir.parent / "util",
            current_dir / "util",
            Path.cwd() / "pinn_spm_param" / "integration_spm",
            Path.cwd() / "pinn_spm_param" / "util",
            Path.cwd() / "integration_spm",
            Path.cwd() / "util",
        ]
    )
    added = False
    for c in candidates:
        if c.exists() and c.is_dir() and str(c) not in sys.path:
            sys.path.insert(0, str(c))
            added = True
    if not added:
        raise FileNotFoundError(
            "Could not locate the repository integration_spm/util directories. Please pass --repo-root pointing to the myPINN-ASSB-V1 repository root."
        )



def calc_error_stats(pred: np.ndarray, true: np.ndarray) -> dict[str, float]:
    err = np.asarray(pred, dtype=float) - np.asarray(true, dtype=float)
    return {
        "rmse": float(np.sqrt(np.mean(err**2))),
        "mae": float(np.mean(np.abs(err))),
        "max_abs": float(np.max(np.abs(err))),
        "mean_signed": float(np.mean(err)),
    }



def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    csv_path, tmpdir = resolve_csv_from_input(input_path)
    try:
        configure_repo_imports(args.repo_root)
        from spm_experimental_1 import make_params_experimental_1
        from spm_int_experimental_1 import simulate_current_profile_experimental_1

        df_raw, encoding_used = read_csv_auto(csv_path, args.encoding)
        colmap = detect_columns(df_raw)

        cycle_series = pd.to_numeric(df_raw[colmap["cycle"]], errors="coerce")
        cycle_start = int(cycle_series.min()) if args.cycle_start is None else int(args.cycle_start)
        cycle_end = int(cycle_series.max()) if args.cycle_end is None else int(args.cycle_end)
        mask = (cycle_series >= cycle_start) & (cycle_series <= cycle_end)
        df = df_raw.loc[mask].copy()
        if args.max_rows is not None:
            df = df.iloc[: int(args.max_rows)].copy()
        if df.empty:
            raise ValueError("The selected cycle range is empty.")

        total_seconds = series_to_seconds(df[colmap["total_time"]])
        total_seconds = total_seconds - float(total_seconds[0])
        current_a = pd.to_numeric(df[colmap["current"]], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        voltage_meas = pd.to_numeric(df[colmap["voltage"]], errors="coerce").to_numpy(dtype=float)

        resolved_device = None if args.device == "auto" else args.device
        params = make_params_experimental_1(device=resolved_device)
        sim = simulate_current_profile_experimental_1(
            time_s=total_seconds,
            current_a=current_a,
            params=params,
            deg_i0_a=args.deg_i0_a,
            deg_ds_c=args.deg_ds_c,
            x_a0=args.x_a0,
            x_c0=args.x_c0,
            n_r=int(args.n_r),
        )

        output_dir = Path(args.output_dir) if args.output_dir else input_path.resolve().parent / f"softlabels_nr{int(args.n_r)}"
        output_dir.mkdir(parents=True, exist_ok=True)
        stem = input_path.stem if input_path.suffix.lower() == ".csv" else csv_path.stem
        prefix = args.output_prefix or f"{stem}_cycle{cycle_start}_{cycle_end}"

        # Convenience 1D exports
        time_out = format_time_like_series_for_output(df[colmap["time"]], mode="duration")
        total_time_out = format_time_like_series_for_output(df[colmap["total_time"]], mode="duration")
        absolute_time_out = format_time_like_series_for_output(df[colmap["absolute_time"]], mode="absolute")
        one_d_df = pd.DataFrame(
            {
                colmap["data_index"]: df[colmap["data_index"]].to_numpy(),
                colmap["cycle"]: df[colmap["cycle"]].to_numpy(),
                colmap["step"]: df[colmap["step"]].to_numpy(),
                colmap["step_type"]: df[colmap["step_type"]].to_numpy(),
                colmap["time"]: time_out.to_numpy(),
                colmap["total_time"]: total_time_out.to_numpy(),
                colmap["current"]: df[colmap["current"]].to_numpy(),
                "测量电压(V)": voltage_meas,
                "SPM电压(V)": sim["voltage"],
                "SPM电压误差(V)": sim["voltage"] - voltage_meas,
                "phie": sim["phie"],
                "phis_c": sim["phis_c"],
                colmap["absolute_time"]: absolute_time_out.to_numpy(),
            }
        )
        one_d_csv = output_dir / f"{prefix}_phie_phis_c_voltage_nr{int(args.n_r)}.csv"
        one_d_df.to_csv(one_d_csv, index=False, encoding="utf-8-sig")

        # Raw state arrays for all four soft labels.
        states_npz = output_dir / f"{prefix}_states_nr{int(args.n_r)}.npz"
        np.savez(
            states_npz,
            t=sim["t"],
            r_a=sim["r_a"],
            r_c=sim["r_c"],
            cs_a=sim["cs_a"],
            cs_c=sim["cs_c"],
            phie=sim["phie"],
            phis_c=sim["phis_c"],
            voltage=sim["voltage"],
            current=current_a,
            measured_voltage=voltage_meas,
            cycle=pd.to_numeric(df[colmap["cycle"]], errors="coerce").to_numpy(),
            step=pd.to_numeric(df[colmap["step"]], errors="coerce").to_numpy(),
            data_index=pd.to_numeric(df[colmap["data_index"]], errors="coerce").to_numpy(),
        )

        # Save separate field files for convenience.
        np.savez(output_dir / f"{prefix}_cs_a_nr{int(args.n_r)}.npz", t=sim["t"], r_a=sim["r_a"], cs_a=sim["cs_a"])
        np.savez(output_dir / f"{prefix}_cs_c_nr{int(args.n_r)}.npz", t=sim["t"], r_c=sim["r_c"], cs_c=sim["cs_c"])
        pd.DataFrame({"t_s": sim["t"], "phie": sim["phie"]}).to_csv(
            output_dir / f"{prefix}_phie_nr{int(args.n_r)}.csv", index=False, encoding="utf-8-sig"
        )
        pd.DataFrame({"t_s": sim["t"], "phis_c": sim["phis_c"]}).to_csv(
            output_dir / f"{prefix}_phis_c_nr{int(args.n_r)}.csv", index=False, encoding="utf-8-sig"
        )

        if args.write_solution_alias:
            alias_path = output_dir / "solution.npz"
            np.savez(
                alias_path,
                t=sim["t"],
                r_a=sim["r_a"],
                r_c=sim["r_c"],
                cs_a=sim["cs_a"],
                cs_c=sim["cs_c"],
                phie=sim["phie"],
                phis_c=sim["phis_c"],
                voltage=sim["voltage"],
            )
        else:
            alias_path = None

        diagnostic: dict[str, object] = {"skipped": bool(args.skip_voltage_diagnostic)}
        if not args.skip_voltage_diagnostic:
            diag_n = min(int(args.diag_prefix_rows), len(df))
            prefix_t = total_seconds[:diag_n]
            prefix_i = current_a[:diag_n]
            prefix_v = voltage_meas[:diag_n]
            sim_nr12 = simulate_current_profile_experimental_1(
                time_s=prefix_t,
                current_a=prefix_i,
                params=params,
                deg_i0_a=args.deg_i0_a,
                deg_ds_c=args.deg_ds_c,
                x_a0=args.x_a0,
                x_c0=args.x_c0,
                n_r=12,
            )
            sim_requested = simulate_current_profile_experimental_1(
                time_s=prefix_t,
                current_a=prefix_i,
                params=params,
                deg_i0_a=args.deg_i0_a,
                deg_ds_c=args.deg_ds_c,
                x_a0=args.x_a0,
                x_c0=args.x_c0,
                n_r=int(args.n_r),
            )
            diagnostic = {
                "skipped": False,
                "prefix_rows": diag_n,
                "stats_n_r_12": calc_error_stats(sim_nr12["voltage"], prefix_v),
                f"stats_n_r_{int(args.n_r)}": calc_error_stats(sim_requested["voltage"], prefix_v),
                "interpretation": (
                    "If n_r=12 is nearly exact while the requested n_r is not, then the measured voltage column was almost certainly generated by the same built-in solver with n_r=12."
                ),
            }
            diag_df = pd.DataFrame(
                {
                    "t_s": prefix_t,
                    "current_A": prefix_i,
                    "measured_voltage_V": prefix_v,
                    "pred_voltage_n_r_12_V": sim_nr12["voltage"],
                    f"pred_voltage_n_r_{int(args.n_r)}_V": sim_requested["voltage"],
                    "err_n_r_12_V": sim_nr12["voltage"] - prefix_v,
                    f"err_n_r_{int(args.n_r)}_V": sim_requested["voltage"] - prefix_v,
                }
            )
            diag_df.to_csv(output_dir / f"{prefix}_voltage_diagnostic_prefix{diag_n}.csv", index=False, encoding="utf-8-sig")

        summary = {
            "input": str(input_path),
            "resolved_csv": str(csv_path),
            "encoding_used": encoding_used,
            "cycle_start": cycle_start,
            "cycle_end": cycle_end,
            "n_rows": int(len(df)),
            "n_cycles": int(pd.to_numeric(df[colmap["cycle"]], errors="coerce").nunique()),
            "n_r": int(args.n_r),
            "device_used": sim["device"],
            "deg_i0_a": float(args.deg_i0_a if args.deg_i0_a is not None else params["default_deg_i0_a"].item()),
            "deg_ds_c": float(args.deg_ds_c if args.deg_ds_c is not None else params["default_deg_ds_c"].item()),
            "x_a0": float(args.x_a0 if args.x_a0 is not None else params["default_x_a0"].item()),
            "x_c0": float(args.x_c0 if args.x_c0 is not None else params["default_x_c0"].item()),
            "current_min_A": float(np.min(current_a)),
            "current_max_A": float(np.max(current_a)),
            "voltage_min_V": float(np.min(voltage_meas)),
            "voltage_max_V": float(np.max(voltage_meas)),
            "columns": colmap,
            "outputs": {
                "one_d_csv": str(one_d_csv),
                "states_npz": str(states_npz),
                "cs_a_npz": str(output_dir / f"{prefix}_cs_a_nr{int(args.n_r)}.npz"),
                "cs_c_npz": str(output_dir / f"{prefix}_cs_c_nr{int(args.n_r)}.npz"),
                "phie_csv": str(output_dir / f"{prefix}_phie_nr{int(args.n_r)}.csv"),
                "phis_c_csv": str(output_dir / f"{prefix}_phis_c_nr{int(args.n_r)}.csv"),
                "solution_alias": str(alias_path) if alias_path is not None else None,
            },
            "voltage_diagnostic": diagnostic,
        }
        summary_path = output_dir / f"{prefix}_summary_nr{int(args.n_r)}.json"
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    finally:
        if tmpdir is not None:
            tmpdir.cleanup()


if __name__ == "__main__":
    main()
