"""Generate model-voltage truth CSV from the local ZHB_ASSB_NCM811.xlsx record sheet.

This script ignores the source voltage column and only uses the current/time sequence.
The output CSV preserves the original key columns and inserts the model voltage right
after the current column.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch

CURRENT_DIR = Path(__file__).resolve().parent
UTIL_DIR = CURRENT_DIR.parent / "util"
if str(UTIL_DIR) not in sys.path:
    sys.path.insert(0, str(UTIL_DIR))

from spm_experimental_1 import make_params_experimental_1
from spm_int_experimental_1 import simulate_current_profile_experimental_1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--xlsx", type=str, required=True, help="Path to ZHB_ASSB_NCM811.xlsx")
    parser.add_argument("--sheet", type=str, default="record", help="Worksheet name containing the record data")
    parser.add_argument("--cycle-start", type=int, default=5)
    parser.add_argument("--cycle-end", type=int, default=521)
    parser.add_argument("--output-dir", type=str, default=None, help="Default: same folder as the xlsx file")
    parser.add_argument("--output-name", type=str, default="ZHB_ASSB_NCM811_cycle5_521_voltage_truth_experimental_1.csv")
    parser.add_argument("--npz-name", type=str, default="ZHB_ASSB_NCM811_cycle5_521_states_experimental_1.npz")
    parser.add_argument("--summary-name", type=str, default="ZHB_ASSB_NCM811_cycle5_521_summary_experimental_1.json")
    parser.add_argument("--n-r", type=int, default=12)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--deg-i0-a", type=float, default=None)
    parser.add_argument("--deg-ds-c", type=float, default=None)
    parser.add_argument("--x-a0", type=float, default=None)
    parser.add_argument("--x-c0", type=float, default=None)
    parser.add_argument("--save-npz", action="store_true", help="Also save the 4-field states as NPZ for later soft-label use")
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
        "absolute_time": _first_existing(df, ["绝对时间", "日期时间", "测试时间", "datetime", "timestamp"]),
    }


def read_record_sheet(xlsx_path: str, preferred_sheet: str) -> tuple[pd.DataFrame, str]:
    xls = pd.ExcelFile(xlsx_path, engine="openpyxl")
    sheet_map = {str(name).lower(): name for name in xls.sheet_names}
    if preferred_sheet.lower() in sheet_map:
        sheet_name = sheet_map[preferred_sheet.lower()]
    elif "record" in sheet_map:
        sheet_name = sheet_map["record"]
    else:
        raise KeyError(f"Cannot find sheet '{preferred_sheet}'. Available sheets: {xls.sheet_names}")
    df = pd.read_excel(xlsx_path, sheet_name=sheet_name, engine="openpyxl")
    return df, sheet_name


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
            # Excel durations are commonly stored as fractions of a day.
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


def main() -> None:
    args = parse_args()
    xlsx_path = Path(args.xlsx)
    if not xlsx_path.exists():
        raise FileNotFoundError(f"Input file not found: {xlsx_path}")

    out_dir = Path(args.output_dir) if args.output_dir else xlsx_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    df_raw, sheet_name = read_record_sheet(str(xlsx_path), args.sheet)
    colmap = detect_columns(df_raw)

    cycle_series = pd.to_numeric(df_raw[colmap["cycle"]], errors="coerce")
    mask = (cycle_series >= int(args.cycle_start)) & (cycle_series <= int(args.cycle_end))
    df = df_raw.loc[mask].copy()
    if df.empty:
        raise ValueError("The selected cycle range is empty.")

    total_seconds = series_to_seconds(df[colmap["total_time"]])
    total_seconds = total_seconds - float(total_seconds[0])
    current_a = pd.to_numeric(df[colmap["current"]], errors="coerce").fillna(0.0).to_numpy(dtype=float)

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
    voltage_pred = sim["voltage"]

    time_out = format_time_like_series_for_output(df[colmap["time"]], mode="duration")
    total_time_out = format_time_like_series_for_output(df[colmap["total_time"]], mode="duration")
    absolute_time_out = format_time_like_series_for_output(df[colmap["absolute_time"]], mode="absolute")

    output_df = pd.DataFrame(
        {
            colmap["data_index"]: df[colmap["data_index"]].to_numpy(),
            colmap["cycle"]: df[colmap["cycle"]].to_numpy(),
            colmap["step"]: df[colmap["step"]].to_numpy(),
            colmap["step_type"]: df[colmap["step_type"]].to_numpy(),
            colmap["time"]: time_out.to_numpy(),
            colmap["total_time"]: total_time_out.to_numpy(),
            colmap["current"]: df[colmap["current"]].to_numpy(),
            "拟合电压(V)": voltage_pred,
            colmap["absolute_time"]: absolute_time_out.to_numpy(),
        }
    )

    csv_path = out_dir / args.output_name
    output_df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    summary = {
        "input_xlsx": str(xlsx_path),
        "sheet_name": sheet_name,
        "cycle_start": int(args.cycle_start),
        "cycle_end": int(args.cycle_end),
        "n_rows": int(len(output_df)),
        "n_r": int(args.n_r),
        "device_used": sim["device"],
        "deg_i0_a": float(args.deg_i0_a if args.deg_i0_a is not None else params["default_deg_i0_a"].item()),
        "deg_ds_c": float(args.deg_ds_c if args.deg_ds_c is not None else params["default_deg_ds_c"].item()),
        "x_a0": float(args.x_a0 if args.x_a0 is not None else params["default_x_a0"].item()),
        "x_c0": float(args.x_c0 if args.x_c0 is not None else params["default_x_c0"].item()),
        "output_csv": str(csv_path),
    }
    (out_dir / args.summary_name).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.save_npz:
        np.savez(
            out_dir / args.npz_name,
            t=sim["t"],
            r_a=sim["r_a"],
            r_c=sim["r_c"],
            cs_a=sim["cs_a"],
            cs_c=sim["cs_c"],
            phie=sim["phie"],
            phis_c=sim["phis_c"],
            voltage=sim["voltage"],
            current=current_a,
            cycle=df[colmap["cycle"]].to_numpy(),
            data_index=df[colmap["data_index"]].to_numpy(),
        )


if __name__ == "__main__":
    main()
