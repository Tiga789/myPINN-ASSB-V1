from __future__ import annotations

import tempfile
import zipfile
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch

ENCODING_CANDIDATES_ASSBfinal1 = ["utf-8", "utf-8-sig", "gbk", "gb18030", "latin1"]


def _normalize_col_name_ASSBfinal1(name: object) -> str:
    text = str(name).strip().lower()
    for token in [" ", "（", "）", "(", ")", "_", "-", "/", "\\", "."]:
        text = text.replace(token, "")
    return text


def _first_existing_ASSBfinal1(df: pd.DataFrame, candidates: Iterable[str]) -> str:
    normalized = {_normalize_col_name_ASSBfinal1(col): col for col in df.columns}
    for candidate in candidates:
        key = _normalize_col_name_ASSBfinal1(candidate)
        if key in normalized:
            return normalized[key]
    raise KeyError(f"Missing required column. Tried: {list(candidates)}")


def detect_columns_ASSBfinal1(df: pd.DataFrame) -> dict[str, str]:
    return {
        "cycle": _first_existing_ASSBfinal1(df, ["循环号", "循环", "cycle", "循环序号"]),
        "total_time": _first_existing_ASSBfinal1(df, ["总时间", "累计时间", "累积时间", "totaltime"]),
        "current": _first_existing_ASSBfinal1(df, ["电流(A)", "电流", "current(a)", "current"]),
    }


def series_to_seconds_ASSBfinal1(series: pd.Series) -> np.ndarray:
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


def resolve_csv_from_input_ASSBfinal1(input_path: Path) -> tuple[Path, tempfile.TemporaryDirectory | None]:
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    if input_path.suffix.lower() == ".csv":
        return input_path, None
    if input_path.suffix.lower() != ".zip":
        raise ValueError("PROFILE_PATH must point to a CSV or a ZIP containing one CSV.")

    tmpdir = tempfile.TemporaryDirectory(prefix="assbfinal1_csv_")
    with zipfile.ZipFile(input_path, "r") as zf:
        csv_members = [m for m in zf.namelist() if m.lower().endswith(".csv")]
        if len(csv_members) != 1:
            raise ValueError(f"ZIP must contain exactly one CSV. Found: {csv_members}")
        zf.extract(csv_members[0], path=tmpdir.name)
        return Path(tmpdir.name) / csv_members[0], tmpdir


def read_csv_auto_ASSBfinal1(csv_path: Path, encoding: str = "auto") -> tuple[pd.DataFrame, str]:
    attempted: list[str] = []
    candidates = ENCODING_CANDIDATES_ASSBfinal1 if encoding == "auto" else [encoding]
    last_error: Exception | None = None
    for enc in candidates:
        try:
            return pd.read_csv(csv_path, encoding=enc), enc
        except Exception as exc:  # noqa: BLE001
            attempted.append(enc)
            last_error = exc
    raise RuntimeError(f"Failed to read CSV with encodings {attempted}") from last_error


def load_current_profile_ASSBfinal1(
    profile_path: str,
    encoding: str = "auto",
    cycle_start: int | None = None,
    cycle_end: int | None = None,
) -> dict[str, object]:
    input_path = Path(profile_path)
    csv_path, tmpdir = resolve_csv_from_input_ASSBfinal1(input_path)
    try:
        df, used_encoding = read_csv_auto_ASSBfinal1(csv_path, encoding=encoding)
        cols = detect_columns_ASSBfinal1(df)

        if cycle_start is None:
            cycle_start = int(pd.to_numeric(df[cols["cycle"]], errors="coerce").dropna().min())
        if cycle_end is None:
            cycle_end = int(pd.to_numeric(df[cols["cycle"]], errors="coerce").dropna().max())

        cycle_numeric = pd.to_numeric(df[cols["cycle"]], errors="coerce")
        keep = cycle_numeric.between(cycle_start, cycle_end, inclusive="both")
        df_f = df.loc[keep].copy()
        if df_f.empty:
            raise ValueError("No rows left after cycle filtering.")

        time_s = series_to_seconds_ASSBfinal1(df_f[cols["total_time"]])
        time_s = time_s - float(time_s[0])
        current_a = pd.to_numeric(df_f[cols["current"]], errors="coerce").ffill().bfill().to_numpy(dtype=float)
        if time_s.ndim != 1 or current_a.ndim != 1 or time_s.size != current_a.size:
            raise ValueError("Profile arrays must be one-dimensional and aligned.")
        if time_s.size < 2:
            raise ValueError("Need at least two time points.")
        if np.any(np.diff(time_s) < 0.0):
            raise ValueError("Time axis must be nondecreasing.")

        return {
            "time_s": time_s.astype(np.float64),
            "current_a": current_a.astype(np.float64),
            "cycle_start": int(cycle_start),
            "cycle_end": int(cycle_end),
            "encoding": used_encoding,
            "csv_path": str(csv_path),
            "profile_path": str(input_path),
        }
    finally:
        if tmpdir is not None:
            tmpdir.cleanup()


def sample_profile_zoh_left_ASSBfinal1(
    t_query: torch.Tensor,
    t_ref: torch.Tensor,
    y_ref: torch.Tensor,
) -> torch.Tensor:
    tq = torch.as_tensor(t_query, dtype=torch.float64, device=t_ref.device)
    flat = tq.reshape(-1)
    tref = torch.as_tensor(t_ref, dtype=torch.float64, device=t_ref.device).reshape(-1)
    yref = torch.as_tensor(y_ref, dtype=torch.float64, device=t_ref.device).reshape(-1)
    if tref.numel() != yref.numel():
        raise ValueError("t_ref and y_ref must have the same length.")
    if tref.numel() == 0:
        raise ValueError("Empty profile reference arrays.")
    idx = torch.bucketize(flat.detach(), tref, right=True) - 1
    idx = torch.clamp(idx, min=0, max=tref.numel() - 1)
    out = yref[idx]
    return out.reshape_as(tq)


def select_step_indices_ASSBfinal1(t_query: torch.Tensor, t_nodes: torch.Tensor) -> torch.Tensor:
    tq = torch.as_tensor(t_query, dtype=torch.float64, device=t_nodes.device).reshape(-1)
    tref = torch.as_tensor(t_nodes, dtype=torch.float64, device=t_nodes.device).reshape(-1)
    if tref.numel() < 2:
        raise ValueError("Need at least two profile time nodes.")
    idx = torch.bucketize(tq.detach(), tref, right=True) - 1
    idx = torch.clamp(idx, min=0, max=tref.numel() - 2)
    return idx
