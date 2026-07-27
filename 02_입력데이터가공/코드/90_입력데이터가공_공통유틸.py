#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

TZ_ET = "America/New_York"
# Original paper split:
# train = 2018-2019, validation = 2020, test = 2021.
DEFAULT_TRAIN_START = "2018-01-01"
DEFAULT_TRAIN_END = "2019-12-31"
DEFAULT_VALID_START = "2020-01-01"
DEFAULT_VALID_END = "2020-12-31"
DEFAULT_TEST_START = "2021-01-01"
DEFAULT_TEST_END = "2021-12-31"
DAY_AHEAD_ISSUE_HOUR_ET = 5
CAPACITY_MW = 1985.3

HOURLY_GBR_LAG_STEPS = (1, 2, 3, 6, 12, 24, 48, 168)
HOURLY_GBR_ROLL_WINDOWS = (6, 24, 72)
RT_GBR_LAG_STEPS = (1, 3, 6, 12, 24, 288)
RT_GBR_ROLL_WINDOWS = (12, 72, 288)
REG_LAG_STEPS = HOURLY_GBR_LAG_STEPS
REG_ROLL_WINDOWS = HOURLY_GBR_ROLL_WINDOWS


def project_root_from_here(file_path: str | Path) -> Path:
    return Path(file_path).resolve().parents[2]


def resolve_project_roots(file_path: str | Path) -> dict[str, Path]:
    root = project_root_from_here(file_path)
    return {
        "project_root": root,
        "input_root": root / "01_입력데이터",
        "prep_root": root / "02_입력데이터가공" / "가공데이터",
        "point_root": root / "03_점예측",
        "point_internal_root": root / "03_점예측" / "점예측내부산출물",
        "point_result_root": root / "03_점예측" / "점예측결과",
    }


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def dump_json(path: Path, payload: dict[str, object]) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def add_time_columns(df: pd.DataFrame, timestamp_col: str = "timestamp_utc") -> pd.DataFrame:
    out = df.copy()
    out[timestamp_col] = pd.to_datetime(out[timestamp_col], utc=True, errors="coerce")
    out = out.dropna(subset=[timestamp_col]).copy()
    ts_et = out[timestamp_col].dt.tz_convert(TZ_ET)
    out["timestamp_utc"] = out[timestamp_col]
    out["timestamp_et"] = ts_et.astype(str)
    out["operating_date_et"] = ts_et.dt.strftime("%Y-%m-%d")
    out["hour_et"] = ts_et.dt.hour.astype(int)
    out["minute_et"] = ts_et.dt.minute.astype(int)
    out["slot_5m_et"] = (out["minute_et"] // 5).astype(int)
    out["month"] = ts_et.dt.month.astype(int)
    out["weekday"] = ts_et.dt.dayofweek.astype(int)
    out["year"] = ts_et.dt.year.astype(int)
    return out


def attach_day_ahead_issue_columns(
    df: pd.DataFrame,
    *,
    timestamp_col: str = "timestamp_utc",
    cutoff_hour_et: int = DAY_AHEAD_ISSUE_HOUR_ET,
) -> pd.DataFrame:
    out = df.copy()
    out[timestamp_col] = pd.to_datetime(out[timestamp_col], utc=True, errors="coerce")
    out = out.dropna(subset=[timestamp_col]).copy()
    ts_et = out[timestamp_col].dt.tz_convert(TZ_ET)
    issue_et = ts_et.dt.normalize() - pd.Timedelta(days=1) + pd.Timedelta(hours=cutoff_hour_et)
    out["issue_timestamp_utc"] = issue_et.dt.tz_convert("UTC")
    out["issue_timestamp_et"] = issue_et.astype(str)
    out["issue_operating_date_et"] = issue_et.dt.strftime("%Y-%m-%d")
    return out


def add_split_column(
    df: pd.DataFrame,
    train_start: str = DEFAULT_TRAIN_START,
    train_end: str = DEFAULT_TRAIN_END,
    valid_start: str = DEFAULT_VALID_START,
    valid_end: str = DEFAULT_VALID_END,
    test_start: str = DEFAULT_TEST_START,
    test_end: str = DEFAULT_TEST_END,
) -> pd.DataFrame:
    out = df.copy()
    dates = pd.to_datetime(out["operating_date_et"], errors="coerce")
    out["split"] = "other"
    out.loc[(dates >= pd.Timestamp(train_start)) & (dates <= pd.Timestamp(train_end)), "split"] = "train"
    out.loc[(dates >= pd.Timestamp(valid_start)) & (dates <= pd.Timestamp(valid_end)), "split"] = "valid"
    out.loc[(dates >= pd.Timestamp(test_start)) & (dates <= pd.Timestamp(test_end)), "split"] = "test"
    return out


def add_cyclic_calendar_features(df: pd.DataFrame, *, include_slot: bool = False) -> pd.DataFrame:
    out = df.copy()
    out["month_sin"] = np.sin(2 * np.pi * out["month"].astype(float) / 12.0)
    out["month_cos"] = np.cos(2 * np.pi * out["month"].astype(float) / 12.0)
    out["hour_sin"] = np.sin(2 * np.pi * out["hour_et"].astype(float) / 24.0)
    out["hour_cos"] = np.cos(2 * np.pi * out["hour_et"].astype(float) / 24.0)
    out["dow_sin"] = np.sin(2 * np.pi * out["weekday"].astype(float) / 7.0)
    out["dow_cos"] = np.cos(2 * np.pi * out["weekday"].astype(float) / 7.0)
    if include_slot:
        out["slot_sin"] = np.sin(2 * np.pi * out["slot_5m_et"].astype(float) / 12.0)
        out["slot_cos"] = np.cos(2 * np.pi * out["slot_5m_et"].astype(float) / 12.0)
    return out


def add_direct_lag_features(
    df: pd.DataFrame,
    *,
    source_col: str,
    prefix: str,
    lag_steps: tuple[int, ...],
) -> pd.DataFrame:
    out = df.copy().sort_values("timestamp_utc").reset_index(drop=True)
    series = pd.to_numeric(out[source_col], errors="coerce")
    for lag in lag_steps:
        out[f"{prefix}_lag{lag}"] = series.shift(int(lag))
    return out


def add_direct_rolling_features(
    df: pd.DataFrame,
    *,
    source_col: str,
    prefix: str,
    roll_windows: tuple[int, ...],
) -> pd.DataFrame:
    out = df.copy().sort_values("timestamp_utc").reset_index(drop=True)
    shifted = pd.to_numeric(out[source_col], errors="coerce").shift(1)
    for window in roll_windows:
        out[f"{prefix}_rollmean{window}"] = shifted.rolling(int(window)).mean()
    return out


def overwrite_with_issue_snapshot_features(
    df: pd.DataFrame,
    *,
    value_cols: list[str],
    history_df: pd.DataFrame | None = None,
    feature_timestamp_col: str = "timestamp_utc",
    issue_timestamp_col: str = "issue_timestamp_utc",
    strict_before: bool = True,
) -> pd.DataFrame:
    out = df.copy()
    available_cols = [col for col in value_cols if col in out.columns]
    if history_df is None:
        history_source = out
    else:
        history_source = history_df.copy()
        available_cols = [col for col in available_cols if col in history_source.columns]
    if not available_cols:
        return out

    history = history_source[[feature_timestamp_col] + available_cols].copy()
    history[feature_timestamp_col] = pd.to_datetime(history[feature_timestamp_col], utc=True, errors="coerce")
    history = history.dropna(subset=[feature_timestamp_col]).sort_values(feature_timestamp_col).reset_index(drop=True)
    rename_map = {feature_timestamp_col: "__issuehist_timestamp_utc"}
    rename_map.update({col: f"__issuehist_{col}" for col in available_cols})
    history = history.rename(columns=rename_map)

    left = out.reset_index().rename(columns={"index": "__row_id"})
    left[issue_timestamp_col] = pd.to_datetime(left[issue_timestamp_col], utc=True, errors="coerce")
    left = left.sort_values(issue_timestamp_col).reset_index(drop=True)

    merged = pd.merge_asof(
        left,
        history,
        left_on=issue_timestamp_col,
        right_on="__issuehist_timestamp_utc",
        direction="backward",
        allow_exact_matches=not strict_before,
    )
    for col in available_cols:
        merged[col] = pd.to_numeric(merged[f"__issuehist_{col}"], errors="coerce")
    drop_cols = ["__issuehist_timestamp_utc"] + [f"__issuehist_{col}" for col in available_cols]
    merged = merged.drop(columns=[col for col in drop_cols if col in merged.columns])
    merged = merged.sort_values("__row_id").drop(columns=["__row_id"]).reset_index(drop=True)
    return merged


def attach_issue_history_features(
    df: pd.DataFrame,
    *,
    source_col: str,
    prefix: str,
    lag_steps: tuple[int, ...],
    roll_windows: tuple[int, ...],
    history_df: pd.DataFrame | None = None,
    feature_timestamp_col: str = "timestamp_utc",
    issue_timestamp_col: str = "issue_timestamp_utc",
    strict_before: bool = True,
) -> pd.DataFrame:
    out = df.copy()
    history_source = out if history_df is None else history_df.copy()
    if source_col not in history_source.columns:
        return out

    history = history_source[[feature_timestamp_col, source_col]].copy()
    history[feature_timestamp_col] = pd.to_datetime(history[feature_timestamp_col], utc=True, errors="coerce")
    history[source_col] = pd.to_numeric(history[source_col], errors="coerce")
    history = history.dropna(subset=[feature_timestamp_col, source_col]).sort_values(feature_timestamp_col).reset_index(drop=True)
    if history.empty:
        return out

    history_features = pd.DataFrame({"__issuehist_timestamp_utc": history[feature_timestamp_col]})
    series = history[source_col]
    for lag in lag_steps:
        lag_name = f"{prefix}_lag{lag}"
        history_features[lag_name] = series.shift(max(int(lag) - 1, 0))
    for window in roll_windows:
        roll_name = f"{prefix}_rollmean{window}"
        history_features[roll_name] = series.rolling(int(window)).mean()

    left = out.reset_index().rename(columns={"index": "__row_id"})
    left[issue_timestamp_col] = pd.to_datetime(left[issue_timestamp_col], utc=True, errors="coerce")
    left = left.sort_values(issue_timestamp_col).reset_index(drop=True)
    merged = pd.merge_asof(
        left,
        history_features,
        left_on=issue_timestamp_col,
        right_on="__issuehist_timestamp_utc",
        direction="backward",
        allow_exact_matches=not strict_before,
    )
    lag_cols = [f"{prefix}_lag{lag}" for lag in lag_steps]
    roll_cols = [f"{prefix}_rollmean{window}" for window in roll_windows]
    for col in lag_cols + roll_cols:
        if col in merged.columns:
            merged[col] = pd.to_numeric(merged[col], errors="coerce")
    merged = merged.drop(columns=["__issuehist_timestamp_utc"])
    merged = merged.sort_values("__row_id").drop(columns=["__row_id"]).reset_index(drop=True)
    return merged


def add_jump_label_features(
    df: pd.DataFrame,
    *,
    target_col: str,
    prefix: str,
    split_col: str = "split",
    robust_z_threshold: float = 3.5,
    top_share_1: float = 0.01,
    top_share_05: float = 0.005,
) -> tuple[pd.DataFrame, dict[str, object]]:
    out = df.copy().sort_values("timestamp_utc").reset_index(drop=True)
    value = pd.to_numeric(out[target_col], errors="coerce")
    delta_col = f"{prefix}_delta"
    abs_delta_col = f"{prefix}_abs_delta"
    robust_z_col = f"{prefix}_jump_robust_z"
    mad_flag_col = f"{prefix}_jump_spike_mad_upper"
    top1_flag_col = f"{prefix}_jump_spike_top1pct"
    top05_flag_col = f"{prefix}_jump_spike_top0_5pct"
    body_flag_col = f"{prefix}_body_flag"

    out[delta_col] = value.diff()
    out[abs_delta_col] = out[delta_col].abs()

    train_abs_delta = out.loc[out[split_col] == "train", abs_delta_col].dropna().astype(float)
    if train_abs_delta.empty:
        raise ValueError(f"No train abs-delta rows for {prefix}")

    arr = train_abs_delta.to_numpy(float)
    median = float(np.median(arr))
    mad = float(np.median(np.abs(arr - median)))
    top1_cutoff = float(train_abs_delta.quantile(1.0 - top_share_1))
    top05_cutoff = float(train_abs_delta.quantile(1.0 - top_share_05))
    if not np.isfinite(mad) or mad <= 0:
        mad = 0.0
        mad_upper = top1_cutoff
        robust_z = pd.Series(np.zeros(len(out), dtype=float), index=out.index)
    else:
        mad_scale = 0.6744897501960817
        mad_upper = float(median + robust_z_threshold * mad / mad_scale)
        robust_z = mad_scale * (out[abs_delta_col] - median) / mad

    out[robust_z_col] = robust_z.astype(float)
    out[mad_flag_col] = (out[abs_delta_col] >= mad_upper).astype("Int64")
    out[top1_flag_col] = (out[abs_delta_col] >= top1_cutoff).astype("Int64")
    out[top05_flag_col] = (out[abs_delta_col] >= top05_cutoff).astype("Int64")
    out.loc[out[abs_delta_col].isna(), [mad_flag_col, top1_flag_col, top05_flag_col]] = pd.NA
    out[body_flag_col] = (out[top1_flag_col] == 0).astype("Int64")
    out.loc[out[abs_delta_col].isna(), body_flag_col] = pd.NA

    meta = {
        "target_col": target_col,
        "delta_col": delta_col,
        "abs_delta_col": abs_delta_col,
        "robust_z_col": robust_z_col,
        "mad_flag_col": mad_flag_col,
        "top1_flag_col": top1_flag_col,
        "top05_flag_col": top05_flag_col,
        "body_flag_col": body_flag_col,
        "robust_z_threshold": float(robust_z_threshold),
        "train_n": int(train_abs_delta.shape[0]),
        "train_median_abs_delta": median,
        "train_mad_abs_delta": mad,
        "mad_upper_cutoff": float(mad_upper),
        "top1pct_cutoff": top1_cutoff,
        "top0_5pct_cutoff": top05_cutoff,
        "train_top1pct_count": int((out.loc[out[split_col] == "train", top1_flag_col] == 1).sum()),
        "train_top0_5pct_count": int((out.loc[out[split_col] == "train", top05_flag_col] == 1).sum()),
    }
    return out, meta


def load_hourly_series_source(source_csv: Path, *, timestamp_col: str = "TimeStamp") -> pd.DataFrame:
    raw = pd.read_csv(source_csv, encoding="utf-8-sig")
    if timestamp_col not in raw.columns:
        raise KeyError(f"Missing timestamp column {timestamp_col}: {source_csv}")
    raw["timestamp_utc"] = pd.to_datetime(raw[timestamp_col], utc=True, errors="coerce")
    raw = raw.dropna(subset=["timestamp_utc"]).copy()
    return add_time_columns(raw, "timestamp_utc").sort_values("timestamp_utc").reset_index(drop=True)


def load_rt_5m_series_source(source_csv: Path, *, timestamp_col: str = "TimeStamp") -> pd.DataFrame:
    return load_hourly_series_source(source_csv, timestamp_col=timestamp_col)


def load_reg_price_source(source_csv: Path, target_col: str) -> pd.DataFrame:
    raw = pd.read_csv(source_csv, encoding="utf-8-sig")
    raw["timestamp_utc"] = pd.to_datetime(raw["timestamp_utc"], utc=True, errors="coerce")
    raw = raw.dropna(subset=["timestamp_utc"]).copy()
    if target_col not in raw.columns:
        raise KeyError(f"Missing target column {target_col}: {source_csv}")
    raw[target_col] = pd.to_numeric(raw[target_col], errors="coerce")
    raw = add_time_columns(raw, "timestamp_utc")
    keep_cols = [
        "timestamp_utc",
        "timestamp_et",
        "operating_date_et",
        "hour_et",
        "month",
        "weekday",
        "year",
        target_col,
    ]
    return (
        raw[keep_cols]
        .dropna(subset=[target_col])
        .sort_values("timestamp_utc")
        .drop_duplicates(subset=["timestamp_utc"], keep="last")
        .reset_index(drop=True)
    )


def load_optional_timestamped_features(
    source_csv: Path | None,
    *,
    keep_cols: list[str],
    rename_map: dict[str, str] | None = None,
    aggregate_freq: str | None = None,
    aggregate_func: str = "mean",
) -> pd.DataFrame | None:
    if source_csv is None or not source_csv.exists():
        return None
    raw = pd.read_csv(source_csv, encoding="utf-8-sig")
    if "timestamp_utc" not in raw.columns:
        raise KeyError(f"Missing timestamp_utc column: {source_csv}")
    raw["timestamp_utc"] = pd.to_datetime(raw["timestamp_utc"], utc=True, errors="coerce")
    raw = raw.dropna(subset=["timestamp_utc"]).copy()
    available_cols = [col for col in keep_cols if col in raw.columns]
    if not available_cols:
        return None
    out = raw[["timestamp_utc"] + available_cols].copy()
    for col in available_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    if rename_map:
        valid_map = {src: dst for src, dst in rename_map.items() if src in out.columns}
        out = out.rename(columns=valid_map)
    if aggregate_freq:
        out["timestamp_utc"] = out["timestamp_utc"].dt.floor(aggregate_freq)
        value_cols = [col for col in out.columns if col != "timestamp_utc"]
        out = (
            out.groupby("timestamp_utc", as_index=False)
            .agg({col: aggregate_func for col in value_cols})
            .sort_values("timestamp_utc")
            .reset_index(drop=True)
        )
    else:
        out = (
            out.sort_values("timestamp_utc")
            .drop_duplicates(subset=["timestamp_utc"], keep="last")
            .reset_index(drop=True)
        )
    return out


def aggregate_hourly_abs_stats(df: pd.DataFrame | None, source_col: str, prefix: str) -> pd.DataFrame | None:
    if df is None or source_col not in df.columns:
        return None
    out = df[["timestamp_utc", source_col]].copy()
    out["timestamp_utc"] = pd.to_datetime(out["timestamp_utc"], utc=True, errors="coerce").dt.floor("h")
    out[source_col] = pd.to_numeric(out[source_col], errors="coerce")
    out = out.dropna(subset=["timestamp_utc", source_col]).copy()
    grouped = (
        out.groupby("timestamp_utc", as_index=False)
        .agg(
            **{
                f"{prefix}_mean": (source_col, "mean"),
                f"{prefix}_abs_mean": (source_col, lambda s: s.abs().mean()),
                f"{prefix}_abs_max": (source_col, lambda s: s.abs().max()),
            }
        )
        .sort_values("timestamp_utc")
        .reset_index(drop=True)
    )
    return grouped


def merge_optional_features(base_df: pd.DataFrame, extra_df: pd.DataFrame | None, *, on: str = "timestamp_utc") -> pd.DataFrame:
    if extra_df is None or extra_df.empty:
        return base_df
    return base_df.merge(extra_df, on=on, how="left")


def tokenized_colname(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(name).lower())


def write_split_outputs(
    df: pd.DataFrame,
    out_root: Path,
    all_name: str,
    train_name: str,
    valid_name: str | None,
    test_name: str,
    meta_name: str,
    meta: dict[str, object],
) -> dict[str, object]:
    ensure_dir(out_root)
    all_path = out_root / all_name
    train_path = out_root / train_name
    valid_path = out_root / valid_name if valid_name else None
    test_path = out_root / test_name
    meta_path = out_root / meta_name

    df.to_csv(all_path, index=False, encoding="utf-8-sig")
    df.loc[df["split"] == "train"].to_csv(train_path, index=False, encoding="utf-8-sig")
    if valid_path is not None:
        df.loc[df["split"] == "valid"].to_csv(valid_path, index=False, encoding="utf-8-sig")
    df.loc[df["split"] == "test"].to_csv(test_path, index=False, encoding="utf-8-sig")
    dump_json(meta_path, meta)
    summary = {
        "all_csv": str(all_path),
        "train_csv": str(train_path),
        "valid_csv": str(valid_path) if valid_path is not None else None,
        "test_csv": str(test_path),
        "meta_json": str(meta_path),
        "n_rows": int(len(df)),
        "n_train_rows": int((df["split"] == "train").sum()),
        "n_valid_rows": int((df["split"] == "valid").sum()),
        "n_test_rows": int((df["split"] == "test").sum()),
    }
    return summary
