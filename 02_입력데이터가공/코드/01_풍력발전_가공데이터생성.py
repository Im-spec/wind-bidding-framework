#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _load_common() -> object:
    module_path = Path(__file__).resolve().parent / "90_입력데이터가공_공통유틸.py"
    spec = importlib.util.spec_from_file_location("prep_common", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load helper from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_COMMON = _load_common()
CAPACITY_MW = _COMMON.CAPACITY_MW
DEFAULT_VALID_END = _COMMON.DEFAULT_VALID_END
DEFAULT_VALID_START = _COMMON.DEFAULT_VALID_START
DEFAULT_TEST_END = _COMMON.DEFAULT_TEST_END
DEFAULT_TEST_START = _COMMON.DEFAULT_TEST_START
DEFAULT_TRAIN_END = _COMMON.DEFAULT_TRAIN_END
DEFAULT_TRAIN_START = _COMMON.DEFAULT_TRAIN_START
HOURLY_GBR_LAG_STEPS = _COMMON.HOURLY_GBR_LAG_STEPS
HOURLY_GBR_ROLL_WINDOWS = _COMMON.HOURLY_GBR_ROLL_WINDOWS
add_cyclic_calendar_features = _COMMON.add_cyclic_calendar_features
add_jump_label_features = _COMMON.add_jump_label_features
add_split_column = _COMMON.add_split_column
attach_issue_history_features = _COMMON.attach_issue_history_features
attach_day_ahead_issue_columns = _COMMON.attach_day_ahead_issue_columns
load_hourly_series_source = _COMMON.load_hourly_series_source
resolve_project_roots = _COMMON.resolve_project_roots
write_split_outputs = _COMMON.write_split_outputs

ALL_CSV = "01_풍력발전_가공데이터_전체.csv"
TRAIN_CSV = "02_풍력발전_가공데이터_학습기간.csv"
TEST_CSV = "03_풍력발전_가공데이터_테스트기간.csv"
META_JSON = "04_풍력발전_가공데이터_메타정보.json"


def build_dataset(
    source_csv: Path,
    out_root: Path,
    train_start: str = DEFAULT_TRAIN_START,
    train_end: str = DEFAULT_TRAIN_END,
    valid_start: str = DEFAULT_VALID_START,
    valid_end: str = DEFAULT_VALID_END,
    test_start: str = DEFAULT_TEST_START,
    test_end: str = DEFAULT_TEST_END,
) -> dict[str, object]:
    df = load_hourly_series_source(source_csv)
    df["wind_actual_mw"] = np.clip(pd.to_numeric(df["Wind_MW_Gen"], errors="coerce"), 0.0, CAPACITY_MW)
    df["wind_actual_ratio"] = np.clip(df["wind_actual_mw"] / float(CAPACITY_MW), 0.0, 1.0)
    df["capacity_mw"] = float(CAPACITY_MW)
    df["v100"] = pd.to_numeric(df.get("v100"), errors="coerce")
    df["v10"] = pd.to_numeric(df.get("v10"), errors="coerce")
    df["sp"] = pd.to_numeric(df.get("sp"), errors="coerce")
    df = attach_day_ahead_issue_columns(df)
    df = add_split_column(df, train_start, train_end, valid_start, valid_end, test_start, test_end)
    df, jump_meta = add_jump_label_features(df, target_col="wind_actual_mw", prefix="wind")
    df = add_cyclic_calendar_features(df)
    df = attach_issue_history_features(
        df,
        source_col="wind_actual_mw",
        prefix="wind",
        lag_steps=HOURLY_GBR_LAG_STEPS,
        roll_windows=HOURLY_GBR_ROLL_WINDOWS,
    )

    keep_cols = [
        "timestamp_utc",
        "timestamp_et",
        "issue_timestamp_utc",
        "issue_timestamp_et",
        "issue_operating_date_et",
        "operating_date_et",
        "hour_et",
        "month",
        "weekday",
        "year",
        "split",
        "wind_actual_mw",
        "wind_actual_ratio",
        "capacity_mw",
        "wind_delta",
        "wind_abs_delta",
        "wind_jump_robust_z",
        "wind_jump_spike_mad_upper",
        "wind_jump_spike_top1pct",
        "wind_jump_spike_top0_5pct",
        "wind_body_flag",
        "v100",
        "v10",
        "sp",
        "month_sin",
        "month_cos",
        "hour_sin",
        "hour_cos",
        "dow_sin",
        "dow_cos",
    ]
    keep_cols += [f"wind_lag{lag}" for lag in HOURLY_GBR_LAG_STEPS]
    keep_cols += [f"wind_rollmean{window}" for window in HOURLY_GBR_ROLL_WINDOWS]

    out = df[keep_cols].dropna().reset_index(drop=True)
    meta = {
        "description": "설 논문 풍력 GBR 방식에 맞춘 풍력 가공데이터. 기상변수, cyclical calendar, 긴 lag, rolling mean 포함.",
        "source_csv": str(source_csv),
        "target_col": "wind_actual_mw",
        "feature_groups": {
            "weather": ["v100", "v10", "sp"],
            "time_cyc": ["month_sin", "month_cos", "hour_sin", "hour_cos", "dow_sin", "dow_cos"],
            "lags": [f"wind_lag{lag}" for lag in HOURLY_GBR_LAG_STEPS],
            "rolling": [f"wind_rollmean{window}" for window in HOURLY_GBR_ROLL_WINDOWS],
        },
        "train_start": train_start,
        "train_end": train_end,
        "valid_start": valid_start,
        "valid_end": valid_end,
        "test_start": test_start,
        "test_end": test_end,
        "capacity_mw": float(CAPACITY_MW),
        "jump_labeling": jump_meta,
        "columns": keep_cols,
    }
    return write_split_outputs(out, out_root, ALL_CSV, TRAIN_CSV, None, TEST_CSV, META_JSON, meta)


def main() -> None:
    roots = resolve_project_roots(__file__)
    ap = argparse.ArgumentParser(description="Build richer wind preprocessing dataset for HistGBR.")
    ap.add_argument("--source-csv", default=str(roots["input_root"] / "03_풍력발전_서부_학습데이터.csv"))
    ap.add_argument("--out-root", default=str(roots["prep_root"]))
    ap.add_argument("--train-start", default=DEFAULT_TRAIN_START)
    ap.add_argument("--train-end", default=DEFAULT_TRAIN_END)
    ap.add_argument("--valid-start", default=DEFAULT_VALID_START)
    ap.add_argument("--valid-end", default=DEFAULT_VALID_END)
    ap.add_argument("--test-start", default=DEFAULT_TEST_START)
    ap.add_argument("--test-end", default=DEFAULT_TEST_END)
    args = ap.parse_args()

    summary = build_dataset(
        Path(args.source_csv),
        Path(args.out_root),
        train_start=args.train_start,
        train_end=args.train_end,
        valid_start=args.valid_start,
        valid_end=args.valid_end,
        test_start=args.test_start,
        test_end=args.test_end,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
