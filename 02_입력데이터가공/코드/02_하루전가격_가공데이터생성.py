#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

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
load_optional_timestamped_features = _COMMON.load_optional_timestamped_features
merge_optional_features = _COMMON.merge_optional_features
overwrite_with_issue_snapshot_features = _COMMON.overwrite_with_issue_snapshot_features
resolve_project_roots = _COMMON.resolve_project_roots
write_split_outputs = _COMMON.write_split_outputs

ALL_CSV = "05_하루전가격_가공데이터_전체.csv"
TRAIN_CSV = "06_하루전가격_가공데이터_학습기간.csv"
TEST_CSV = "07_하루전가격_가공데이터_테스트기간.csv"
META_JSON = "08_하루전가격_가공데이터_메타정보.json"


def build_dataset(
    source_csv: Path,
    out_root: Path,
    *,
    iso_load_source_csv: Path | None = None,
    da_as_source_csv: Path | None = None,
    system_summary_source_csv: Path | None = None,
    train_start: str = DEFAULT_TRAIN_START,
    train_end: str = DEFAULT_TRAIN_END,
    valid_start: str = DEFAULT_VALID_START,
    valid_end: str = DEFAULT_VALID_END,
    test_start: str = DEFAULT_TEST_START,
    test_end: str = DEFAULT_TEST_END,
) -> dict[str, object]:
    df = load_hourly_series_source(source_csv)
    df["da_price_actual"] = pd.to_numeric(df["DAM_LBMP"], errors="coerce")
    df["da_load_mw"] = pd.to_numeric(df.get("Load_MW"), errors="coerce")
    df["da_lbmp_mcc"] = pd.to_numeric(df.get("DAM_LBMP_mcc"), errors="coerce")
    df["da_lbmp_mcl"] = pd.to_numeric(df.get("DAM_LBMP_mcl"), errors="coerce")
    df["da_henry_hub_price"] = pd.to_numeric(df.get("Henry_Hub_Price"), errors="coerce")
    df["da_load_forecast_west"] = pd.to_numeric(df.get("Load_Forecast_West"), errors="coerce")

    iso_load = load_optional_timestamped_features(
        iso_load_source_csv,
        keep_cols=["west_load_forecast_mw", "nyiso_load_forecast_mw"],
        rename_map={
            "west_load_forecast_mw": "iso_west_load_forecast_mw",
            "nyiso_load_forecast_mw": "iso_nyiso_load_forecast_mw",
        },
        aggregate_freq="h",
    )
    df = merge_optional_features(df, iso_load)
    if "iso_west_load_forecast_mw" in df.columns:
        df["da_load_forecast_gap_west"] = df["da_load_forecast_west"] - df["iso_west_load_forecast_mw"]

    da_as = load_optional_timestamped_features(
        da_as_source_csv,
        keep_cols=["regulation_price", "spin10_price", "nspin10_price", "spin30_price"],
        rename_map={
            "regulation_price": "da_regulation_price_west",
            "spin10_price": "da_spin10_price_west",
            "nspin10_price": "da_nspin10_price_west",
            "spin30_price": "da_spin30_price_west",
        },
        aggregate_freq="h",
    )
    df = merge_optional_features(df, da_as)

    system_summary = load_optional_timestamped_features(
        system_summary_source_csv,
        keep_cols=["constraint_count", "constraint_shadowcost_sum", "constraint_shadowcost_abs_max"],
        rename_map={
            "constraint_count": "da_constraint_count",
            "constraint_shadowcost_sum": "da_constraint_shadowcost_sum",
            "constraint_shadowcost_abs_max": "da_constraint_shadowcost_abs_max",
        },
        aggregate_freq="h",
    )
    df = merge_optional_features(df, system_summary)

    df = attach_day_ahead_issue_columns(df)
    df = add_split_column(df, train_start, train_end, valid_start, valid_end, test_start, test_end)
    df, jump_meta = add_jump_label_features(df, target_col="da_price_actual", prefix="da_price")
    df = overwrite_with_issue_snapshot_features(
        df,
        value_cols=[
            "da_load_mw",
            "da_lbmp_mcc",
            "da_lbmp_mcl",
            "da_henry_hub_price",
            "da_regulation_price_west",
            "da_spin10_price_west",
            "da_nspin10_price_west",
            "da_spin30_price_west",
            "da_constraint_count",
            "da_constraint_shadowcost_sum",
            "da_constraint_shadowcost_abs_max",
        ],
    )
    df = add_cyclic_calendar_features(df)
    df = attach_issue_history_features(
        df,
        source_col="da_price_actual",
        prefix="da_price",
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
        "da_price_actual",
        "da_price_delta",
        "da_price_abs_delta",
        "da_price_jump_robust_z",
        "da_price_jump_spike_mad_upper",
        "da_price_jump_spike_top1pct",
        "da_price_jump_spike_top0_5pct",
        "da_price_body_flag",
        "da_load_mw",
        "da_lbmp_mcc",
        "da_lbmp_mcl",
        "da_henry_hub_price",
        "da_load_forecast_west",
        "month_sin",
        "month_cos",
        "hour_sin",
        "hour_cos",
        "dow_sin",
        "dow_cos",
    ]
    keep_cols += [f"da_price_lag{lag}" for lag in HOURLY_GBR_LAG_STEPS]
    keep_cols += [f"da_price_rollmean{window}" for window in HOURLY_GBR_ROLL_WINDOWS]

    optional_keep_cols = [
        "iso_west_load_forecast_mw",
        "iso_nyiso_load_forecast_mw",
        "da_load_forecast_gap_west",
        "da_regulation_price_west",
        "da_spin10_price_west",
        "da_nspin10_price_west",
        "da_spin30_price_west",
        "da_constraint_count",
        "da_constraint_shadowcost_sum",
        "da_constraint_shadowcost_abs_max",
    ]
    keep_cols += [col for col in optional_keep_cols if col in df.columns]

    out = df[keep_cols].dropna().reset_index(drop=True)

    meta = {
        "description": "누수 제거 규칙을 반영한 하루전 가격 가공데이터. issue-time 이전 actual과 허용 forecast, 보조서비스, 제약 요약을 포함한다.",
        "source_csv": str(source_csv),
        "target_col": "da_price_actual",
        "feature_groups": {
            "exogenous": [
                "da_load_mw",
                "da_lbmp_mcc",
                "da_lbmp_mcl",
                "da_henry_hub_price",
                "da_load_forecast_west",
            ]
            + [col for col in optional_keep_cols if col in out.columns],
            "time_cyc": ["month_sin", "month_cos", "hour_sin", "hour_cos", "dow_sin", "dow_cos"],
            "lags": [f"da_price_lag{lag}" for lag in HOURLY_GBR_LAG_STEPS],
            "rolling": [f"da_price_rollmean{window}" for window in HOURLY_GBR_ROLL_WINDOWS],
        },
        "train_start": train_start,
        "train_end": train_end,
        "valid_start": valid_start,
        "valid_end": valid_end,
        "test_start": test_start,
        "test_end": test_end,
        "jump_labeling": jump_meta,
        "columns": keep_cols,
    }
    return write_split_outputs(out, out_root, ALL_CSV, TRAIN_CSV, None, TEST_CSV, META_JSON, meta)


def main() -> None:
    roots = resolve_project_roots(__file__)
    ap = argparse.ArgumentParser(description="Build leak-safe DA price preprocessing dataset.")
    ap.add_argument("--source-csv", default=str(roots["input_root"] / "04_DA에너지_서부_학습데이터.csv"))
    ap.add_argument("--out-root", default=str(roots["prep_root"]))
    ap.add_argument("--iso-load-source-csv", default=str(roots["input_root"] / "07_NYISO_ISO부하예측_WEST.csv"))
    ap.add_argument("--da-as-source-csv", default=str(roots["input_root"] / "08_NYISO_DA_보조서비스가격_WEST.csv"))
    ap.add_argument("--system-summary-source-csv", default=str(roots["input_root"] / "13_NYISO_제약및발전기상태_요약.csv"))
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
        iso_load_source_csv=Path(args.iso_load_source_csv),
        da_as_source_csv=Path(args.da_as_source_csv),
        system_summary_source_csv=Path(args.system_summary_source_csv),
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
