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
add_time_columns = _COMMON.add_time_columns
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

ALL_CSV = "13_DA_REG가격_가공데이터_전체.csv"
TRAIN_CSV = "14_DA_REG가격_가공데이터_학습기간.csv"
TEST_CSV = "15_DA_REG가격_가공데이터_테스트기간.csv"
META_JSON = "16_DA_REG가격_가공데이터_메타정보.json"


def load_reg_price_source_full(source_csv: Path, target_col: str) -> pd.DataFrame:
    raw = pd.read_csv(source_csv, encoding="utf-8-sig")
    raw["timestamp_utc"] = pd.to_datetime(raw["timestamp_utc"], utc=True, errors="coerce")
    raw = raw.dropna(subset=["timestamp_utc"]).copy()
    if target_col not in raw.columns:
        raise KeyError(f"Missing target column {target_col}: {source_csv}")
    raw[target_col] = pd.to_numeric(raw[target_col], errors="coerce")
    raw = add_time_columns(raw, "timestamp_utc")
    base_cols = [
        "timestamp_utc",
        "timestamp_et",
        "operating_date_et",
        "hour_et",
        "month",
        "weekday",
        "year",
        target_col,
    ]
    ordered_cols = base_cols + [col for col in raw.columns if col not in base_cols]
    return (
        raw[ordered_cols]
        .dropna(subset=[target_col])
        .sort_values("timestamp_utc")
        .drop_duplicates(subset=["timestamp_utc"], keep="last")
        .reset_index(drop=True)
    )


def build_dataset(
    source_csv: Path,
    da_source_csv: Path,
    rt_source_csv: Path,
    wind_source_csv: Path,
    out_root: Path,
    *,
    iso_load_source_csv: Path | None = None,
    da_as_source_csv: Path | None = None,
    reg_requirement_source_csv: Path | None = None,
    system_summary_source_csv: Path | None = None,
    ace_source_csv: Path | None = None,
    train_start: str = DEFAULT_TRAIN_START,
    train_end: str = DEFAULT_TRAIN_END,
    valid_start: str = DEFAULT_VALID_START,
    valid_end: str = DEFAULT_VALID_END,
    test_start: str = DEFAULT_TEST_START,
    test_end: str = DEFAULT_TEST_END,
) -> dict[str, object]:
    reg = load_reg_price_source_full(source_csv, "da_reg_actual")

    da = load_hourly_series_source(da_source_csv)[["timestamp_utc", "DAM_LBMP", "Load_Forecast_West", "Henry_Hub_Price"]].copy()
    da["da_price_actual"] = pd.to_numeric(da["DAM_LBMP"], errors="coerce")
    da["da_load_forecast_west"] = pd.to_numeric(da["Load_Forecast_West"], errors="coerce")
    da["da_henry_hub_price"] = pd.to_numeric(da["Henry_Hub_Price"], errors="coerce")
    da = da[["timestamp_utc", "da_price_actual", "da_load_forecast_west", "da_henry_hub_price"]]

    rt = load_hourly_series_source(rt_source_csv)
    rt["timestamp_hour_utc"] = rt["timestamp_utc"].dt.floor("h")
    rt_hourly = (
        rt.groupby("timestamp_hour_utc", as_index=False)
        .agg(
            rt_price_actual=("RT_LBMP", "mean"),
            rt_load_forecast_west=("Load_Forecast_West", "mean"),
            rt_henry_hub_price=("Henry_Hub_Price", "mean"),
        )
        .rename(columns={"timestamp_hour_utc": "timestamp_utc"})
    )

    wind = load_hourly_series_source(wind_source_csv)[["timestamp_utc", "Wind_MW_Gen"]].copy()
    wind["wind_actual_mw"] = pd.to_numeric(wind["Wind_MW_Gen"], errors="coerce")
    wind = wind[["timestamp_utc", "wind_actual_mw"]]

    df = reg.merge(da, on="timestamp_utc", how="left").merge(rt_hourly, on="timestamp_utc", how="left").merge(wind, on="timestamp_utc", how="left")

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

    reg_requirement = load_optional_timestamped_features(
        reg_requirement_source_csv,
        keep_cols=["regulation_requirement_mw"],
        rename_map={"regulation_requirement_mw": "da_reg_requirement_mw"},
        aggregate_freq="h",
    )
    df = merge_optional_features(df, reg_requirement)

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

    ace = load_optional_timestamped_features(
        ace_source_csv,
        keep_cols=["ace_mw"],
        rename_map={"ace_mw": "da_ace_mw"},
        aggregate_freq="h",
    )
    df = merge_optional_features(df, ace)
    if "da_ace_mw" in df.columns:
        df["da_ace_abs"] = df["da_ace_mw"].abs()

    df = attach_day_ahead_issue_columns(df)
    df = add_split_column(df, train_start, train_end, valid_start, valid_end, test_start, test_end)
    df, jump_meta = add_jump_label_features(df, target_col="da_reg_actual", prefix="da_reg")
    df = overwrite_with_issue_snapshot_features(
        df,
        value_cols=[
            "da_price_actual",
            "rt_price_actual",
            "wind_actual_mw",
            "da_henry_hub_price",
            "rt_henry_hub_price",
            "da_regulation_price_west",
            "da_spin10_price_west",
            "da_nspin10_price_west",
            "da_spin30_price_west",
            "da_reg_requirement_mw",
            "da_constraint_count",
            "da_constraint_shadowcost_sum",
            "da_constraint_shadowcost_abs_max",
            "da_ace_mw",
            "da_ace_abs",
        ],
    )
    df = add_cyclic_calendar_features(df)
    df = attach_issue_history_features(
        df,
        source_col="da_reg_actual",
        prefix="da_reg",
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
        "da_reg_actual",
        "da_reg_delta",
        "da_reg_abs_delta",
        "da_reg_jump_robust_z",
        "da_reg_jump_spike_mad_upper",
        "da_reg_jump_spike_top1pct",
        "da_reg_jump_spike_top0_5pct",
        "da_reg_body_flag",
        "da_price_actual",
        "rt_price_actual",
        "wind_actual_mw",
        "da_load_forecast_west",
        "da_henry_hub_price",
        "rt_load_forecast_west",
        "rt_henry_hub_price",
        "month_sin",
        "month_cos",
        "hour_sin",
        "hour_cos",
        "dow_sin",
        "dow_cos",
    ]
    keep_cols += [f"da_reg_lag{lag}" for lag in HOURLY_GBR_LAG_STEPS]
    keep_cols += [f"da_reg_rollmean{window}" for window in HOURLY_GBR_ROLL_WINDOWS]

    optional_keep_cols = [
        "iso_west_load_forecast_mw",
        "iso_nyiso_load_forecast_mw",
        "da_load_forecast_gap_west",
        "da_regulation_price_west",
        "da_spin10_price_west",
        "da_nspin10_price_west",
        "da_spin30_price_west",
        "da_reg_requirement_mw",
        "da_constraint_count",
        "da_constraint_shadowcost_sum",
        "da_constraint_shadowcost_abs_max",
        "da_ace_mw",
        "da_ace_abs",
    ]
    keep_cols += [col for col in optional_keep_cols if col in df.columns]

    required_cols = [
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
        "da_reg_actual",
        "da_price_actual",
        "rt_price_actual",
        "wind_actual_mw",
        "da_load_forecast_west",
        "da_henry_hub_price",
        "rt_load_forecast_west",
        "rt_henry_hub_price",
    ]
    out = df[keep_cols].dropna(subset=required_cols).reset_index(drop=True)

    meta = {
        "description": "설 논문식 DA_REG HistGBR 가공데이터. DA/RT/풍력과 보조서비스·ACE·부하예측 보조변수를 포함.",
        "source_csv": str(source_csv),
        "feature_groups": {
            "exogenous": [
                "da_price_actual",
                "rt_price_actual",
                "wind_actual_mw",
                "da_load_forecast_west",
                "da_henry_hub_price",
                "rt_load_forecast_west",
                "rt_henry_hub_price",
            ] + [col for col in optional_keep_cols if col in out.columns],
            "time_cyc": ["month_sin", "month_cos", "hour_sin", "hour_cos", "dow_sin", "dow_cos"],
            "lags": [f"da_reg_lag{lag}" for lag in HOURLY_GBR_LAG_STEPS],
            "rolling": [f"da_reg_rollmean{window}" for window in HOURLY_GBR_ROLL_WINDOWS],
        },
        "jump_labeling": jump_meta,
        "train_start": train_start,
        "train_end": train_end,
        "valid_start": valid_start,
        "valid_end": valid_end,
        "test_start": test_start,
        "test_end": test_end,
        "columns": keep_cols,
    }
    return write_split_outputs(out, out_root, ALL_CSV, TRAIN_CSV, None, TEST_CSV, META_JSON, meta)


def main() -> None:
    roots = resolve_project_roots(__file__)
    ap = argparse.ArgumentParser(description="Build richer DA REG preprocessing dataset for HistGBR.")
    ap.add_argument("--source-csv", default=str(roots["input_root"] / "05_DA_REG가격_학습데이터.csv"))
    ap.add_argument("--da-source-csv", default=str(roots["input_root"] / "04_DA에너지_서부_학습데이터.csv"))
    ap.add_argument("--rt-source-csv", default=str(roots["input_root"] / "02_RT에너지_서부_학습데이터.csv"))
    ap.add_argument("--wind-source-csv", default=str(roots["input_root"] / "03_풍력발전_서부_학습데이터.csv"))
    ap.add_argument("--out-root", default=str(roots["prep_root"]))
    ap.add_argument("--iso-load-source-csv", default=str(roots["input_root"] / "07_NYISO_ISO부하예측_WEST.csv"))
    ap.add_argument("--da-as-source-csv", default=str(roots["input_root"] / "08_NYISO_DA_보조서비스가격_WEST.csv"))
    ap.add_argument("--reg-requirement-source-csv", default=str(roots["input_root"] / "10_NYISO_Regulation_Requirement_NYCA.csv"))
    ap.add_argument("--system-summary-source-csv", default=str(roots["input_root"] / "13_NYISO_제약및발전기상태_요약.csv"))
    ap.add_argument("--ace-source-csv", default=str(roots["input_root"] / "12_NYISO_ACE_NYCA.csv"))
    ap.add_argument("--train-start", default=DEFAULT_TRAIN_START)
    ap.add_argument("--train-end", default=DEFAULT_TRAIN_END)
    ap.add_argument("--valid-start", default=DEFAULT_VALID_START)
    ap.add_argument("--valid-end", default=DEFAULT_VALID_END)
    ap.add_argument("--test-start", default=DEFAULT_TEST_START)
    ap.add_argument("--test-end", default=DEFAULT_TEST_END)
    args = ap.parse_args()

    summary = build_dataset(
        Path(args.source_csv),
        Path(args.da_source_csv),
        Path(args.rt_source_csv),
        Path(args.wind_source_csv),
        Path(args.out_root),
        iso_load_source_csv=Path(args.iso_load_source_csv),
        da_as_source_csv=Path(args.da_as_source_csv),
        reg_requirement_source_csv=Path(args.reg_requirement_source_csv),
        system_summary_source_csv=Path(args.system_summary_source_csv),
        ace_source_csv=Path(args.ace_source_csv),
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
