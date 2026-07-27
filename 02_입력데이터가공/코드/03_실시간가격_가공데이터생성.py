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
RT_GBR_LAG_STEPS = _COMMON.RT_GBR_LAG_STEPS
RT_GBR_ROLL_WINDOWS = _COMMON.RT_GBR_ROLL_WINDOWS
aggregate_hourly_abs_stats = _COMMON.aggregate_hourly_abs_stats
add_cyclic_calendar_features = _COMMON.add_cyclic_calendar_features
add_jump_label_features = _COMMON.add_jump_label_features
add_split_column = _COMMON.add_split_column
attach_issue_history_features = _COMMON.attach_issue_history_features
attach_day_ahead_issue_columns = _COMMON.attach_day_ahead_issue_columns
load_optional_timestamped_features = _COMMON.load_optional_timestamped_features
load_rt_5m_series_source = _COMMON.load_rt_5m_series_source
merge_optional_features = _COMMON.merge_optional_features
overwrite_with_issue_snapshot_features = _COMMON.overwrite_with_issue_snapshot_features
resolve_project_roots = _COMMON.resolve_project_roots
tokenized_colname = _COMMON.tokenized_colname
write_split_outputs = _COMMON.write_split_outputs

ALL_CSV = "09_실시간가격_가공데이터_전체.csv"
TRAIN_CSV = "10_실시간가격_가공데이터_학습기간.csv"
TEST_CSV = "11_실시간가격_가공데이터_테스트기간.csv"
META_JSON = "12_실시간가격_가공데이터_메타정보.json"


def _attach_optional_rt_features(
    df: pd.DataFrame,
    *,
    iso_load_source_csv: Path | None,
    rt_as_source_csv: Path | None,
    reg_movement_source_csv: Path | None,
    ace_source_csv: Path | None,
    fuelmix_source_csv: Path | None,
    system_summary_source_csv: Path | None,
) -> pd.DataFrame:
    out = df.copy()
    out["timestamp_hour_utc"] = out["timestamp_utc"].dt.floor("h")

    iso_load = load_optional_timestamped_features(
        iso_load_source_csv,
        keep_cols=["west_load_forecast_mw", "nyiso_load_forecast_mw"],
        rename_map={
            "west_load_forecast_mw": "iso_west_load_forecast_mw",
            "nyiso_load_forecast_mw": "iso_nyiso_load_forecast_mw",
        },
        aggregate_freq="h",
    )
    if iso_load is not None:
        iso_load = iso_load.rename(columns={"timestamp_utc": "timestamp_hour_utc"})
        out = merge_optional_features(out, iso_load, on="timestamp_hour_utc")
        if "iso_west_load_forecast_mw" in out.columns:
            out["rt_load_forecast_gap_west"] = out["rt_load_forecast_west"] - out["iso_west_load_forecast_mw"]

    rt_as = load_optional_timestamped_features(
        rt_as_source_csv,
        keep_cols=["regulation_price", "spin10_price", "nspin10_price", "spin30_price"],
        rename_map={
            "regulation_price": "rt_regulation_price_west",
            "spin10_price": "rt_spin10_price_west",
            "nspin10_price": "rt_nspin10_price_west",
            "spin30_price": "rt_spin30_price_west",
        },
        aggregate_freq="h",
    )
    if rt_as is not None:
        rt_as = rt_as.rename(columns={"timestamp_utc": "timestamp_hour_utc"})
        out = merge_optional_features(out, rt_as, on="timestamp_hour_utc")

    reg_movement = load_optional_timestamped_features(
        reg_movement_source_csv,
        keep_cols=["regulation_movement_mw"],
    )
    reg_movement_hourly = aggregate_hourly_abs_stats(reg_movement, "regulation_movement_mw", "rt_reg_movement")
    if reg_movement_hourly is not None:
        reg_movement_hourly = reg_movement_hourly.rename(columns={"timestamp_utc": "timestamp_hour_utc"})
        out = merge_optional_features(out, reg_movement_hourly, on="timestamp_hour_utc")

    ace = load_optional_timestamped_features(
        ace_source_csv,
        keep_cols=["ace_mw"],
        rename_map={"ace_mw": "rt_ace_mw"},
    )
    if ace is not None:
        out = merge_optional_features(out, ace, on="timestamp_utc")
        out["rt_ace_abs"] = out["rt_ace_mw"].abs()

    system_summary = load_optional_timestamped_features(
        system_summary_source_csv,
        keep_cols=["constraint_count", "constraint_shadowcost_sum", "constraint_shadowcost_abs_max"],
        rename_map={
            "constraint_count": "rt_constraint_count",
            "constraint_shadowcost_sum": "rt_constraint_shadowcost_sum",
            "constraint_shadowcost_abs_max": "rt_constraint_shadowcost_abs_max",
        },
        aggregate_freq="h",
    )
    if system_summary is not None:
        system_summary = system_summary.rename(columns={"timestamp_utc": "timestamp_hour_utc"})
        out = merge_optional_features(out, system_summary, on="timestamp_hour_utc")

    if fuelmix_source_csv is not None and fuelmix_source_csv.exists():
        raw_fuelmix = pd.read_csv(fuelmix_source_csv, encoding="utf-8-sig")
        required = {"timestamp_utc", "fuel_category", "gen_mw"}
        if required.issubset(raw_fuelmix.columns):
            raw_fuelmix["timestamp_utc"] = pd.to_datetime(raw_fuelmix["timestamp_utc"], utc=True, errors="coerce")
            raw_fuelmix["fuel_category"] = raw_fuelmix["fuel_category"].astype(str).str.strip()
            raw_fuelmix["fuel_category_norm"] = raw_fuelmix["fuel_category"].map(tokenized_colname)
            raw_fuelmix["gen_mw"] = pd.to_numeric(raw_fuelmix["gen_mw"], errors="coerce")
            fuelmix = raw_fuelmix.dropna(subset=["timestamp_utc", "gen_mw"]).copy()
            fuelmix = fuelmix.loc[fuelmix["fuel_category_norm"].ne("")].copy()
            if not fuelmix.empty:
                fuelmix["timestamp_hour_utc"] = fuelmix["timestamp_utc"].dt.floor("h")
                fuelmix_hour = (
                    fuelmix.pivot_table(
                        index="timestamp_hour_utc",
                        columns="fuel_category_norm",
                        values="gen_mw",
                        aggfunc="mean",
                    )
                    .reset_index()
                )
                fuel_cols = [col for col in fuelmix_hour.columns if col != "timestamp_hour_utc"]
                if fuel_cols:
                    total = fuelmix_hour[fuel_cols].sum(axis=1).replace(0, pd.NA)
                    gas_cols = [col for col in fuel_cols if "gas" in col]
                    renewable_cols = [
                        col
                        for col in fuel_cols
                        if any(token in col for token in ("wind", "solar", "hydro", "renew"))
                    ]
                    share_cols = ["timestamp_hour_utc"]
                    if gas_cols:
                        fuelmix_hour["rt_fuelmix_gas_share"] = (fuelmix_hour[gas_cols].sum(axis=1) / total).clip(lower=0, upper=1)
                        share_cols.append("rt_fuelmix_gas_share")
                    if renewable_cols:
                        fuelmix_hour["rt_fuelmix_renewable_share"] = (fuelmix_hour[renewable_cols].sum(axis=1) / total).clip(lower=0, upper=1)
                        share_cols.append("rt_fuelmix_renewable_share")
                    if len(share_cols) > 1:
                        out = merge_optional_features(out, fuelmix_hour[share_cols], on="timestamp_hour_utc")

    return out


def build_dataset(
    source_csv: Path,
    out_root: Path,
    *,
    iso_load_source_csv: Path | None = None,
    rt_as_source_csv: Path | None = None,
    reg_movement_source_csv: Path | None = None,
    ace_source_csv: Path | None = None,
    fuelmix_source_csv: Path | None = None,
    system_summary_source_csv: Path | None = None,
    train_start: str = DEFAULT_TRAIN_START,
    train_end: str = DEFAULT_TRAIN_END,
    valid_start: str = DEFAULT_VALID_START,
    valid_end: str = DEFAULT_VALID_END,
    test_start: str = DEFAULT_TEST_START,
    test_end: str = DEFAULT_TEST_END,
) -> dict[str, object]:
    df = load_rt_5m_series_source(source_csv)
    df["rt_price_actual"] = pd.to_numeric(df["RT_LBMP"], errors="coerce")
    df["rt_dam_lbmp"] = pd.to_numeric(df.get("DAM_LBMP"), errors="coerce")
    df["rt_load_mw"] = pd.to_numeric(df.get("Load_MW"), errors="coerce")
    df["rt_load_forecast_west"] = pd.to_numeric(df.get("Load_Forecast_West"), errors="coerce")
    df["rt_wind_gen_act_norm"] = pd.to_numeric(df.get("Wind_Gen_Act_Norm"), errors="coerce")
    df["rt_henry_hub_price"] = pd.to_numeric(df.get("Henry_Hub_Price"), errors="coerce")

    df = _attach_optional_rt_features(
        df,
        iso_load_source_csv=iso_load_source_csv,
        rt_as_source_csv=rt_as_source_csv,
        reg_movement_source_csv=reg_movement_source_csv,
        ace_source_csv=ace_source_csv,
        fuelmix_source_csv=fuelmix_source_csv,
        system_summary_source_csv=system_summary_source_csv,
    )

    df = attach_day_ahead_issue_columns(df)
    df = add_split_column(df, train_start, train_end, valid_start, valid_end, test_start, test_end)
    df, jump_meta = add_jump_label_features(df, target_col="rt_price_actual", prefix="rt_price")
    df = overwrite_with_issue_snapshot_features(
        df,
        value_cols=[
            "rt_dam_lbmp",
            "rt_load_mw",
            "rt_wind_gen_act_norm",
            "rt_henry_hub_price",
            "rt_regulation_price_west",
            "rt_spin10_price_west",
            "rt_nspin10_price_west",
            "rt_spin30_price_west",
            "rt_reg_movement_mean",
            "rt_reg_movement_abs_mean",
            "rt_reg_movement_abs_max",
            "rt_ace_mw",
            "rt_ace_abs",
            "rt_fuelmix_gas_share",
            "rt_fuelmix_renewable_share",
            "rt_constraint_count",
            "rt_constraint_shadowcost_sum",
            "rt_constraint_shadowcost_abs_max",
        ],
    )
    df = add_cyclic_calendar_features(df, include_slot=True)
    df = attach_issue_history_features(
        df,
        source_col="rt_price_actual",
        prefix="rt_price",
        lag_steps=RT_GBR_LAG_STEPS,
        roll_windows=RT_GBR_ROLL_WINDOWS,
    )

    keep_cols = [
        "timestamp_utc",
        "timestamp_et",
        "issue_timestamp_utc",
        "issue_timestamp_et",
        "issue_operating_date_et",
        "operating_date_et",
        "hour_et",
        "minute_et",
        "slot_5m_et",
        "month",
        "weekday",
        "year",
        "split",
        "rt_price_actual",
        "rt_price_delta",
        "rt_price_abs_delta",
        "rt_price_jump_robust_z",
        "rt_price_jump_spike_mad_upper",
        "rt_price_jump_spike_top1pct",
        "rt_price_jump_spike_top0_5pct",
        "rt_price_body_flag",
        "rt_dam_lbmp",
        "rt_load_mw",
        "rt_load_forecast_west",
        "rt_wind_gen_act_norm",
        "rt_henry_hub_price",
        "month_sin",
        "month_cos",
        "hour_sin",
        "hour_cos",
        "dow_sin",
        "dow_cos",
        "slot_sin",
        "slot_cos",
    ]
    keep_cols += [f"rt_price_lag{lag}" for lag in RT_GBR_LAG_STEPS]
    keep_cols += [f"rt_price_rollmean{window}" for window in RT_GBR_ROLL_WINDOWS]

    optional_keep_cols = [
        "iso_west_load_forecast_mw",
        "iso_nyiso_load_forecast_mw",
        "rt_load_forecast_gap_west",
        "rt_regulation_price_west",
        "rt_spin10_price_west",
        "rt_nspin10_price_west",
        "rt_spin30_price_west",
        "rt_reg_movement_mean",
        "rt_reg_movement_abs_mean",
        "rt_reg_movement_abs_max",
        "rt_ace_mw",
        "rt_ace_abs",
        "rt_fuelmix_gas_share",
        "rt_fuelmix_renewable_share",
        "rt_constraint_count",
        "rt_constraint_shadowcost_sum",
        "rt_constraint_shadowcost_abs_max",
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
        "minute_et",
        "slot_5m_et",
        "month",
        "weekday",
        "year",
        "split",
        "rt_price_actual",
        "rt_dam_lbmp",
        "rt_load_mw",
        "rt_load_forecast_west",
        "rt_wind_gen_act_norm",
        "rt_henry_hub_price",
    ]
    out = df[keep_cols].dropna(subset=required_cols).reset_index(drop=True)

    meta = {
        "description": "설 논문식 실시간가격 HistGBR 가공데이터. 시장/수요/풍력/보조서비스/시스템 상태 변수를 포함.",
        "source_csv": str(source_csv),
        "target_col": "rt_price_actual",
        "feature_groups": {
            "exogenous": [
                "rt_dam_lbmp",
                "rt_load_mw",
                "rt_load_forecast_west",
                "rt_wind_gen_act_norm",
                "rt_henry_hub_price",
            ] + [col for col in optional_keep_cols if col in out.columns],
            "time_cyc": [
                "month_sin",
                "month_cos",
                "hour_sin",
                "hour_cos",
                "dow_sin",
                "dow_cos",
                "slot_sin",
                "slot_cos",
            ],
            "lags": [f"rt_price_lag{lag}" for lag in RT_GBR_LAG_STEPS],
            "rolling": [f"rt_price_rollmean{window}" for window in RT_GBR_ROLL_WINDOWS],
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
    ap = argparse.ArgumentParser(description="Build richer RT price preprocessing dataset for HistGBR.")
    ap.add_argument("--source-csv", default=str(roots["input_root"] / "02_RT에너지_서부_학습데이터.csv"))
    ap.add_argument("--out-root", default=str(roots["prep_root"]))
    ap.add_argument("--iso-load-source-csv", default=str(roots["input_root"] / "07_NYISO_ISO부하예측_WEST.csv"))
    ap.add_argument("--rt-as-source-csv", default=str(roots["input_root"] / "09_NYISO_RT_보조서비스가격_WEST.csv"))
    ap.add_argument("--reg-movement-source-csv", default=str(roots["input_root"] / "11_NYISO_Regulation_Movement_NYCA.csv"))
    ap.add_argument("--ace-source-csv", default=str(roots["input_root"] / "12_NYISO_ACE_NYCA.csv"))
    ap.add_argument("--fuelmix-source-csv", default=str(roots["input_root"] / "14_NYISO_실시간연료믹스_NYCA.csv"))
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
        rt_as_source_csv=Path(args.rt_as_source_csv),
        reg_movement_source_csv=Path(args.reg_movement_source_csv),
        ace_source_csv=Path(args.ace_source_csv),
        fuelmix_source_csv=Path(args.fuelmix_source_csv),
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
