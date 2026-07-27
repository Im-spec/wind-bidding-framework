#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEBUG_INTERNAL_DIRNAME = "90_디버깅용_점예측내부산출물"

MODEL_PARAMS = {
    "learning_rate": 0.05,
    "max_depth": 8,
    "max_iter": 700,
    "min_samples_leaf": 32,
    "l2_regularization": 0.1,
    "early_stopping": True,
    "validation_fraction": 0.1,
    "n_iter_no_change": 20,
    "random_state": 42,
}

SERIES_SPECS: dict[str, dict[str, object]] = {
    "wind": {
        "prefix": "wind",
        "dataset_name": "01_?띾젰諛쒖쟾_媛怨듬뜲?댄꽣_?꾩껜.csv",
        "target_col": "wind_actual_mw",
        "lag1_col": "wind_lag1",
        "body_flag_col": "wind_body_flag",
        "feature_cols": [
            "v100",
            "v10",
            "sp",
            "month_sin",
            "month_cos",
            "hour_sin",
            "hour_cos",
            "dow_sin",
            "dow_cos",
            "wind_lag1",
            "wind_lag2",
            "wind_lag3",
            "wind_lag6",
            "wind_lag12",
            "wind_lag24",
            "wind_lag48",
            "wind_lag168",
            "wind_rollmean6",
            "wind_rollmean24",
            "wind_rollmean72",
        ],
        "base_cols": [
            "timestamp_utc",
            "timestamp_et",
            "issue_timestamp_utc",
            "issue_timestamp_et",
            "issue_operating_date_et",
            "operating_date_et",
            "hour_et",
            "month",
            "weekday",
            "split",
            "wind_actual_mw",
            "wind_delta",
            "wind_abs_delta",
            "wind_jump_spike_top1pct",
            "wind_jump_spike_top0_5pct",
            "wind_body_flag",
            "wind_actual_ratio",
            "capacity_mw",
        ],
        "export_kind": "wind_hourly",
        "train_result_name": "01_?숈뒿湲곌컙_?띾젰諛쒖쟾_?덉륫寃곌낵.csv",
        "valid_result_name": "16_寃利앷린媛??띾젰諛쒖쟾_?덉륫寃곌낵.csv",
        "test_result_name": "02_?뚯뒪?멸린媛??띾젰諛쒖쟾_?덉륫寃곌낵.csv",
        "summary_csv_name": "13_?띾젰諛쒖쟾_?먯삁痢〓え???좏슚?깆슂??csv",
        "model_label": "?띾젰諛쒖쟾",
        "selected_model_key": "histgbr",
        "model_desc": "HistGradientBoostingRegressor",
        "summary_note": "???쇰Ц ?띾젰 GBR 諛⑹떇??richer feature瑜??꾩옱 援ъ“濡??댁떇",
    },
    "da_energy": {
        "prefix": "da_energy",
        "dataset_name": "05_?섎（?꾧?寃?媛怨듬뜲?댄꽣_?꾩껜.csv",
        "target_col": "da_price_actual",
        "lag1_col": "da_price_lag1",
        "body_flag_col": "da_price_body_flag",
        "feature_cols": [
            "da_load_mw",
            "da_lbmp_mcc",
            "da_lbmp_mcl",
            "da_henry_hub_price",
            "da_load_forecast_west",
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
            "month_sin",
            "month_cos",
            "hour_sin",
            "hour_cos",
            "dow_sin",
            "dow_cos",
            "da_price_lag1",
            "da_price_lag2",
            "da_price_lag3",
            "da_price_lag6",
            "da_price_lag12",
            "da_price_lag24",
            "da_price_lag48",
            "da_price_lag168",
            "da_price_rollmean6",
            "da_price_rollmean24",
            "da_price_rollmean72",
        ],
        "base_cols": [
            "timestamp_utc",
            "timestamp_et",
            "issue_timestamp_utc",
            "issue_timestamp_et",
            "issue_operating_date_et",
            "operating_date_et",
            "hour_et",
            "month",
            "weekday",
            "split",
            "da_price_actual",
            "da_price_delta",
            "da_price_abs_delta",
            "da_price_jump_spike_top1pct",
            "da_price_jump_spike_top0_5pct",
            "da_price_body_flag",
        ],
        "export_kind": "hourly",
        "train_result_name": "03_?숈뒿湲곌컙_DA?먮꼫吏_?덉륫寃곌낵.csv",
        "valid_result_name": "17_寃利앷린媛?DA?먮꼫吏_?덉륫寃곌낵.csv",
        "test_result_name": "04_?뚯뒪?멸린媛?DA?먮꼫吏_?덉륫寃곌낵.csv",
        "summary_csv_name": "14_DA?먮꼫吏_?먯삁痢〓え???좏슚?깆슂??csv",
        "model_label": "DA?먮꼫吏",
        "selected_model_key": "histgbr",
        "model_desc": "HistGradientBoostingRegressor",
        "summary_note": "???쇰Ц??richer feature瑜??ъ슜??direct point forecast",
    },
    "rt_energy": {
        "prefix": "rt_energy",
        "dataset_name": "09_?ㅼ떆媛꾧?寃?媛怨듬뜲?댄꽣_?꾩껜.csv",
        "target_col": "rt_price_actual",
        "lag1_col": "rt_price_lag1",
        "body_flag_col": "rt_price_body_flag",
        "feature_cols": [
            "rt_dam_lbmp",
            "rt_load_mw",
            "rt_load_forecast_west",
            "rt_wind_gen_act_norm",
            "rt_henry_hub_price",
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
            "month_sin",
            "month_cos",
            "hour_sin",
            "hour_cos",
            "dow_sin",
            "dow_cos",
            "slot_sin",
            "slot_cos",
            "rt_price_lag1",
            "rt_price_lag3",
            "rt_price_lag6",
            "rt_price_lag12",
            "rt_price_lag24",
            "rt_price_lag288",
            "rt_price_rollmean12",
            "rt_price_rollmean72",
            "rt_price_rollmean288",
        ],
        "base_cols": [
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
            "split",
            "rt_price_actual",
            "rt_price_delta",
            "rt_price_abs_delta",
            "rt_price_jump_spike_top1pct",
            "rt_price_jump_spike_top0_5pct",
            "rt_price_body_flag",
        ],
        "export_kind": "rt_5m",
        "train_result_name": "05_?숈뒿湲곌컙_RT?먮꼫吏_?덉륫寃곌낵.csv",
        "valid_result_name": "18_寃利앷린媛?RT?먮꼫吏_?덉륫寃곌낵.csv",
        "test_result_name": "06_?뚯뒪?멸린媛?RT?먮꼫吏_?덉륫寃곌낵.csv",
        "summary_csv_name": "15_RT?먮꼫吏_?먯삁痢〓え???좏슚?깆슂??csv",
        "model_label": "RT?먮꼫吏",
        "selected_model_key": "histgbr",
        "model_desc": "HistGradientBoostingRegressor",
        "summary_note": "???쇰Ц??richer feature瑜??ъ슜??direct point forecast",
    },
    "da_reg": {
        "prefix": "da_reg",
        "dataset_name": "13_DA_REG媛寃?媛怨듬뜲?댄꽣_?꾩껜.csv",
        "target_col": "da_reg_actual",
        "lag1_col": "da_reg_lag1",
        "body_flag_col": "da_reg_body_flag",
        "feature_cols": [
            "da_price_actual",
            "rt_price_actual",
            "wind_actual_mw",
            "da_load_forecast_west",
            "da_henry_hub_price",
            "rt_load_forecast_west",
            "rt_henry_hub_price",
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
            "month_sin",
            "month_cos",
            "hour_sin",
            "hour_cos",
            "dow_sin",
            "dow_cos",
            "da_reg_lag1",
            "da_reg_lag2",
            "da_reg_lag3",
            "da_reg_lag6",
            "da_reg_lag12",
            "da_reg_lag24",
            "da_reg_lag48",
            "da_reg_lag168",
            "da_reg_rollmean6",
            "da_reg_rollmean24",
            "da_reg_rollmean72",
        ],
        "base_cols": [
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
            "da_reg_jump_spike_top1pct",
            "da_reg_jump_spike_top0_5pct",
            "da_reg_body_flag",
        ],
        "export_kind": "hourly",
        "train_result_name": "07_?숈뒿湲곌컙_?섎（??REG媛寃??덉륫寃곌낵.csv",
        "valid_result_name": "19_검증기간_하루전_REG가격_예측결과.csv",
        "test_result_name": "08_?뚯뒪?멸린媛??섎（??REG媛寃??덉륫寃곌낵.csv",
        "summary_csv_name": "11_DA_REG媛寃??먯삁痢〓え???좏슚?깆슂??csv",
        "model_label": "하루전 REG가격",
        "selected_model_key": "histgbr",
        "model_desc": "HistGradientBoostingRegressor",
        "summary_note": "DA_REG瑜??낅┰ ?쒓퀎?대줈 遺꾨━??direct point forecast ?곸슜",
    },
    "rt_reg": {
        "prefix": "rt_reg",
        "dataset_name": "17_RT_REG媛寃?媛怨듬뜲?댄꽣_?꾩껜.csv",
        "target_col": "rt_reg_actual",
        "lag1_col": "rt_reg_lag1",
        "body_flag_col": "rt_reg_body_flag",
        "feature_cols": [
            "da_price_actual",
            "rt_price_actual",
            "wind_actual_mw",
            "da_load_forecast_west",
            "da_henry_hub_price",
            "rt_load_forecast_west",
            "rt_henry_hub_price",
            "iso_west_load_forecast_mw",
            "iso_nyiso_load_forecast_mw",
            "rt_load_forecast_gap_west",
            "rt_regulation_price_west",
            "rt_spin10_price_west",
            "rt_nspin10_price_west",
            "rt_spin30_price_west",
            "rt_reg_requirement_mw",
            "rt_reg_movement_mean",
            "rt_reg_movement_abs_mean",
            "rt_reg_movement_abs_max",
            "rt_constraint_count",
            "rt_constraint_shadowcost_sum",
            "rt_constraint_shadowcost_abs_max",
            "rt_ace_mw",
            "rt_ace_abs",
            "month_sin",
            "month_cos",
            "hour_sin",
            "hour_cos",
            "dow_sin",
            "dow_cos",
            "rt_reg_lag1",
            "rt_reg_lag2",
            "rt_reg_lag3",
            "rt_reg_lag6",
            "rt_reg_lag12",
            "rt_reg_lag24",
            "rt_reg_lag48",
            "rt_reg_lag168",
            "rt_reg_rollmean6",
            "rt_reg_rollmean24",
            "rt_reg_rollmean72",
        ],
        "base_cols": [
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
            "rt_reg_actual",
            "rt_reg_delta",
            "rt_reg_abs_delta",
            "rt_reg_jump_spike_top1pct",
            "rt_reg_jump_spike_top0_5pct",
            "rt_reg_body_flag",
        ],
        "export_kind": "hourly",
        "train_result_name": "09_?숈뒿湲곌컙_?ㅼ떆媛?REG媛寃??덉륫寃곌낵.csv",
        "valid_result_name": "20_검증기간_실시간_REG가격_예측결과.csv",
        "test_result_name": "10_?뚯뒪?멸린媛??ㅼ떆媛?REG媛寃??덉륫寃곌낵.csv",
        "summary_csv_name": "12_RT_REG媛寃??먯삁痢〓え???좏슚?깆슂??csv",
        "model_label": "실시간 REG가격",
        "selected_model_key": "histgbr",
        "model_desc": "HistGradientBoostingRegressor",
        "summary_note": "RT_REG瑜??낅┰ ?쒓퀎?대줈 遺꾨━??direct point forecast ?곸슜",
    },
}

SERIES_SPECS["wind"].update(
    {
        "dataset_name": "01_풍력발전_가공데이터_전체.csv",
        "train_result_name": "01_학습기간_풍력발전_예측결과.csv",
        "valid_result_name": "16_검증기간_풍력발전_예측결과.csv",
        "test_result_name": "02_테스트기간_풍력발전_예측결과.csv",
        "summary_csv_name": "13_풍력발전_점예측모델_유효성요약.csv",
        "model_label": "풍력발전",
    }
)
SERIES_SPECS["da_energy"].update(
    {
        "dataset_name": "05_하루전가격_가공데이터_전체.csv",
        "train_result_name": "03_학습기간_DA에너지_예측결과.csv",
        "valid_result_name": "17_검증기간_DA에너지_예측결과.csv",
        "test_result_name": "04_테스트기간_DA에너지_예측결과.csv",
        "summary_csv_name": "14_DA에너지_점예측모델_유효성요약.csv",
        "model_label": "DA에너지",
    }
)
SERIES_SPECS["rt_energy"].update(
    {
        "dataset_name": "09_실시간가격_가공데이터_전체.csv",
        "train_result_name": "05_학습기간_RT에너지_예측결과.csv",
        "valid_result_name": "18_검증기간_RT에너지_예측결과.csv",
        "test_result_name": "06_테스트기간_RT에너지_예측결과.csv",
        "summary_csv_name": "15_RT에너지_점예측모델_유효성요약.csv",
        "model_label": "RT에너지",
    }
)
SERIES_SPECS["da_reg"].update(
    {
        "dataset_name": "13_DA_REG가격_가공데이터_전체.csv",
        "train_result_name": "07_학습기간_하루전_REG가격_예측결과.csv",
        "valid_result_name": "19_검증기간_하루전_REG가격_예측결과.csv",
        "test_result_name": "08_테스트기간_하루전_REG가격_예측결과.csv",
        "summary_csv_name": "11_DA_REG가격_점예측모델_유효성요약.csv",
        "model_label": "하루전 REG가격",
    }
)
SERIES_SPECS["rt_reg"].update(
    {
        "dataset_name": "17_RT_REG가격_가공데이터_전체.csv",
        "train_result_name": "09_학습기간_실시간_REG가격_예측결과.csv",
        "valid_result_name": "20_검증기간_실시간_REG가격_예측결과.csv",
        "test_result_name": "10_테스트기간_실시간_REG가격_예측결과.csv",
        "summary_csv_name": "12_RT_REG가격_점예측모델_유효성요약.csv",
        "model_label": "실시간 REG가격",
    }
)


def resolve_project_roots(_: str | Path | None = None) -> dict[str, Path]:
    point_root = PROJECT_ROOT / "03_점예측"
    return {
        "project_root": PROJECT_ROOT,
        "prep_root": PROJECT_ROOT / "02_입력데이터가공" / "가공데이터",
        "point_root": point_root,
        "internal_root": point_root / DEBUG_INTERNAL_DIRNAME,
        "result_root": point_root / "점예측결과",
    }


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def dump_json(path: Path, payload: dict[str, object]) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def selected_model_key(spec: dict[str, object]) -> str:
    return "histgbr"


def selected_pred_col(spec: dict[str, object]) -> str:
    return f"{spec['prefix']}_pred"


def selected_metric_block(spec: dict[str, object], split_name: str) -> str:
    return split_name


INTERNAL_OUTPUT_DIRS = {
    "wind": "01_풍력발전",
    "da_energy": "02_DA에너지",
    "rt_energy": "03_RT에너지",
    "da_reg": "04_DA_REG가격",
    "rt_reg": "05_RT_REG가격",
}


def output_names(prefix: str) -> dict[str, str]:
    series_dir = INTERNAL_OUTPUT_DIRS.get(prefix, prefix)
    return {
        "all_pred_csv": str(Path(series_dir) / "01_원시예측결과_전체.csv"),
        "train_pred_csv": str(Path(series_dir) / "02_원시예측결과_학습기간.csv"),
        "valid_pred_csv": str(Path(series_dir) / "03_원시예측결과_검증기간.csv"),
        "test_pred_csv": str(Path(series_dir) / "04_원시예측결과_테스트기간.csv"),
        "metrics_csv": str(Path(series_dir) / "05_원시평가지표.csv"),
        "metrics_json": str(Path(series_dir) / "06_원시평가지표.json"),
        "coef_csv": str(Path(series_dir) / "07_특성요약.csv"),
        "model_joblib": str(Path(series_dir) / "08_모형.joblib"),
        "meta_json": str(Path(series_dir) / "09_메타정보.json"),
    }


def load_series_dataset(dataset_csv: Path, spec: dict[str, object]) -> pd.DataFrame:
    df = pd.read_csv(dataset_csv, encoding="utf-8-sig")
    if "timestamp_utc" in df.columns:
        df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True, errors="coerce")
    required = set(spec["base_cols"]) | {str(spec["target_col"]), str(spec["lag1_col"])}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Dataset is missing columns: {missing}")
    return df


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    corr = float(np.corrcoef(y_true, y_pred)[0, 1]) if len(y_true) > 1 else float("nan")
    bias = float(np.mean(y_true - y_pred))
    return {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "corr": corr,
        "bias_true_minus_pred": bias,
    }


def train_direct_model(
    df: pd.DataFrame,
    spec: dict[str, object],
) -> tuple[pd.DataFrame, dict[str, object], pd.DataFrame, dict[str, object]]:
    target_col = str(spec["target_col"])
    body_flag_col = str(spec.get("body_flag_col", "")) or None
    requested_feature_cols = list(spec["feature_cols"])
    feature_cols = [col for col in requested_feature_cols if col in df.columns]
    if not feature_cols:
        raise ValueError(f"No usable feature columns remain for {spec['prefix']}")
    base_cols = list(spec["base_cols"])
    pred_col = f"{spec['prefix']}_pred"

    usable = df[df["split"].isin(["train", "valid", "test"])].copy()
    usable = usable.dropna(subset=feature_cols + [target_col]).reset_index(drop=True)
    train_df = usable[usable["split"] == "train"].copy()
    valid_df = usable[usable["split"] == "valid"].copy()
    test_df = usable[usable["split"] == "test"].copy()
    if train_df.empty or valid_df.empty or test_df.empty:
        raise ValueError(f"Train/valid/test rows are empty for {spec['prefix']}")

    fit_train_df = train_df.copy()
    training_subset = "all"
    if body_flag_col and body_flag_col in train_df.columns:
        body_train_df = train_df[train_df[body_flag_col] == 1].copy()
        if not body_train_df.empty:
            fit_train_df = body_train_df
            training_subset = "body_only"

    X_train = fit_train_df[feature_cols].astype(float)
    y_train = fit_train_df[target_col].astype(float)
    X_all = usable[feature_cols].astype(float)

    model = HistGradientBoostingRegressor(**MODEL_PARAMS)
    model.fit(X_train, y_train)
    pred_all = model.predict(X_all)

    pred_df = usable[base_cols].copy()
    pred_df[pred_col] = pred_all.astype(float)
    train_pred = pred_df[pred_df["split"] == "train"].copy()
    valid_pred = pred_df[pred_df["split"] == "valid"].copy()
    test_pred = pred_df[pred_df["split"] == "test"].copy()

    metrics = {
        "series": spec["prefix"],
        "model_family": "HistGradientBoostingRegressor",
        "feature_count": len(feature_cols),
        "requested_feature_count": len(requested_feature_cols),
        "features": feature_cols,
        "missing_optional_features": [col for col in requested_feature_cols if col not in df.columns],
        "model_params": dict(MODEL_PARAMS),
        "training_subset": training_subset,
        "train": _metrics(train_pred[target_col].to_numpy(float), train_pred[pred_col].to_numpy(float)),
        "valid": _metrics(valid_pred[target_col].to_numpy(float), valid_pred[pred_col].to_numpy(float)),
        "test": _metrics(test_pred[target_col].to_numpy(float), test_pred[pred_col].to_numpy(float)),
        "n_train": int(len(train_pred)),
        "n_train_total": int(len(train_pred)),
        "n_train_fit": int(len(fit_train_df)),
        "n_valid": int(len(valid_pred)),
        "n_test": int(len(test_pred)),
    }

    coef_df = pd.DataFrame({"feature": feature_cols, "importance": np.nan})
    model_bundle = {
        "model": model,
        "feature_columns": feature_cols,
        "target_col": target_col,
        "model_family": "HistGradientBoostingRegressor",
        "model_params": dict(MODEL_PARAMS),
        "training_subset": training_subset,
    }
    return pred_df, metrics, coef_df, model_bundle


def write_model_outputs(
    pred_df: pd.DataFrame,
    metrics: dict[str, object],
    coef_df: pd.DataFrame,
    model_bundle: dict[str, object],
    out_root: Path,
    prefix: str,
) -> dict[str, str]:
    ensure_dir(out_root)
    names = output_names(prefix)
    all_path = out_root / names["all_pred_csv"]
    train_path = out_root / names["train_pred_csv"]
    valid_path = out_root / names["valid_pred_csv"]
    test_path = out_root / names["test_pred_csv"]
    metrics_csv_path = out_root / names["metrics_csv"]
    metrics_json_path = out_root / names["metrics_json"]
    coef_csv_path = out_root / names["coef_csv"]
    model_path = out_root / names["model_joblib"]
    meta_path = out_root / names["meta_json"]

    for path in [
        all_path,
        train_path,
        valid_path,
        test_path,
        metrics_csv_path,
        metrics_json_path,
        coef_csv_path,
        model_path,
        meta_path,
    ]:
        ensure_dir(path.parent)

    pred_df.to_csv(all_path, index=False, encoding="utf-8-sig")
    pred_df.loc[pred_df["split"] == "train"].to_csv(train_path, index=False, encoding="utf-8-sig")
    pred_df.loc[pred_df["split"] == "valid"].to_csv(valid_path, index=False, encoding="utf-8-sig")
    pred_df.loc[pred_df["split"] == "test"].to_csv(test_path, index=False, encoding="utf-8-sig")
    pd.DataFrame(
        [
            {"block": "train", **metrics["train"]},
            {"block": "valid", **metrics["valid"]},
            {"block": "test", **metrics["test"]},
        ]
    ).to_csv(metrics_csv_path, index=False, encoding="utf-8-sig")
    coef_df.to_csv(coef_csv_path, index=False, encoding="utf-8-sig")
    joblib.dump(model_bundle, model_path)
    dump_json(metrics_json_path, metrics)
    dump_json(
        meta_path,
        {
            "prefix": prefix,
            "model_family": model_bundle["model_family"],
            "feature_columns": model_bundle["feature_columns"],
        },
    )
    return {
        "all_csv": str(all_path),
        "train_csv": str(train_path),
        "valid_csv": str(valid_path),
        "test_csv": str(test_path),
        "metrics_csv": str(metrics_csv_path),
        "metrics_json": str(metrics_json_path),
        "coef_csv": str(coef_csv_path),
        "model_joblib": str(model_path),
        "meta_json": str(meta_path),
    }


def export_frame(pred_df: pd.DataFrame, spec: dict[str, object], split_name: str) -> pd.DataFrame:
    part = pred_df[pred_df["split"] == split_name].copy()
    target_col = str(spec["target_col"])
    pred_col = selected_pred_col(spec)

    if spec["export_kind"] == "wind_hourly":
        return pd.DataFrame(
            {
                "시각_UTC": part["timestamp_utc"],
                "실제값": pd.to_numeric(part[target_col], errors="coerce"),
                "예측값": pd.to_numeric(part[pred_col], errors="coerce"),
            }
        )

    if spec["export_kind"] == "hourly":
        return pd.DataFrame(
            {
                "예측기준시각_ET": part["issue_timestamp_et"].astype(str),
                "TimeStamp": part["timestamp_utc"],
                "예측순번": pd.to_numeric(part["hour_et"], errors="coerce").astype(int),
                "실제값": pd.to_numeric(part[target_col], errors="coerce"),
                "예측값": pd.to_numeric(part[pred_col], errors="coerce"),
            }
        )

    return pd.DataFrame(
        {
            "예측기준시각_ET": part["issue_timestamp_et"].astype(str),
            "TimeStamp": part["timestamp_utc"],
            "예측순번": (
                pd.to_numeric(part["hour_et"], errors="coerce").astype(int) * 12
                + pd.to_numeric(part["slot_5m_et"], errors="coerce").astype(int)
            ),
            "실제값": pd.to_numeric(part[target_col], errors="coerce"),
            "예측값": pd.to_numeric(part[pred_col], errors="coerce"),
        }
    )


def _build_summary_frame(spec: dict[str, object], metrics_df: pd.DataFrame, test_pred_df: pd.DataFrame) -> pd.DataFrame:
    row_test = metrics_df.loc[metrics_df["block"] == "test"].iloc[0]

    return pd.DataFrame(
        [
            {
                "변수": str(spec["model_label"]),
                "최종사용모델": str(spec["model_desc"]),
                "학습구간": "2018-2019",
                "검증구간": "2020",
                "테스트구간": "2021",
                "평가구간": "테스트구간",
                "표본수_test": int(len(test_pred_df)),
                "최종모델_RMSE": float(row_test["rmse"]),
                "최종모델_MAE": float(row_test["mae"]),
                "최종모델_R2": float(row_test["r2"]),
                "비고": str(spec["summary_note"]),
            }
        ]
    )


def export_nonreg_series(series_key: str, pred_root: Path, user_out_root: Path) -> dict[str, object]:
    spec = SERIES_SPECS[series_key]
    names = output_names(str(spec["prefix"]))
    pred_csv = pred_root / names["all_pred_csv"]
    pred_df = pd.read_csv(pred_csv, encoding="utf-8-sig")
    if "timestamp_utc" in pred_df.columns:
        pred_df["timestamp_utc"] = pd.to_datetime(pred_df["timestamp_utc"], utc=True, errors="coerce")

    train_out = user_out_root / str(spec["train_result_name"])
    valid_out = user_out_root / str(spec["valid_result_name"])
    test_out = user_out_root / str(spec["test_result_name"])
    ensure_dir(user_out_root)
    export_frame(pred_df, spec, "train").to_csv(train_out, index=False, encoding="utf-8-sig")
    export_frame(pred_df, spec, "valid").to_csv(valid_out, index=False, encoding="utf-8-sig")
    export_frame(pred_df, spec, "test").to_csv(test_out, index=False, encoding="utf-8-sig")
    return {
        "series_key": series_key,
        "source_pred_csv": str(pred_csv),
        "train_result_csv": str(train_out),
        "valid_result_csv": str(valid_out),
        "test_result_csv": str(test_out),
    }


def rebuild_nonreg_export_summary(
    pred_root: Path,
    user_out_root: Path,
    meta_out_root: Path | None = None,
) -> dict[str, object]:
    ensure_dir(user_out_root)
    if meta_out_root is None:
        meta_out_root = pred_root
    ensure_dir(meta_out_root)

    meta_json = meta_out_root / "98_점예측_내보내기메타정보.json"
    outputs: dict[str, object] = {
        "summary_csvs": {},
        "meta_json": str(meta_json),
    }

    for series_key, spec in SERIES_SPECS.items():
        names = output_names(str(spec["prefix"]))
        metrics_df = pd.read_csv(pred_root / names["metrics_csv"], encoding="utf-8-sig")
        test_pred_df = pd.read_csv(pred_root / names["test_pred_csv"], encoding="utf-8-sig")
        summary = _build_summary_frame(spec, metrics_df, test_pred_df)
        summary_path = user_out_root / str(spec["summary_csv_name"])
        summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
        outputs["summary_csvs"][series_key] = str(summary_path)

    dump_json(meta_json, outputs)
    return outputs
