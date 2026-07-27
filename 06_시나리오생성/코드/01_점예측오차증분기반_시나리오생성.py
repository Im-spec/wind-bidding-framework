#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial import cKDTree


ROOT = Path(__file__).resolve().parents[2]
INPUT_ROOT = ROOT / "01_입력데이터"
POINT_ROOT = ROOT / "03_점예측" / "점예측결과"
ERROR_ROOT = ROOT / "04_오차데이터" / "오차데이터결과"
FIT_ROOT = ROOT / "05_오차증분_분포적합" / "오차증분_분포적합결과"
STABLE_PARAM_CSV = FIT_ROOT / "05_참고_Stable분포_정밀적합" / "62_통합_Stable분포_적합모수.csv"
SCENARIO_ROOT = ROOT / "06_시나리오생성" / "시나리오생성결과"
PROGRESS_JSON = SCENARIO_ROOT / "98_시나리오생성_진행상황.json"

META_JSON = SCENARIO_ROOT / "99_scenario_generation_meta.json"

TZ_ET = "America/New_York"
CAPACITY_MW = 1985.3
SKEWED_T_BETA_EPS = 1e-6

WIND_DIR = SCENARIO_ROOT / "__풍력시나리오__"
ENERGY_DIR = SCENARIO_ROOT / "__DA_RT에너지결합시나리오__"
REG_DIR = SCENARIO_ROOT / "__DA_REG_RT_REG결합시나리오__"
OFFICIAL_FAMILIES = [
    "gaussian",
    "laplace",
    "skewed_t",
    "stable",
    "levy_jump_gbody",
    "levy_jump_lbody",
    "mixed_selective_jump",
]
LEVY_JUMP_BODY_OVERRIDES = {
    "levy_jump": None,
    "levy_jump_gbody": "gaussian",
    "levy_jump_lbody": "laplace",
}
MIXED_FAMILY_MAP = {
    "mixed_selective_jump": {
        "wind": "levy_jump_gbody",
        "da": "levy_jump_lbody",
        "rt": "laplace",
        "da_reg": "levy_jump_lbody",
        "rt_reg": "gaussian",
    }
}
SUPPORTED_FAMILIES = {
    "gaussian",
    "laplace",
    "skewed_t",
    "stable",
    *LEVY_JUMP_BODY_OVERRIDES.keys(),
    *MIXED_FAMILY_MAP.keys(),
}
STABLE_DATASET_KEYS = {
    "wind": "wind",
    "da": "da_energy",
    "rt": "rt_energy",
    "da_reg": "da_reg",
    "rt_reg": "rt_reg",
}


def _effective_family(family: str, dataset_key: str) -> str:
    fam = str(family).strip().lower()
    if fam in MIXED_FAMILY_MAP:
        return MIXED_FAMILY_MAP[fam].get(dataset_key, "laplace")
    return fam


def _family_lookup_key(value: object) -> str:
    return str(value).strip().lower().replace("_", "").replace("-", "")

POINT_FILE_NAMES = {
    "wind": ("01_학습기간_풍력발전_예측결과.csv", "02_테스트기간_풍력발전_예측결과.csv"),
    "da": ("03_학습기간_DA에너지_예측결과.csv", "04_테스트기간_DA에너지_예측결과.csv"),
    "rt": ("05_학습기간_RT에너지_예측결과.csv", "06_테스트기간_RT에너지_예측결과.csv"),
    "da_reg": ("07_학습기간_하루전_REG가격_예측결과.csv", "08_테스트기간_하루전_REG가격_예측결과.csv"),
    "rt_reg": ("09_학습기간_실시간_REG가격_예측결과.csv", "10_테스트기간_실시간_REG가격_예측결과.csv"),
}
VALID_POINT_FILE_NAMES = {
    "wind": "16_검증기간_풍력발전_예측결과.csv",
    "da": "17_검증기간_DA에너지_예측결과.csv",
    "rt": "18_검증기간_RT에너지_예측결과.csv",
    "da_reg": "19_검증기간_하루전_REG가격_예측결과.csv",
    "rt_reg": "20_검증기간_실시간_REG가격_예측결과.csv",
}

ERROR_FILE_PREFIXES = {
    "wind": "01_",
    "da": "02_",
    "rt": "03_",
    "da_reg": "04_",
    "rt_reg": "05_",
}
FIT_FILE_PREFIXES = {
    "wind": "11_",
    "da": "12_",
    "rt": "13_",
    "da_reg": "14_",
    "rt_reg": "15_",
}

WIND_SCORE_COLS = ["Wind_pc1", "Wind_pc2"]
ENERGY_SCORE_COLS = ["DA_pc1", "DA_pc2", "RT_pc1", "RT_pc2"]
REG_SCORE_COLS = ["DA_REG_pc1", "DA_REG_pc2", "RT_REG_pc1", "RT_REG_pc2"]
ALL_SCORE_COLS = WIND_SCORE_COLS + ENERGY_SCORE_COLS + REG_SCORE_COLS


@dataclass
class PcaModel:
    mean_vec: np.ndarray
    components: np.ndarray

    def score(self, paths: np.ndarray) -> np.ndarray:
        centered = np.asarray(paths, dtype=np.float64) - self.mean_vec[None, :]
        return centered @ self.components.T


def _load_local_module(module_name: str, filename: str) -> object:
    code_dir = Path(__file__).resolve().parent
    module_path = code_dir / filename
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:  # pragma: no cover
        raise ImportError(f"Unable to load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_LEVY_JUMP = _load_local_module("scenario_levy_jump_common_util", "90_레비점프공통유틸.py")
build_levy_jump_model = _LEVY_JUMP.build_levy_jump_model
sample_levy_jump_increments = _LEVY_JUMP.sample_levy_jump_increments


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _dump_json(path: Path, obj: object) -> None:
    _ensure_dir(path.parent)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def _now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _write_progress(payload: dict[str, object]) -> None:
    _dump_json(PROGRESS_JSON, payload)


def _find_files_by_prefix(folder: Path, prefixes: tuple[str, ...] | str) -> list[Path]:
    if isinstance(prefixes, str):
        prefixes = (prefixes,)
    files = sorted(
        path for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() == ".csv" and any(path.name.startswith(prefix) for prefix in prefixes)
    )
    if not files:
        raise FileNotFoundError(f"No CSV found in {folder} with prefixes={prefixes}")
    return files


def _to_et_timestamp(series: pd.Series) -> pd.Series:
    ts = pd.to_datetime(series, utc=True, errors="coerce")
    return ts.dt.tz_convert(TZ_ET)


def _resolve_issue_time(target_et: pd.Series, explicit_issue: pd.Series | None) -> pd.Series:
    if explicit_issue is not None:
        parsed = pd.to_datetime(explicit_issue, utc=True, errors="coerce").dt.tz_convert(TZ_ET)
        return parsed
    midnight = target_et.dt.floor("D")
    return midnight - pd.Timedelta(days=1) + pd.Timedelta(hours=5)


def _normalize_point_frame(path: Path, kind: str) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    target_col = next(col for col in ("시각_UTC", "목표시각_UTC", "TimeStamp") if col in df.columns)
    issue_col = "예측기준시각_ET" if "예측기준시각_ET" in df.columns else None
    actual_col = next(col for col in ("실측값", "실제값") if col in df.columns)
    pred_col = "예측값"
    ts_utc = pd.to_datetime(df[target_col], utc=True, errors="coerce")
    target_et = ts_utc.dt.tz_convert(TZ_ET)
    issue_et = _resolve_issue_time(target_et, df[issue_col] if issue_col else None)
    out = pd.DataFrame(
        {
            "target_timestamp_utc": ts_utc,
            "target_timestamp_et": target_et,
            "issue_timestamp_et": issue_et,
            "issue_timestamp_utc": issue_et.dt.tz_convert("UTC"),
            "operating_date_et": target_et.dt.strftime("%Y-%m-%d"),
            "hour_et": target_et.dt.hour.astype(int),
            "actual_value": pd.to_numeric(df[actual_col], errors="coerce"),
            "pred_value": pd.to_numeric(df[pred_col], errors="coerce"),
            "kind": kind,
        }
    )
    out = out.dropna(subset=["target_timestamp_utc", "issue_timestamp_utc", "pred_value"]).copy()
    return out.sort_values("target_timestamp_utc").reset_index(drop=True)


def _load_point_results(kind: str) -> pd.DataFrame:
    file_names = list(POINT_FILE_NAMES[kind])
    valid_name = VALID_POINT_FILE_NAMES.get(kind)
    if valid_name:
        file_names.insert(1, valid_name)
    frames = [_normalize_point_frame(POINT_ROOT / name, kind) for name in file_names]
    df = pd.concat(frames, ignore_index=True)
    return df.sort_values(["target_timestamp_utc", "issue_timestamp_utc"]).drop_duplicates(
        subset=["target_timestamp_utc"], keep="last"
    ).reset_index(drop=True)


def _load_error_history(kind: str) -> pd.DataFrame:
    path = _find_files_by_prefix(ERROR_ROOT, ERROR_FILE_PREFIXES[kind])[0]
    df = pd.read_csv(path, encoding="utf-8-sig")
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True, errors="coerce")
    df["error_value"] = pd.to_numeric(df["error_value"], errors="coerce")
    df["delta_error"] = pd.to_numeric(df["delta_error"], errors="coerce")
    if "horizon_index" in df.columns:
        df["horizon_index"] = pd.to_numeric(df["horizon_index"], errors="coerce")
    else:
        df["horizon_index"] = np.nan
    if "hour_et" in df.columns:
        df["hour_et"] = pd.to_numeric(df["hour_et"], errors="coerce")
    else:
        df["hour_et"] = np.nan
    return df.dropna(subset=["timestamp_utc", "error_value"]).sort_values("timestamp_utc").reset_index(drop=True)


def _load_fit_params(kind: str, family: str) -> dict[str, float]:
    effective_family = _effective_family(family, kind)
    if effective_family == "stable":
        df = pd.read_csv(STABLE_PARAM_CSV, encoding="utf-8-sig")
        dataset_key = STABLE_DATASET_KEYS[kind]
        row = df.loc[df["dataset_key"].astype(str).str.lower() == dataset_key]
        if row.empty:
            raise KeyError(f"{kind} stable row not found in {STABLE_PARAM_CSV.name}")
        return row.iloc[0].to_dict()

    fit_dir = FIT_ROOT / "03_참고_기본분포_상세표"
    path = _find_files_by_prefix(fit_dir, FIT_FILE_PREFIXES[kind])[0]
    df = pd.read_csv(path, encoding="utf-8-sig")
    lookup_family = "laplace" if effective_family.lower().startswith("levy_jump") else effective_family.lower()
    lookup_key = _family_lookup_key(lookup_family)
    row = df.loc[df["distribution"].map(_family_lookup_key) == lookup_key]
    if row.empty:
        raise KeyError(f"{kind} {family} row not found in {path.name}")
    return row.iloc[0].to_dict()


def _fallback_issue_timestamp_et(operating_date_et: str) -> pd.Timestamp:
    midnight_et = pd.Timestamp(operating_date_et).tz_localize(TZ_ET)
    return midnight_et - pd.Timedelta(days=1) + pd.Timedelta(hours=5)


def _canonical_target_timestamps(issue_timestamp_utc: pd.Timestamp) -> pd.DatetimeIndex:
    start_utc = pd.Timestamp(issue_timestamp_utc) + pd.Timedelta(hours=19)
    return pd.date_range(start=start_utc, periods=24, freq="h", tz="UTC")


def _reindex_hourly_series(hours: np.ndarray, values: np.ndarray) -> np.ndarray:
    ser = pd.Series(np.asarray(values, dtype=np.float64), index=pd.Index(hours, dtype=int))
    ser = ser.groupby(level=0).mean().reindex(range(24))
    ser = ser.interpolate(method="linear", limit_direction="both")
    return ser.to_numpy(dtype=np.float64)


def _reindex_hourly_matrix(hours: np.ndarray, values: np.ndarray) -> np.ndarray:
    frame = pd.DataFrame(np.asarray(values, dtype=np.float64).T, index=pd.Index(hours, dtype=int))
    frame = frame.groupby(level=0).mean().reindex(range(24))
    frame = frame.interpolate(method="linear", axis=0, limit_direction="both")
    return frame.to_numpy(dtype=np.float64).T


def _regularize_hourly_point_day(day_df: pd.DataFrame, operating_date_et: str, kind: str) -> pd.DataFrame:
    if day_df.empty:
        raise ValueError(f"Missing {kind} point forecasts for {operating_date_et}")
    raw_rows = len(day_df)
    day_df = day_df.sort_values("target_timestamp_utc").reset_index(drop=True)
    grouped = (
        day_df.groupby("hour_et", as_index=False)
        .agg(
            pred_value=("pred_value", "mean"),
            actual_value=("actual_value", "mean"),
        )
        .sort_values("hour_et")
        .reset_index(drop=True)
    )
    issue_utc_series = pd.to_datetime(day_df["issue_timestamp_utc"], utc=True, errors="coerce").dropna()
    if issue_utc_series.empty:
        issue_et = _fallback_issue_timestamp_et(operating_date_et)
        issue_utc = issue_et.tz_convert("UTC")
    else:
        issue_utc = pd.Timestamp(issue_utc_series.iloc[0])
        issue_et = issue_utc.tz_convert(TZ_ET)

    hours = grouped["hour_et"].to_numpy(dtype=int)
    pred_24 = _reindex_hourly_series(hours, grouped["pred_value"].to_numpy(dtype=np.float64))
    actual_arr = grouped["actual_value"].to_numpy(dtype=np.float64)
    if np.isfinite(actual_arr).any():
        actual_24 = _reindex_hourly_series(hours, actual_arr)
    else:
        actual_24 = np.full(24, np.nan, dtype=np.float64)

    target_utc = _canonical_target_timestamps(issue_utc)
    out = pd.DataFrame(
        {
            "target_timestamp_utc": target_utc,
            "target_timestamp_et": target_utc.tz_convert(TZ_ET),
            "issue_timestamp_et": issue_et,
            "issue_timestamp_utc": issue_utc,
            "operating_date_et": operating_date_et,
            "hour_et": np.arange(24, dtype=int),
            "actual_value": actual_24,
            "pred_value": pred_24,
            "kind": kind,
        }
    )
    if raw_rows != 24 or len(np.unique(hours)) != 24:
        print(
            f"[repair] point-day regularized | kind={kind} | date={operating_date_et} | rows={raw_rows} -> 24",
            flush=True,
        )
    return out


def _initial_error_pool(error_df: pd.DataFrame, issue_timestamp_utc: pd.Timestamp) -> np.ndarray:
    base = error_df.loc[error_df["timestamp_utc"] < issue_timestamp_utc].copy()
    if base.empty:
        return np.array([], dtype=np.float64)
    if "horizon_index" in base.columns:
        pool = base.loc[base["horizon_index"] == 0, "error_value"].dropna().to_numpy(dtype=np.float64)
        if len(pool) >= 30:
            return pool
    if "hour_et" in base.columns:
        pool = base.loc[base["hour_et"] == 0, "error_value"].dropna().to_numpy(dtype=np.float64)
        if len(pool) >= 30:
            return pool
    return base["error_value"].dropna().to_numpy(dtype=np.float64)


def _sample_initial_errors(
    rng: np.random.Generator,
    error_df: pd.DataFrame,
    issue_timestamp_utc: pd.Timestamp,
    n_scenarios: int,
) -> np.ndarray:
    pool = _initial_error_pool(error_df, issue_timestamp_utc)
    if len(pool) == 0:
        return np.zeros(n_scenarios, dtype=np.float64)
    if len(pool) == 1:
        return np.full(n_scenarios, float(pool[0]), dtype=np.float64)
    return rng.choice(pool, size=n_scenarios, replace=True).astype(np.float64)


def _sample_increments(
    rng: np.random.Generator,
    dataset_key: str,
    family: str,
    params: dict[str, float],
    size: tuple[int, int],
    levy_jump_models: dict[str, object],
) -> np.ndarray:
    fam = _effective_family(family, dataset_key)
    if fam == "gaussian":
        return rng.normal(loc=float(params.get("mu", 0.0)), scale=float(params.get("sigma", 1.0)), size=size)
    if fam == "laplace":
        return rng.laplace(loc=float(params.get("loc", 0.0)), scale=float(params.get("scale", 1.0)), size=size)
    if fam == "skewed_t":
        a = max(float(params["a"]), 1e-8)
        b = max(float(params["b"]), 1e-8)
        loc = float(params.get("loc", 0.0))
        scale = max(float(params.get("scale", 1.0)), 1e-8)
        beta_draw = rng.beta(a, b, size=size)
        beta_draw = np.clip(beta_draw, SKEWED_T_BETA_EPS, 1.0 - SKEWED_T_BETA_EPS)
        numerator = (2.0 * beta_draw - 1.0) * np.sqrt(a + b)
        denominator = 2.0 * np.sqrt(beta_draw * (1.0 - beta_draw))
        return loc + scale * (numerator / denominator)
    if fam == "stable":
        stats.levy_stable.parameterization = "S1"
        return stats.levy_stable.rvs(
            float(params["alpha"]),
            float(params["beta"]),
            loc=float(params["loc"]),
            scale=max(float(params["scale"]), 1e-8),
            size=size,
            random_state=rng,
        )
    if fam.startswith("levy_jump"):
        family_models = levy_jump_models.get(fam) or levy_jump_models.get("levy_jump", {})
        model = family_models.get(dataset_key)
        if model is None:
            raise KeyError(f"Missing levy_jump model for family={family}, dataset={dataset_key}")
        return sample_levy_jump_increments(rng, model, size)
    raise ValueError(f"Unsupported family: {family}")


def _build_process_paths(
    hat: np.ndarray,
    initial_errors: np.ndarray,
    increments: np.ndarray,
    clip_min: float | None = None,
    clip_max: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    hat = np.asarray(hat, dtype=np.float64)
    initial_errors = np.asarray(initial_errors, dtype=np.float64).reshape(-1)
    n_scenarios = len(initial_errors)
    horizon = len(hat)
    if increments.shape != (n_scenarios, max(0, horizon - 1)):
        raise ValueError(
            f"Increment shape mismatch: expected {(n_scenarios, max(0, horizon - 1))}, got {increments.shape}"
        )
    err = np.empty((n_scenarios, horizon), dtype=np.float64)
    err[:, 0] = initial_errors
    if horizon > 1:
        err[:, 1:] = initial_errors[:, None] + np.cumsum(increments, axis=1)
    scn = hat[None, :] + err
    if clip_min is not None or clip_max is not None:
        scn = np.clip(scn, clip_min, clip_max)
        err = scn - hat[None, :]
    return err, scn


def _load_actual_from_error_csv(prefix: str, value_col: str, label: str, hourly_mean: bool = False) -> pd.DataFrame:
    path = _find_files_by_prefix(ERROR_ROOT, prefix)[0]
    df = pd.read_csv(path, encoding="utf-8-sig")
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True, errors="coerce")
    df[label] = pd.to_numeric(df[value_col], errors="coerce")
    df = df.dropna(subset=["timestamp_utc", label]).copy()
    ts_et = df["timestamp_utc"].dt.tz_convert(TZ_ET)
    df["operating_date"] = ts_et.dt.strftime("%Y-%m-%d")
    df["hour"] = ts_et.dt.hour.astype(int)
    agg = "mean" if hourly_mean else "last"
    return (
        df.groupby(["operating_date", "hour"], as_index=False)[label]
        .agg(agg)
        .sort_values(["operating_date", "hour"])
        .reset_index(drop=True)
    )


def _integrated_hourly_actual() -> pd.DataFrame:
    wind = _load_actual_from_error_csv("01_", "y_true_raw_mw", "Wind")
    da = _load_actual_from_error_csv("02_", "y_true_price", "DA")
    rt = _load_actual_from_error_csv("03_", "y_true_price", "RT", hourly_mean=True)
    da_reg = _load_actual_from_error_csv("04_", "y_true_price", "DA_REG")
    rt_reg = _load_actual_from_error_csv("05_", "y_true_price", "RT_REG")

    hourly = wind.merge(da, on=["operating_date", "hour"], how="inner")
    hourly = hourly.merge(rt, on=["operating_date", "hour"], how="inner")
    hourly = hourly.merge(da_reg, on=["operating_date", "hour"], how="inner")
    hourly = hourly.merge(rt_reg, on=["operating_date", "hour"], how="inner")
    return hourly.sort_values(["operating_date", "hour"]).reset_index(drop=True)


def _daily_path_pivot(hourly: pd.DataFrame, value_col: str) -> pd.DataFrame:
    pivot = hourly.pivot(index="operating_date", columns="hour", values=value_col).sort_index().sort_index(axis=1)
    pivot = pivot.reindex(columns=range(24))
    return pivot.dropna(axis=0, how="any")


def _fit_pca_model(paths: pd.DataFrame) -> tuple[PcaModel, pd.DataFrame]:
    x = paths.to_numpy(dtype=np.float64)
    mean_vec = x.mean(axis=0)
    centered = x - mean_vec
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    components = vt[:2].copy()
    scores = centered @ components.T
    row_mean = x.mean(axis=1)
    if np.corrcoef(scores[:, 0], row_mean)[0, 1] < 0:
        components[0] *= -1.0
        scores[:, 0] *= -1.0
    trend = np.linspace(-1.0, 1.0, x.shape[1])
    if components.shape[0] > 1 and float(np.dot(components[1], trend)) < 0:
        components[1] *= -1.0
        scores[:, 1] *= -1.0
    score_df = pd.DataFrame({"operating_date": paths.index, "pc1": scores[:, 0], "pc2": scores[:, 1]})
    return PcaModel(mean_vec=mean_vec, components=components), score_df


def _gaussian_rank_scores(score_frame: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=score_frame.index)
    if "operating_date" in score_frame.columns:
        out["operating_date"] = score_frame["operating_date"]
    for col in [c for c in score_frame.columns if c != "operating_date"]:
        rank = score_frame[col].rank(method="average")
        u = rank / (len(score_frame) + 1.0)
        out[col] = stats.norm.ppf(u)
    return out


def _nearest_psd_corr(matrix: np.ndarray) -> np.ndarray:
    sym = 0.5 * (matrix + matrix.T)
    vals, vecs = np.linalg.eigh(sym)
    vals = np.clip(vals, 1e-6, None)
    psd = (vecs * vals) @ vecs.T
    d = np.sqrt(np.diag(psd))
    corr = psd / np.outer(d, d)
    np.fill_diagonal(corr, 1.0)
    return corr


def _prepare_joint_models(calibration_end_exclusive: str) -> tuple[dict[str, PcaModel], np.ndarray, dict[str, float]]:
    hourly = _integrated_hourly_actual()
    hourly = hourly.loc[hourly["operating_date"] < calibration_end_exclusive].copy()
    common_dates: set[str] | None = None
    pivots: dict[str, pd.DataFrame] = {}
    for label in ("Wind", "DA", "RT", "DA_REG", "RT_REG"):
        piv = _daily_path_pivot(hourly, label)
        pivots[label] = piv
        common_dates = set(piv.index) if common_dates is None else common_dates.intersection(piv.index)
    ordered_dates = sorted(common_dates or [])
    if len(ordered_dates) < 30:
        raise ValueError("PCA joint calibration days are insufficient.")
    models: dict[str, PcaModel] = {}
    score_frame = pd.DataFrame({"operating_date": ordered_dates})
    explained: dict[str, float] = {}
    for label, piv in pivots.items():
        aligned = piv.loc[ordered_dates]
        model, scores = _fit_pca_model(aligned)
        models[label] = model
        score_frame[f"{label}_pc1"] = scores["pc1"].to_numpy()
        score_frame[f"{label}_pc2"] = scores["pc2"].to_numpy()
        centered = aligned.to_numpy(dtype=np.float64) - model.mean_vec
        _, s, _ = np.linalg.svd(centered, full_matrices=False)
        ratio = (s**2) / np.sum(s**2)
        explained[f"{label}_pc1_ratio"] = float(ratio[0]) if len(ratio) > 0 else np.nan
        explained[f"{label}_pc2_ratio"] = float(ratio[1]) if len(ratio) > 1 else np.nan
    z = _gaussian_rank_scores(score_frame)
    z_values = z.drop(columns=["operating_date"], errors="ignore").to_numpy(dtype=np.float64)
    corr = _nearest_psd_corr(np.corrcoef(z_values, rowvar=False))
    return models, corr, {"joint_calibration_days": len(ordered_dates), **explained}


def _conditional_gaussian_sample(
    rng: np.random.Generator,
    cond_draws: np.ndarray,
    sigma_cond: np.ndarray,
    sigma_target: np.ndarray,
    sigma_target_cond: np.ndarray,
) -> np.ndarray:
    sigma_cond = _nearest_psd_corr(sigma_cond)
    sigma_target = _nearest_psd_corr(sigma_target)
    sigma_target_cond = np.asarray(sigma_target_cond, dtype=np.float64)
    sigma_cond_inv = np.linalg.pinv(sigma_cond)
    cond_mean = cond_draws @ sigma_cond_inv @ sigma_target_cond.T
    cond_cov = sigma_target - sigma_target_cond @ sigma_cond_inv @ sigma_target_cond.T
    cond_cov = _nearest_psd_corr(cond_cov)
    noise = rng.multivariate_normal(np.zeros(sigma_target.shape[0]), cond_cov, size=len(cond_draws), method="eigh")
    return cond_mean + noise


def _sample_hierarchical_latent_targets(
    rng: np.random.Generator,
    n: int,
    corr: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sigma_w = _nearest_psd_corr(corr[:2, :2])
    sigma_e = _nearest_psd_corr(corr[2:6, 2:6])
    sigma_r = _nearest_psd_corr(corr[6:10, 6:10])
    sigma_ew = np.asarray(corr[2:6, :2], dtype=np.float64)
    sigma_rx = np.asarray(corr[6:10, :6], dtype=np.float64)
    sigma_x = _nearest_psd_corr(corr[:6, :6])

    z_wind = rng.multivariate_normal(np.zeros(2), sigma_w, size=n, method="eigh")
    z_energy = _conditional_gaussian_sample(
        rng=rng,
        cond_draws=z_wind,
        sigma_cond=sigma_w,
        sigma_target=sigma_e,
        sigma_target_cond=sigma_ew,
    )
    z_wind_energy = np.hstack([z_wind, z_energy])
    z_reg = _conditional_gaussian_sample(
        rng=rng,
        cond_draws=z_wind_energy,
        sigma_cond=sigma_x,
        sigma_target=sigma_r,
        sigma_target_cond=sigma_rx,
    )
    return z_wind, z_energy, z_reg


def _rank_to_uint16(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.int64)
    ranks[order] = np.arange(len(values), dtype=np.int64)
    if len(values) == 1:
        return np.zeros(1, dtype=np.uint32)
    return np.round(ranks.astype(np.float64) * (65535.0 / (len(values) - 1))).astype(np.uint32)


def _u_to_uint16(u: np.ndarray) -> np.ndarray:
    return np.clip(np.floor(u * 65535.0), 0, 65535).astype(np.uint32)


def _part1by1(n: np.ndarray) -> np.ndarray:
    n = n & np.uint32(0x0000FFFF)
    n = (n | (n << np.uint32(8))) & np.uint32(0x00FF00FF)
    n = (n | (n << np.uint32(4))) & np.uint32(0x0F0F0F0F)
    n = (n | (n << np.uint32(2))) & np.uint32(0x33333333)
    n = (n | (n << np.uint32(1))) & np.uint32(0x55555555)
    return n


def _morton_code(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return _part1by1(x) | (_part1by1(y) << np.uint32(1))


def _rank_normalize_columns(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    out = np.empty_like(arr, dtype=np.float64)
    n = arr.shape[0]
    for col_idx in range(arr.shape[1]):
        ranks = stats.rankdata(arr[:, col_idx], method="average")
        u = ranks / (n + 1.0)
        out[:, col_idx] = stats.norm.ppf(u)
    return out


def _match_pc12(generated_scores: np.ndarray, target_z: np.ndarray) -> np.ndarray:
    gx = _rank_to_uint16(generated_scores[:, 0])
    gy = _rank_to_uint16(generated_scores[:, 1])
    tx = _u_to_uint16(stats.norm.cdf(target_z[:, 0]))
    ty = _u_to_uint16(stats.norm.cdf(target_z[:, 1]))
    order_g = np.argsort(_morton_code(gx, gy), kind="mergesort")
    order_t = np.argsort(_morton_code(tx, ty), kind="mergesort")
    chosen = np.empty(len(generated_scores), dtype=np.int64)
    chosen[order_t] = order_g
    return chosen


def _match_rank_normal_nearest(generated_scores: np.ndarray, target_z: np.ndarray) -> np.ndarray:
    """Match each latent target to the nearest generated score in rank-normal space."""
    generated_z = _rank_normalize_columns(generated_scores)
    tree = cKDTree(generated_z)
    _, chosen = tree.query(np.asarray(target_z, dtype=np.float64), k=1)
    return np.asarray(chosen, dtype=np.int64)


def _select_joint_indices(
    *,
    matching_mode: str,
    wind_scores: np.ndarray,
    da_scores: np.ndarray,
    rt_scores: np.ndarray,
    da_reg_scores: np.ndarray | None,
    rt_reg_scores: np.ndarray | None,
    z_wind: np.ndarray,
    z_energy: np.ndarray,
    z_reg: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray] | None]:
    if matching_mode == "variable_pc12_morton":
        selected_energy = {
            "Wind": _match_pc12(wind_scores[:, :2], z_wind[:, 0:2]),
            "DA": _match_pc12(da_scores[:, :2], z_energy[:, 0:2]),
            "RT": _match_pc12(rt_scores[:, :2], z_energy[:, 2:4]),
        }
        selected_reg = None
        if da_reg_scores is not None and rt_reg_scores is not None:
            selected_reg = {
                "DA_REG": _match_pc12(da_reg_scores[:, :2], z_reg[:, 0:2]),
                "RT_REG": _match_pc12(rt_reg_scores[:, :2], z_reg[:, 2:4]),
            }
        return selected_energy, selected_reg

    if matching_mode == "block_pc12_euclidean":
        wind_idx = _match_rank_normal_nearest(wind_scores[:, :2], z_wind[:, 0:2])
        energy_scores = np.column_stack([da_scores[:, :2], rt_scores[:, :2]])
        energy_idx = _match_rank_normal_nearest(energy_scores, z_energy[:, :4])
        selected_energy = {
            "Wind": wind_idx,
            "DA": energy_idx,
            "RT": energy_idx,
        }
        selected_reg = None
        if da_reg_scores is not None and rt_reg_scores is not None:
            reg_scores = np.column_stack([da_reg_scores[:, :2], rt_reg_scores[:, :2]])
            reg_idx = _match_rank_normal_nearest(reg_scores, z_reg[:, :4])
            selected_reg = {
                "DA_REG": reg_idx,
                "RT_REG": reg_idx,
            }
        return selected_energy, selected_reg

    raise ValueError(f"Unsupported matching mode: {matching_mode}")


def _aggregate_rt_hourly(
    operating_date_et: str,
    issue_utc: pd.Timestamp,
    issue_et: str,
    rt_day: pd.DataFrame,
    err_5m: np.ndarray,
    scn_5m: np.ndarray,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    group = rt_day[["target_timestamp_utc", "hour_et", "pred_value"]].copy()
    group["row_idx"] = np.arange(len(group))
    hourly_rows: list[dict[str, object]] = []
    err_parts: list[np.ndarray] = []
    for hour_et, sub in group.groupby("hour_et", sort=True):
        idx = sub["row_idx"].to_numpy(dtype=int)
        hourly_rows.append(
            {
                "hour_et": int(hour_et),
                "target_timestamp_utc": sub["target_timestamp_utc"].iloc[0],
                "pred_value": float(sub["pred_value"].mean()),
            }
        )
        err_parts.append(err_5m[:, idx].mean(axis=1))
    hourly = pd.DataFrame(hourly_rows).sort_values("hour_et").reset_index(drop=True)
    hours = hourly["hour_et"].to_numpy(dtype=int)
    hat_24 = _reindex_hourly_series(hours, hourly["pred_value"].to_numpy(dtype=np.float64))
    err_24 = _reindex_hourly_matrix(hours, np.column_stack(err_parts))
    scn_24 = hat_24[None, :] + err_24
    target_utc = _canonical_target_timestamps(issue_utc)
    hourly_24 = pd.DataFrame(
        {
            "hour_et": np.arange(24, dtype=int),
            "target_timestamp_utc": target_utc,
            "target_timestamp_et": target_utc.tz_convert(TZ_ET),
            "issue_timestamp_utc": issue_utc,
            "issue_timestamp_et": issue_et,
            "operating_date_et": operating_date_et,
            "pred_value": hat_24,
        }
    )
    if len(rt_day) != 288 or len(np.unique(hours)) != 24:
        print(
            f"[repair] rt-hourly regularized | date={operating_date_et} | rows_5m={len(rt_day)} | hours={len(np.unique(hours))} -> 24",
            flush=True,
        )
    return hourly_24, err_24, scn_24


def _energy_marginals_for_day(
    operating_date_et: str,
    point_map: dict[str, pd.DataFrame],
    error_map: dict[str, pd.DataFrame],
    family: str,
    fit_map: dict[str, dict[str, float]],
    levy_jump_models: dict[str, object],
    rng: np.random.Generator,
    n_scenarios: int,
) -> dict[str, object]:
    wind_day = point_map["wind"].loc[point_map["wind"]["operating_date_et"] == operating_date_et].copy()
    da_day = point_map["da"].loc[point_map["da"]["operating_date_et"] == operating_date_et].copy()
    rt_day = point_map["rt"].loc[point_map["rt"]["operating_date_et"] == operating_date_et].copy()
    if rt_day.empty:
        raise ValueError(f"Missing RT energy point forecasts for {operating_date_et}")
    wind_day = _regularize_hourly_point_day(wind_day, operating_date_et, "wind")
    da_day = _regularize_hourly_point_day(da_day, operating_date_et, "da")
    rt_day = rt_day.sort_values("target_timestamp_utc").reset_index(drop=True)
    issue_utc = pd.Timestamp(da_day["issue_timestamp_utc"].iloc[0])
    issue_et = str(da_day["issue_timestamp_et"].iloc[0])

    wind_init = _sample_initial_errors(rng, error_map["wind"], issue_utc, n_scenarios)
    da_init = _sample_initial_errors(rng, error_map["da"], issue_utc, n_scenarios)
    rt_init = _sample_initial_errors(rng, error_map["rt"], issue_utc, n_scenarios)

    wind_incr = _sample_increments(rng, "wind", family, fit_map["wind"], (n_scenarios, 23), levy_jump_models)
    da_incr = _sample_increments(rng, "da", family, fit_map["da"], (n_scenarios, 23), levy_jump_models)
    rt_incr = _sample_increments(rng, "rt", family, fit_map["rt"], (n_scenarios, max(0, len(rt_day) - 1)), levy_jump_models)

    wind_err, wind_scn = _build_process_paths(
        hat=wind_day["pred_value"].to_numpy(dtype=np.float64),
        initial_errors=wind_init,
        increments=wind_incr,
        clip_min=0.0,
        clip_max=CAPACITY_MW,
    )
    da_err, da_scn = _build_process_paths(
        hat=da_day["pred_value"].to_numpy(dtype=np.float64),
        initial_errors=da_init,
        increments=da_incr,
    )
    rt_err_5m, rt_scn_5m = _build_process_paths(
        hat=rt_day["pred_value"].to_numpy(dtype=np.float64),
        initial_errors=rt_init,
        increments=rt_incr,
    )
    rt_hourly, rt_err, rt_scn = _aggregate_rt_hourly(
        operating_date_et=operating_date_et,
        issue_utc=issue_utc,
        issue_et=issue_et,
        rt_day=rt_day,
        err_5m=rt_err_5m,
        scn_5m=rt_scn_5m,
    )

    return {
        "issue_utc": issue_utc,
        "issue_et": issue_et,
        "wind_day": wind_day[["hour_et", "target_timestamp_utc", "pred_value"]].rename(columns={"pred_value": "hat"}),
        "da_day": da_day[["hour_et", "target_timestamp_utc", "pred_value"]].rename(columns={"pred_value": "hat"}),
        "rt_day": rt_hourly.rename(columns={"pred_value": "hat"}),
        "wind_err": wind_err,
        "wind_scn": wind_scn,
        "da_err": da_err,
        "da_scn": da_scn,
        "rt_err": rt_err,
        "rt_scn": rt_scn,
    }


def _reg_marginals_for_day(
    operating_date_et: str,
    point_map: dict[str, pd.DataFrame],
    error_map: dict[str, pd.DataFrame],
    family: str,
    fit_map: dict[str, dict[str, float]],
    levy_jump_models: dict[str, object],
    rng: np.random.Generator,
    n_scenarios: int,
) -> dict[str, object]:
    da_reg_day = point_map["da_reg"].loc[point_map["da_reg"]["operating_date_et"] == operating_date_et].copy()
    rt_reg_day = point_map["rt_reg"].loc[point_map["rt_reg"]["operating_date_et"] == operating_date_et].copy()
    da_reg_day = _regularize_hourly_point_day(da_reg_day, operating_date_et, "da_reg")
    rt_reg_day = _regularize_hourly_point_day(rt_reg_day, operating_date_et, "rt_reg")
    issue_utc = pd.Timestamp(da_reg_day["issue_timestamp_utc"].iloc[0])
    issue_et = str(da_reg_day["issue_timestamp_et"].iloc[0])

    da_reg_init = _sample_initial_errors(rng, error_map["da_reg"], issue_utc, n_scenarios)
    rt_reg_init = _sample_initial_errors(rng, error_map["rt_reg"], issue_utc, n_scenarios)
    da_reg_incr = _sample_increments(rng, "da_reg", family, fit_map["da_reg"], (n_scenarios, 23), levy_jump_models)
    rt_reg_incr = _sample_increments(rng, "rt_reg", family, fit_map["rt_reg"], (n_scenarios, 23), levy_jump_models)
    da_reg_err, da_reg_scn = _build_process_paths(
        hat=da_reg_day["pred_value"].to_numpy(dtype=np.float64),
        initial_errors=da_reg_init,
        increments=da_reg_incr,
    )
    rt_reg_err, rt_reg_scn = _build_process_paths(
        hat=rt_reg_day["pred_value"].to_numpy(dtype=np.float64),
        initial_errors=rt_reg_init,
        increments=rt_reg_incr,
    )
    return {
        "issue_utc": issue_utc,
        "issue_et": issue_et,
        "da_reg_day": da_reg_day[["hour_et", "target_timestamp_utc", "pred_value"]].rename(columns={"pred_value": "hat"}),
        "rt_reg_day": rt_reg_day[["hour_et", "target_timestamp_utc", "pred_value"]].rename(columns={"pred_value": "hat"}),
        "da_reg_err": da_reg_err,
        "da_reg_scn": da_reg_scn,
        "rt_reg_err": rt_reg_err,
        "rt_reg_scn": rt_reg_scn,
    }


def _assemble_energy_day(operating_date_et: str, energy: dict[str, object], selected: dict[str, np.ndarray]) -> pd.DataFrame:
    wind_idx = selected["Wind"]
    da_idx = selected["DA"]
    rt_idx = selected["RT"]
    wind_day = energy["wind_day"]
    da_day = energy["da_day"]
    rt_day = energy["rt_day"]
    rows: list[dict[str, object]] = []
    for scenario_id in range(len(wind_idx)):
        for h in range(24):
            wind_val = float(energy["wind_scn"][wind_idx[scenario_id], h])
            da_val = float(energy["da_scn"][da_idx[scenario_id], h])
            rt_val = float(energy["rt_scn"][rt_idx[scenario_id], h])
            rows.append(
                {
                    "operating_date_et": operating_date_et,
                    "hour_et": int(da_day.iloc[h]["hour_et"]),
                    "timestamp_utc": pd.Timestamp(da_day.iloc[h]["target_timestamp_utc"]),
                    "scenario_id": int(scenario_id),
                    "joint_scenario_id": int(scenario_id),
                    "issue_timestamp_utc": pd.Timestamp(energy["issue_utc"]),
                    "issue_timestamp_et": energy["issue_et"],
                    "wind_hat_mw": float(wind_day.iloc[h]["hat"]),
                    "da_energy_hat": float(da_day.iloc[h]["hat"]),
                    "rt_energy_hat": float(rt_day.iloc[h]["hat"]),
                    "wind_error_scn": float(energy["wind_err"][wind_idx[scenario_id], h]),
                    "da_energy_error_scn": float(energy["da_err"][da_idx[scenario_id], h]),
                    "rt_energy_error_scn": float(energy["rt_err"][rt_idx[scenario_id], h]),
                    "wind_scn_mw": wind_val,
                    "da_energy_scn": da_val,
                    "rt_energy_scn": rt_val,
                    "spread_hat": float(rt_day.iloc[h]["hat"] - da_day.iloc[h]["hat"]),
                    "spread_error_scn": float(
                        energy["rt_err"][rt_idx[scenario_id], h] - energy["da_err"][da_idx[scenario_id], h]
                    ),
                    "spread_scn": float(rt_val - da_val),
                    "wind_source_scenario_id": int(wind_idx[scenario_id]),
                    "da_source_scenario_id": int(da_idx[scenario_id]),
                    "rt_source_scenario_id": int(rt_idx[scenario_id]),
                }
            )
    return pd.DataFrame(rows)


def _split_energy_blocks(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    shared_cols = [
        "operating_date_et",
        "hour_et",
        "timestamp_utc",
        "scenario_id",
        "joint_scenario_id",
        "issue_timestamp_utc",
        "issue_timestamp_et",
    ]
    wind_cols = shared_cols + [
        "wind_hat_mw",
        "wind_error_scn",
        "wind_scn_mw",
        "wind_source_scenario_id",
    ]
    energy_cols = shared_cols + [
        "da_energy_hat",
        "rt_energy_hat",
        "da_energy_error_scn",
        "rt_energy_error_scn",
        "da_energy_scn",
        "rt_energy_scn",
        "spread_hat",
        "spread_error_scn",
        "spread_scn",
        "da_source_scenario_id",
        "rt_source_scenario_id",
    ]
    return (
        df.loc[:, wind_cols].copy(),
        df.loc[:, energy_cols].copy(),
    )


def _assemble_reg_day(operating_date_et: str, reg: dict[str, object], selected: dict[str, np.ndarray]) -> pd.DataFrame:
    da_idx = selected["DA_REG"]
    rt_idx = selected["RT_REG"]
    da_day = reg["da_reg_day"]
    rt_day = reg["rt_reg_day"]
    rows: list[dict[str, object]] = []
    for scenario_id in range(len(da_idx)):
        for h in range(24):
            da_val = float(reg["da_reg_scn"][da_idx[scenario_id], h])
            rt_val = float(reg["rt_reg_scn"][rt_idx[scenario_id], h])
            rows.append(
                {
                    "operating_date_et": operating_date_et,
                    "hour_et": int(da_day.iloc[h]["hour_et"]),
                    "timestamp_utc": pd.Timestamp(da_day.iloc[h]["target_timestamp_utc"]),
                    "scenario_id": int(scenario_id),
                    "joint_scenario_id": int(scenario_id),
                    "issue_timestamp_utc": pd.Timestamp(reg["issue_utc"]),
                    "issue_timestamp_et": reg["issue_et"],
                    "da_reg_hat": float(da_day.iloc[h]["hat"]),
                    "rt_reg_hat": float(rt_day.iloc[h]["hat"]),
                    "da_reg_error_scn": float(reg["da_reg_err"][da_idx[scenario_id], h]),
                    "rt_reg_error_scn": float(reg["rt_reg_err"][rt_idx[scenario_id], h]),
                    "da_reg_scn": da_val,
                    "rt_reg_scn": rt_val,
                    "penalty_price_scn": float(max(0.0, da_val, rt_val)),
                    "da_reg_source_scenario_id": int(da_idx[scenario_id]),
                    "rt_reg_source_scenario_id": int(rt_idx[scenario_id]),
                }
            )
    return pd.DataFrame(rows)


def _save_day_parquet(df: pd.DataFrame, root: Path, family: str, operating_date_et: str, file_name: str) -> Path:
    out_dir = _ensure_dir(root / f"joint_case_{family.lower()}" / "joint_day_parts" / f"operating_date_et={operating_date_et}")
    out_path = out_dir / file_name
    df.to_parquet(out_path, index=False)
    return out_path


def main() -> None:
    global SCENARIO_ROOT, PROGRESS_JSON, META_JSON, WIND_DIR, ENERGY_DIR, REG_DIR
    parser = argparse.ArgumentParser(description="PC1+PC2 ?? wind/energy/reg 3?? ?? ??? ?? ?? 24?? ???? ??")
    parser.add_argument("--families", nargs="+", default=OFFICIAL_FAMILIES)
    parser.add_argument("--date-start", required=True)
    parser.add_argument("--date-end", required=True)
    parser.add_argument("--n-scenarios", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=777)
    parser.add_argument("--chunk-index", type=int, default=0)
    parser.add_argument("--chunk-count", type=int, default=1)
    parser.add_argument(
        "--scenario-root-name",
        default="시나리오생성결과",
        help="Output folder name under 06_시나리오생성. Use a new name to keep experimental scenario sets separate.",
    )
    parser.add_argument("--progress-name", default="98_시나리오생성_진행상황.json")
    parser.add_argument("--meta-name", default="99_scenario_generation_meta.json")
    parser.add_argument(
        "--matching-mode",
        choices=["variable_pc12_morton", "block_pc12_euclidean"],
        default="variable_pc12_morton",
        help="Scenario assembly rule: variable-wise 2D Morton rank matching or block-wise Euclidean matching.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--skip-reg",
        action="store_true",
        help="Skip DA/RT regulation-price scenario generation when downstream settlement does not use REG scenarios.",
    )
    args = parser.parse_args()

    if args.chunk_count < 1:
        raise ValueError("--chunk-count must be at least 1")
    if not (0 <= args.chunk_index < args.chunk_count):
        raise ValueError("--chunk-index must satisfy 0 <= chunk-index < chunk-count")

    SCENARIO_ROOT = ROOT / "06_시나리오생성" / args.scenario_root_name
    WIND_DIR = SCENARIO_ROOT / "__풍력시나리오__"
    ENERGY_DIR = SCENARIO_ROOT / "__DA_RT에너지결합시나리오__"
    REG_DIR = SCENARIO_ROOT / "__DA_REG_RT_REG결합시나리오__"
    PROGRESS_JSON = SCENARIO_ROOT / args.progress_name
    META_JSON = SCENARIO_ROOT / args.meta_name

    point_map = {kind: _load_point_results(kind) for kind in POINT_FILE_NAMES}
    error_map = {kind: _load_error_history(kind) for kind in ERROR_FILE_PREFIXES}
    normalized_families = [str(x).strip().lower() for x in args.families]
    unsupported = [fam for fam in normalized_families if fam not in SUPPORTED_FAMILIES]
    if unsupported:
        raise ValueError(f"Unsupported families: {unsupported}")
    args.families = normalized_families
    requested_levy_families = sorted(
        {
            _effective_family(fam, kind)
            for fam in args.families
            for kind in ERROR_FILE_PREFIXES
            if _effective_family(fam, kind).startswith("levy_jump")
        }
    )
    levy_jump_models = {
        fam: {
            kind: build_levy_jump_model(kind, error_map[kind], body_family_override=LEVY_JUMP_BODY_OVERRIDES[fam])
            for kind in ERROR_FILE_PREFIXES
        }
        for fam in requested_levy_families
    }
    models, joint_corr, model_meta = _prepare_joint_models(args.date_start)

    common_operating_dates = set(point_map["wind"]["operating_date_et"].unique().tolist())
    for kind in ("da", "rt", "da_reg", "rt_reg"):
        common_operating_dates &= set(point_map[kind]["operating_date_et"].unique().tolist())
    all_dates_full = sorted(
        date for date in common_operating_dates
        if args.date_start <= date <= args.date_end
    )
    all_dates = [date for idx, date in enumerate(all_dates_full) if idx % args.chunk_count == args.chunk_index]
    family_meta: dict[str, list[dict[str, object]]] = {}
    total_steps = max(1, len(args.families) * len(all_dates))
    started_at = _now_iso()
    run_started = time.perf_counter()
    completed_steps = 0
    current_family: str | None = None
    current_date: str | None = None

    _write_progress(
        {
            "status": "running",
            "started_at": started_at,
            "updated_at": started_at,
            "date_start": args.date_start,
            "date_end": args.date_end,
            "chunk_index": int(args.chunk_index),
            "chunk_count": int(args.chunk_count),
            "families": args.families,
            "n_scenarios": int(args.n_scenarios),
            "total_days_full": len(all_dates_full),
            "total_days": len(all_dates),
            "total_steps": int(total_steps),
            "completed_steps": 0,
            "progress_ratio": 0.0,
            "current_family": None,
            "current_date": None,
            "last_completed_family": None,
            "last_completed_date": None,
            "elapsed_seconds": 0.0,
        }
    )

    try:
        for family_idx, family in enumerate(args.families):
            current_family = family
            fit_map = {kind: _load_fit_params(kind, family) for kind in FIT_FILE_PREFIXES}
            rng_family = np.random.default_rng(args.seed + family_idx * 1000)
            family_records: list[dict[str, object]] = []
            print(f"[family-start] {family} | days={len(all_dates)} | n_scenarios={args.n_scenarios}", flush=True)
            for day_idx, operating_date_et in enumerate(all_dates):
                current_date = operating_date_et
                step_no = completed_steps + 1
                pct = 100.0 * completed_steps / total_steps
                print(
                    f"[{step_no}/{total_steps}] start | family={family} | date={operating_date_et} | progress={pct:5.1f}%",
                    flush=True,
                )
                _write_progress(
                    {
                        "status": "running",
                        "started_at": started_at,
                        "updated_at": _now_iso(),
                        "date_start": args.date_start,
                        "date_end": args.date_end,
                        "chunk_index": int(args.chunk_index),
                        "chunk_count": int(args.chunk_count),
                        "families": args.families,
                        "n_scenarios": int(args.n_scenarios),
                        "total_days_full": len(all_dates_full),
                        "total_days": len(all_dates),
                        "total_steps": int(total_steps),
                        "completed_steps": int(completed_steps),
                        "progress_ratio": float(completed_steps / total_steps),
                        "current_family": family,
                        "current_date": operating_date_et,
                        "last_completed_family": family if completed_steps > 0 else None,
                        "last_completed_date": all_dates[day_idx - 1] if day_idx > 0 else None,
                        "elapsed_seconds": round(time.perf_counter() - run_started, 1),
                    }
                )

                rng_day = np.random.default_rng(rng_family.integers(0, 2**31 - 1) + day_idx)
                energy = _energy_marginals_for_day(operating_date_et, point_map, error_map, family, fit_map, levy_jump_models, rng_day, args.n_scenarios)

                wind_scores = models["Wind"].score(energy["wind_scn"])
                da_scores = models["DA"].score(energy["da_scn"])
                rt_scores = models["RT"].score(energy["rt_scn"])

                z_wind, z_energy, z_reg = _sample_hierarchical_latent_targets(rng_day, args.n_scenarios, joint_corr)
                selected_energy, _ = _select_joint_indices(
                    matching_mode=args.matching_mode,
                    wind_scores=wind_scores,
                    da_scores=da_scores,
                    rt_scores=rt_scores,
                    da_reg_scores=None,
                    rt_reg_scores=None,
                    z_wind=z_wind,
                    z_energy=z_energy,
                    z_reg=z_reg,
                )

                energy_df = _assemble_energy_day(operating_date_et, energy, selected_energy)
                wind_df, da_rt_energy_df = _split_energy_blocks(energy_df)
                wind_path = _save_day_parquet(
                    wind_df, WIND_DIR, family, operating_date_et, f"01_{operating_date_et}_wind_scenarios.parquet"
                )
                energy_path = _save_day_parquet(
                    da_rt_energy_df, ENERGY_DIR, family, operating_date_et, f"01_{operating_date_et}_DA_RT_energy_joint_scenarios.parquet"
                )
                reg_path = None
                if not args.skip_reg:
                    reg = _reg_marginals_for_day(operating_date_et, point_map, error_map, family, fit_map, levy_jump_models, rng_day, args.n_scenarios)
                    da_reg_scores = models["DA_REG"].score(reg["da_reg_scn"])
                    rt_reg_scores = models["RT_REG"].score(reg["rt_reg_scn"])
                    _, selected_reg = _select_joint_indices(
                        matching_mode=args.matching_mode,
                        wind_scores=wind_scores,
                        da_scores=da_scores,
                        rt_scores=rt_scores,
                        da_reg_scores=da_reg_scores,
                        rt_reg_scores=rt_reg_scores,
                        z_wind=z_wind,
                        z_energy=z_energy,
                        z_reg=z_reg,
                    )
                    if selected_reg is None:
                        raise RuntimeError("REG matching returned no selected indices.")
                    reg_df = _assemble_reg_day(operating_date_et, reg, selected_reg)
                    reg_path = _save_day_parquet(
                        reg_df, REG_DIR, family, operating_date_et, f"01_{operating_date_et}_DA_REG_RT_REG_joint_scenarios.parquet"
                    )
                family_records.append(
                    {
                        "operating_date_et": operating_date_et,
                        "wind_path": str(wind_path),
                        "energy_path": str(energy_path),
                        "reg_path": "" if reg_path is None else str(reg_path),
                    }
                )
                completed_steps += 1
                pct_done = 100.0 * completed_steps / total_steps
                print(
                    f"[{completed_steps}/{total_steps}] done  | family={family} | date={operating_date_et} | progress={pct_done:5.1f}%",
                    flush=True,
                )
                _write_progress(
                    {
                        "status": "running",
                        "started_at": started_at,
                        "updated_at": _now_iso(),
                        "date_start": args.date_start,
                        "date_end": args.date_end,
                        "chunk_index": int(args.chunk_index),
                        "chunk_count": int(args.chunk_count),
                        "families": args.families,
                        "n_scenarios": int(args.n_scenarios),
                        "total_days_full": len(all_dates_full),
                        "total_days": len(all_dates),
                        "total_steps": int(total_steps),
                        "completed_steps": int(completed_steps),
                        "progress_ratio": float(completed_steps / total_steps),
                        "current_family": family,
                        "current_date": operating_date_et,
                        "last_completed_family": family,
                        "last_completed_date": operating_date_et,
                        "elapsed_seconds": round(time.perf_counter() - run_started, 1),
                    }
                )
            family_meta[family] = family_records
    except Exception as exc:
        _write_progress(
            {
                "status": "failed",
                "started_at": started_at,
                "updated_at": _now_iso(),
                "date_start": args.date_start,
                "date_end": args.date_end,
                "chunk_index": int(args.chunk_index),
                "chunk_count": int(args.chunk_count),
                "families": args.families,
                "n_scenarios": int(args.n_scenarios),
                "total_days_full": len(all_dates_full),
                "total_days": len(all_dates),
                "total_steps": int(total_steps),
                "completed_steps": int(completed_steps),
                "progress_ratio": float(completed_steps / total_steps),
                "current_family": current_family,
                "current_date": current_date,
                "last_completed_family": current_family if completed_steps > 0 else None,
                "last_completed_date": current_date if completed_steps > 0 else None,
                "elapsed_seconds": round(time.perf_counter() - run_started, 1),
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            }
        )
        raise

    _dump_json(
        META_JSON,
        {
            "scenario_root": str(SCENARIO_ROOT),
            "wind_root": str(WIND_DIR),
            "energy_root": str(ENERGY_DIR),
            "reg_root": str(REG_DIR),
            "families": args.families,
            "family_generation_modes": {
                family: (
                    "stable_increment_process"
                    if str(family).lower() == "stable"
                    else
                    "skewed_t_increment_process"
                    if str(family).lower() == "skewed_t"
                    else
                    "gaussian_body_bernoulli_jump_mixture"
                    if str(family).lower() == "levy_jump_gbody"
                    else "laplace_body_bernoulli_jump_mixture"
                    if str(family).lower() == "levy_jump_lbody"
                    else "preferred_body_bernoulli_jump_mixture"
                    if str(family).lower() == "levy_jump"
                    else f"mixed_variable_family:{MIXED_FAMILY_MAP[str(family).lower()]}"
                    if str(family).lower() in MIXED_FAMILY_MAP
                    else "increment_process"
                )
                for family in args.families
            },
            "levy_jump_models": {
                fam: {k: v.to_meta() for k, v in model_map.items()}
                for fam, model_map in levy_jump_models.items()
            },
            "date_start": args.date_start,
            "date_end": args.date_end,
            "chunk_index": int(args.chunk_index),
            "chunk_count": int(args.chunk_count),
            "n_scenarios": args.n_scenarios,
            "seed": args.seed,
            "skip_reg": bool(args.skip_reg),
            "matching_mode": args.matching_mode,
            "joint_method": "hierarchical_3block_pc12_conditional_coupling",
            "score_matching_method": (
                "block_pc12_rank_normal_euclidean_nearest_neighbor"
                if args.matching_mode == "block_pc12_euclidean"
                else "pc1_pc2_morton_rank_matching"
            ),
            "initial_error_mode": "empirical_horizon0_bootstrap",
            "path_process_mode": "scenario_specific_initial_error_plus_cumulative_increment",
            "wind_score_cols": WIND_SCORE_COLS,
            "energy_score_cols": ENERGY_SCORE_COLS,
            "reg_score_cols": REG_SCORE_COLS,
            "joint_score_corr": np.round(joint_corr, 6).tolist(),
            "model_meta": model_meta,
            "families_detail": family_meta,
        },
    )
    _write_progress(
        {
            "status": "completed",
            "started_at": started_at,
            "updated_at": _now_iso(),
            "date_start": args.date_start,
            "date_end": args.date_end,
            "chunk_index": int(args.chunk_index),
            "chunk_count": int(args.chunk_count),
            "families": args.families,
            "n_scenarios": int(args.n_scenarios),
            "total_days_full": len(all_dates_full),
            "total_days": len(all_dates),
            "total_steps": int(total_steps),
            "completed_steps": int(completed_steps),
            "progress_ratio": 1.0,
            "elapsed_seconds": round(time.perf_counter() - run_started, 1),
        }
    )


if __name__ == "__main__":
    main()










