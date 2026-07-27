#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover
    plt = None

CAPACITY_MW = 1985.3
TZ_ET = "America/New_York"

VALIDATION_DETAIL_CSV = "11_실현정산_시간별.csv"
VALIDATION_ANNUAL_CSV = "12_연간수익_비교.csv"
VALIDATION_PENALTY_CSV = "13_패널티요약.csv"
VALIDATION_DISTORTION_CSV = "14_낙찰왜곡요약.csv"
VALIDATION_META_JSON = "15_검증_메타정보.json"
VALIDATION_CUMULATIVE_PNG = "16_연간누적정산금_비교.png"
VALIDATION_REPRESENTATIVE_PNG = "17_대표일_입찰곡선_예시.png"

_FAMILY_CASE_DIRS = {
    "gaussian": "분포사례_가우시안",
    "laplace": "분포사례_라플라스",
    "skewed_t": "분포사례_SkewedT",
    "skewedt": "분포사례_SkewedT",
    "stable": "분포사례_Stable",
    "stable_qmap": "분포사례_Stable_QMap",
    "levy_jump": "분포사례_레비점프",
    "levy_jump_gbody": "분포사례_레비점프_Gbody",
    "levy_jump_lbody": "분포사례_레비점프_Lbody",
    "mixed_selective_jump": "분포사례_혼합선택점프",
    "mixed_selective_jump_qmap": "분포사례_혼합선택점프_QMap",
    "gaussianjump": "분포사례_가우시안점프",
}
PROJECT_OVERRIDE_ENV = "WIND_BID_PROJECT_ROOT"
FALLBACK_PROJECT_DIRNAME = "논문"


def ensure_dir(path: Path | str) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def dump_json(path: Path | str, obj: object) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as fp:
        json.dump(obj, fp, ensure_ascii=False, indent=2)


def _candidate_project_roots(anchor_path: Path) -> list[Path]:
    code_root = anchor_path.parent if anchor_path.is_file() else anchor_path
    candidates: list[Path] = []
    override = os.environ.get(PROJECT_OVERRIDE_ENV, "").strip()
    if override:
        candidates.append(Path(override))
    candidates.append(code_root.parent)
    candidates.append(code_root.parent.parent / FALLBACK_PROJECT_DIRNAME)
    out: list[Path] = []
    seen: set[str] = set()
    for cand in candidates:
        key = str(cand).lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(cand)
    return out


def _is_flat_project_root(base: Path) -> bool:
    required = {
        "01_입력데이터",
        "02_입력데이터가공",
        "03_점예측",
        "04_오차데이터",
        "05_오차증분_분포적합",
        "06_시나리오생성",
        "07_리스크관리_입찰곡선최적화",
    }
    if not base.exists() or not base.is_dir():
        return False
    names = {p.name for p in base.iterdir() if p.is_dir()}
    return required.issubset(names)


def _find_project_root(anchor_path: Path) -> Path:
    for cand in [anchor_path.parent if anchor_path.is_file() else anchor_path, *(anchor_path.parent if anchor_path.is_file() else anchor_path).parents]:
        if _is_flat_project_root(cand):
            return cand
    for cand in _candidate_project_roots(anchor_path):
        if _is_flat_project_root(cand):
            return cand
    raise FileNotFoundError(f"Could not locate 8-folder project root from: {anchor_path}")


def _find_first_file(folder: Path, *keywords: str, suffix: str | None = None) -> Path:
    files = sorted(path for path in folder.rglob("*") if path.is_file())
    for path in files:
        name = path.name
        if suffix is not None and path.suffix.lower() != suffix.lower():
            continue
        if all(keyword in name for keyword in keywords):
            return path
    raise FileNotFoundError(f"No file found in {folder} with keywords={keywords} suffix={suffix}")


def _maybe_find_first_file(folder: Path, *keywords: str, suffix: str | None = None) -> Path | None:
    try:
        return _find_first_file(folder, *keywords, suffix=suffix)
    except FileNotFoundError:
        return None


def resolve_project_roots(anchor: Path | str | None = None) -> dict[str, Path]:
    anchor_path = Path(anchor) if anchor is not None else Path(__file__).resolve()
    code_root = anchor_path.parent if anchor_path.is_file() else anchor_path
    project_root = _find_project_root(anchor_path)
    return {
        "project_root": project_root,
        "code_root": code_root,
        "input_root": project_root / "01_입력데이터",
        "prep_root": project_root / "02_입력데이터가공",
        "point_root": project_root / "03_점예측",
        "error_root": project_root / "04_오차데이터",
        "fit_root": project_root / "05_오차증분_분포적합",
        "scenario_root": project_root / "06_시나리오생성",
        "opt_root": project_root / "07_리스크관리_입찰곡선최적화",
        "analysis_root": project_root / "08_결과확인_데이터분석",
    }


def official_scenario_root(anchor: Path | str | None = None) -> Path:
    roots = resolve_project_roots(anchor)
    # The reproduction package keeps only the final S5000 family roots.
    for name in ("시나리오생성결과_s5000_laplace", "시나리오생성결과_s5000_gaussian"):
        candidate = roots["scenario_root"] / name
        if candidate.is_dir():
            return candidate
    return roots["scenario_root"]


def default_source_roots(anchor: Path | str | None = None) -> dict[str, Path]:
    roots = resolve_project_roots(anchor)
    error_root = roots["error_root"]
    official_error_root = error_root / "오차데이터결과"
    error_search_root = official_error_root if official_error_root.is_dir() else error_root
    point_root = roots["point_root"]
    scenario_root = official_scenario_root(anchor)
    opt_root = roots["opt_root"]
    analysis_root = roots["analysis_root"]
    wind_error_csv = _maybe_find_first_file(error_search_root, "풍력", "오차시계열", suffix=".csv")
    if wind_error_csv is None:
        wind_error_csv = _maybe_find_first_file(point_root, "테스트기간", "풍력발전", "예측결과", suffix=".csv")
    if wind_error_csv is None:
        wind_error_csv = _find_first_file(point_root, "풍력", "예측결과", suffix=".csv")
    return {
        "out_root": opt_root,
        "wind_error_csv": wind_error_csv,
        "da_energy_error_csv": _find_first_file(error_search_root, "DA에너지", "오차시계열", suffix=".csv"),
        "rt_energy_error_csv": _find_first_file(error_search_root, "RT에너지", "오차시계열", suffix=".csv"),
        "da_reg_error_csv": _find_first_file(error_search_root, "DA_REG", "오차시계열", suffix=".csv"),
        "rt_reg_error_csv": _find_first_file(error_search_root, "RT_REG", "오차시계열", suffix=".csv"),
        "wind_scenario_root": scenario_root / "__풍력시나리오__",
        "da_energy_scenario_root": scenario_root / "__DA에너지시나리오__",
        "rt_energy_scenario_root": scenario_root / "__RT에너지시나리오__",
        "spread_scenario_root": scenario_root / "__스프레드시나리오__",
        "energy_joint_scenario_root": scenario_root / "__DA_RT에너지결합시나리오__",
        "reg_joint_scenario_root": scenario_root / "__DA_REG_RT_REG결합시나리오__",
        "reg_m_root": opt_root,
        "reg_point_forecast_root": point_root,
        "reg_residual_conditional_root": opt_root,
        "analysis_root": analysis_root,
    }


def family_case_dir(case_name: str) -> str:
    key = str(case_name).strip().lower()
    return _FAMILY_CASE_DIRS.get(key, f"분포사례_{key}")


def energy_joint_case_dir(case_name: str) -> str:
    return f"joint_case_{str(case_name).strip().lower()}"


def reg_joint_case_dir(case_name: str) -> str:
    return f"joint_case_{str(case_name).strip().lower()}"


def _coerce_time_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "timestamp_utc" in out.columns:
        out["timestamp_utc"] = pd.to_datetime(out["timestamp_utc"], utc=True, errors="coerce")
    elif "timestamp_et" in out.columns:
        out["timestamp_et"] = pd.to_datetime(out["timestamp_et"], errors="coerce")
        out["timestamp_utc"] = out["timestamp_et"].dt.tz_convert("UTC")
    elif {"operating_date_et", "hour_et"}.issubset(out.columns):
        dt = pd.to_datetime(out["operating_date_et"], errors="coerce") + pd.to_timedelta(pd.to_numeric(out["hour_et"], errors="coerce"), unit="h")
        out["timestamp_utc"] = dt.dt.tz_localize(TZ_ET, nonexistent="shift_forward", ambiguous="NaT").dt.tz_convert("UTC")
    else:
        raise ValueError("Could not infer time columns; need timestamp_utc or operating_date_et/hour_et.")
    out = out.dropna(subset=["timestamp_utc"]).copy()
    ts_et = out["timestamp_utc"].dt.tz_convert(TZ_ET)
    if "timestamp_et" not in out.columns:
        out["timestamp_et"] = ts_et.astype(str)
    if "operating_date_et" not in out.columns:
        out["operating_date_et"] = ts_et.dt.strftime("%Y-%m-%d")
    else:
        out["operating_date_et"] = out["operating_date_et"].astype(str)
    if "hour_et" not in out.columns:
        out["hour_et"] = ts_et.dt.hour.astype(int)
    else:
        out["hour_et"] = pd.to_numeric(out["hour_et"], errors="coerce").astype("Int64")
        out = out.dropna(subset=["hour_et"]).copy()
        out["hour_et"] = out["hour_et"].astype(int)
    return out


def _date_mask(df: pd.DataFrame, start: str | None = None, end: str | None = None) -> pd.Series:
    dt = pd.to_datetime(df["operating_date_et"], errors="coerce")
    mask = pd.Series(True, index=df.index)
    if start:
        mask &= dt >= pd.Timestamp(start)
    if end:
        mask &= dt <= pd.Timestamp(end)
    return mask


def load_hourly_actual(csv_path: Path | str, source_col: str, target_name: str, start: str | None = None, end: str | None = None) -> pd.DataFrame:
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    df = _coerce_time_columns(df)
    if source_col not in df.columns:
        raise KeyError(f"Column {source_col!r} not found in {csv_path}")
    out_cols = ["timestamp_utc", "timestamp_et", "operating_date_et", "hour_et"]
    if "split" in df.columns:
        out_cols.append("split")
    out = df.loc[:, out_cols].copy()
    out[target_name] = pd.to_numeric(df[source_col], errors="coerce")
    out = out.dropna(subset=[target_name]).copy()
    out = out[_date_mask(out, start, end)].copy()
    out = out.sort_values(["operating_date_et", "hour_et", "timestamp_utc"]).drop_duplicates(subset=["timestamp_utc"], keep="last")
    return out.reset_index(drop=True)


def _load_wind_actual_bundle(csv_path: Path | str, start: str | None = None, end: str | None = None) -> pd.DataFrame:
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    df = _coerce_time_columns(df)
    actual_col = "y_true_raw_mw" if "y_true_raw_mw" in df.columns else "y_true_price"
    pred_col = "y_pred_raw_mw" if "y_pred_raw_mw" in df.columns else "y_pred_price"
    out = df[["timestamp_utc", "timestamp_et", "operating_date_et", "hour_et"]].copy()
    out["wind_actual_mw"] = pd.to_numeric(df[actual_col], errors="coerce")
    out["wind_point_forecast_mw"] = pd.to_numeric(df[pred_col], errors="coerce")
    out["capacity_mw"] = pd.to_numeric(df["capacity_mw"], errors="coerce") if "capacity_mw" in df.columns else CAPACITY_MW
    out = out.dropna(subset=["wind_actual_mw", "wind_point_forecast_mw"]).copy()
    out = out[_date_mask(out, start, end)].copy()
    out = out.sort_values(["operating_date_et", "hour_et", "timestamp_utc"]).drop_duplicates(subset=["timestamp_utc"], keep="last")
    return out.reset_index(drop=True)


def build_validation_actual_bundle(
    wind_error_csv: Path | str,
    da_energy_error_csv: Path | str,
    rt_energy_error_csv: Path | str,
    da_reg_error_csv: Path | str,
    rt_reg_error_csv: Path | str,
    start: str,
    end: str,
) -> pd.DataFrame:
    wind = _load_wind_actual_bundle(wind_error_csv, start=start, end=end)
    da = load_hourly_actual(da_energy_error_csv, "y_true_price", "da_energy_actual", start=start, end=end)
    rt = load_hourly_actual(rt_energy_error_csv, "y_true_price", "rt_energy_actual", start=start, end=end)
    da_reg = load_hourly_actual(da_reg_error_csv, "y_true_price", "da_reg_actual", start=start, end=end)
    rt_reg = load_hourly_actual(rt_reg_error_csv, "y_true_price", "rt_reg_actual", start=start, end=end)
    out = wind.merge(da[["timestamp_utc", "da_energy_actual"]], on="timestamp_utc", how="inner")
    out = out.merge(rt[["timestamp_utc", "rt_energy_actual"]], on="timestamp_utc", how="inner")
    out = out.merge(da_reg[["timestamp_utc", "da_reg_actual"]], on="timestamp_utc", how="inner")
    out = out.merge(rt_reg[["timestamp_utc", "rt_reg_actual"]], on="timestamp_utc", how="inner")
    return out.sort_values(["operating_date_et", "hour_et", "timestamp_utc"]).reset_index(drop=True)


def _day_partition_root(root: Path | str) -> Path:
    root = Path(root)
    for name in ("joint_day_parts", "scenario_day_parts", "시나리오_일별분할"):
        candidate = root / name
        if candidate.exists():
            return candidate
    return root


def _is_flat_scenario_hint(root: Path) -> bool:
    if (root / "joint_day_parts").exists() or (root / "scenario_day_parts").exists() or (root / "시나리오_일별분할").exists():
        return False
    return any(part.startswith("__") and part.endswith("__") for part in root.parts)


def _scenario_mapping_csv(root: Path) -> Path:
    for cand in [root, *root.parents]:
        if cand.name == "06_시나리오생성":
            return cand / "99_색인01_파일이름대응표.csv"
    project_root = resolve_project_roots(root)["project_root"]
    return project_root / "06_시나리오생성" / "99_색인01_파일이름대응표.csv"


def _load_scenario_mapping(root: Path) -> pd.DataFrame:
    csv_path = _scenario_mapping_csv(root)
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    df["원래경로"] = df["원래경로"].astype(str)
    df["새파일명"] = df["새파일명"].astype(str)
    return df


def _scenario_case_key(case_dir: str) -> str:
    low = case_dir.lower()
    if "mixed_selective_jump_qmap" in low:
        return "mixed_selective_jump_qmap"
    if "mixed_selective_jump" in low:
        return "mixed_selective_jump"
    if "stable_qmap" in low:
        return "stable_qmap"
    if "stable" in low or "stable" in case_dir:
        return "stable"
    if "levy_jump_gbody" in low:
        return "levy_jump_gbody"
    if "levy_jump_lbody" in low:
        return "levy_jump_lbody"
    if "gaussian" in low or "가우시안" in case_dir:
        return "gaussian"
    if "laplace" in low or "라플라스" in case_dir:
        return "laplace"
    if "skewed_t" in low or "skewedt" in low:
        return "skewed_t"
    if "levy_jump" in low or "레비점프" in case_dir:
        return "levy_jump"
    return ""


def _flat_scenario_rows(root: Path) -> pd.DataFrame:
    # Final S5000 inputs are kept as ordinary date-partitioned parquet trees.
    # A flat filename-map adapter is intentionally unsupported in this package.
    return pd.DataFrame()


def _list_partition_dates(root: Path | str) -> list[str]:
    root = Path(root)
    if _is_flat_scenario_hint(root):
        rows = _flat_scenario_rows(root)
        if rows.empty:
            return []
        return sorted(rows["operating_date_et"].dropna().astype(str).unique().tolist())
    day_root = _day_partition_root(root)
    if not day_root.exists():
        return []
    dates: list[str] = []
    for child in day_root.iterdir():
        if child.is_dir() and child.name.startswith("operating_date_et="):
            dates.append(child.name.split("=", 1)[1])
    return sorted(set(dates))


def list_available_dates(root: Path | str) -> list[str]:
    return _list_partition_dates(root)


def list_available_energy_joint_dates(root: Path | str) -> list[str]:
    return _list_partition_dates(root)


def list_available_reg_joint_dates(root: Path | str) -> list[str]:
    return _list_partition_dates(root)


def list_available_3block_joint_dates(
    wind_root: Path | str,
    energy_root: Path | str,
) -> list[str]:
    wind_dates = set(_list_partition_dates(wind_root))
    energy_dates = set(_list_partition_dates(energy_root))
    return sorted(wind_dates & energy_dates)


def _read_parquet_robust(path: Path | str) -> pd.DataFrame:
    path = Path(path)
    try:
        return pd.read_parquet(path)
    except Exception as exc:
        if "Repetition level histogram size mismatch" not in str(exc):
            raise
        try:
            return pd.read_parquet(path, engine="fastparquet")
        except Exception:
            raise exc


def _load_day_frame(root: Path | str, operating_date_et: str) -> pd.DataFrame:
    root = Path(root)
    if _is_flat_scenario_hint(root):
        rows = _flat_scenario_rows(root)
        rows = rows.loc[rows["operating_date_et"].astype(str) == str(operating_date_et)].copy()
        if rows.empty:
            raise FileNotFoundError(f"Missing flat scenario file for {root} date={operating_date_et}")
        part_path = Path(rows.sort_values("새파일명").iloc[0]["새경로"])
        if part_path.suffix.lower() == ".parquet":
            df = _read_parquet_robust(part_path)
        else:
            df = pd.read_csv(part_path, encoding="utf-8-sig")
    else:
        day_root = _day_partition_root(root)
        day_dir = day_root / f"operating_date_et={operating_date_et}"
        if not day_dir.exists():
            raise FileNotFoundError(f"Missing day directory: {day_dir}")
        parquet_files = sorted(day_dir.glob("*.parquet"))
        csv_files = sorted(day_dir.glob("*.csv"))
        if parquet_files:
            df = _read_parquet_robust(parquet_files[0])
        elif csv_files:
            df = pd.read_csv(csv_files[0], encoding="utf-8-sig")
        else:
            raise FileNotFoundError(f"No parquet/csv part found under {day_dir}")
    if "joint_scenario_id" in df.columns and "scenario_id" not in df.columns:
        df = df.rename(columns={"joint_scenario_id": "scenario_id"})
    elif "scenario_id" not in df.columns:
        for candidate in ["wind_scenario_id", "da_scenario_id", "rt_scenario_id", "spread_scenario_id", "reg_scenario_id"]:
            if candidate in df.columns:
                df = df.rename(columns={candidate: "scenario_id"})
                break
    if "timestamp_utc" in df.columns:
        df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True, errors="coerce")
    if "hour_et" in df.columns:
        df["hour_et"] = pd.to_numeric(df["hour_et"], errors="coerce").astype("Int64")
        df = df.dropna(subset=["hour_et"]).copy()
        df["hour_et"] = df["hour_et"].astype(int)
    df["operating_date_et"] = str(operating_date_et)
    return df.reset_index(drop=True)


def load_wind_scenario_day(root: Path | str, operating_date_et: str) -> pd.DataFrame:
    return _load_day_frame(root, operating_date_et)


def load_da_energy_scenario_day(root: Path | str, operating_date_et: str) -> pd.DataFrame:
    return _load_day_frame(root, operating_date_et)


def load_hourly_price_scenario_day(root: Path | str, operating_date_et: str, prefix: str = "rt_energy") -> pd.DataFrame:
    return _load_day_frame(root, operating_date_et)


def load_energy_joint_scenario_day(root: Path | str, operating_date_et: str) -> pd.DataFrame:
    return _load_day_frame(root, operating_date_et)


def load_3block_wind_energy_day(
    wind_root: Path | str,
    energy_root: Path | str,
    operating_date_et: str,
) -> pd.DataFrame:
    wind_df = _load_day_frame(wind_root, operating_date_et)
    energy_df = _load_day_frame(energy_root, operating_date_et)
    merge_keys = [
        col for col in [
            "operating_date_et",
            "hour_et",
            "timestamp_utc",
            "scenario_id",
            "joint_scenario_id",
            "issue_timestamp_utc",
            "issue_timestamp_et",
        ]
        if col in wind_df.columns and col in energy_df.columns
    ]
    if not merge_keys:
        raise KeyError("Could not determine merge keys for wind-energy joint scenario load.")
    merged = wind_df.merge(
        energy_df,
        on=merge_keys,
        how="inner",
        validate="one_to_one",
    )
    return merged.reset_index(drop=True)


def load_reg_joint_penalty_day(root: Path | str, operating_date_et: str) -> pd.DataFrame:
    df = _load_day_frame(root, operating_date_et)
    if "penalty_price_scn" not in df.columns:
        da_col = "da_reg_scn" if "da_reg_scn" in df.columns else None
        rt_col = "rt_reg_scn" if "rt_reg_scn" in df.columns else None
        if da_col and rt_col:
            da_vals = pd.to_numeric(df[da_col], errors="coerce").fillna(0.0)
            rt_vals = pd.to_numeric(df[rt_col], errors="coerce").fillna(0.0)
            df["penalty_price_scn"] = np.maximum(da_vals, rt_vals)
        else:
            raise KeyError("reg joint scenario day does not contain penalty_price_scn or DA/RT REG scenario columns.")
    return df


def extract_hourly_scenario_delta(day_df: pd.DataFrame, hour_et: int, value_col: str) -> pd.DataFrame:
    if "scenario_id" not in day_df.columns:
        raise KeyError("Scenario dataframe must contain scenario_id.")
    cur = day_df.loc[day_df["hour_et"] == int(hour_et), ["scenario_id", value_col]].copy()
    cur[value_col] = pd.to_numeric(cur[value_col], errors="coerce")
    cur = cur.dropna(subset=[value_col]).sort_values("scenario_id").reset_index(drop=True)
    if cur.empty:
        return pd.DataFrame(columns=["scenario_id", "hour_et", "delta_value"])
    if int(hour_et) <= 0:
        return pd.DataFrame({"scenario_id": cur["scenario_id"].to_numpy(dtype=np.int64), "hour_et": int(hour_et), "delta_value": np.zeros(len(cur), dtype=np.float64)})
    prev = day_df.loc[day_df["hour_et"] == int(hour_et) - 1, ["scenario_id", value_col]].copy()
    prev[value_col] = pd.to_numeric(prev[value_col], errors="coerce")
    prev = prev.dropna(subset=[value_col]).rename(columns={value_col: "prev_value"})
    merged = cur.merge(prev, on="scenario_id", how="left")
    merged["delta_value"] = merged[value_col].to_numpy(dtype=np.float64) - merged["prev_value"].fillna(merged[value_col]).to_numpy(dtype=np.float64)
    return merged[["scenario_id"]].assign(hour_et=int(hour_et), delta_value=merged["delta_value"].to_numpy(dtype=np.float64))


def _load_reg_export_frame(path: Path, split: str, actual_col: str, pred_col: str) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    rename_map = {"TimeStamp": "timestamp_utc", "실제값": actual_col, "예측값": pred_col}
    df = df.rename(columns=rename_map)
    required = ["timestamp_utc", actual_col, pred_col]
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise ValueError(f"Missing REG export columns in {path}: {missing}")
    df = df.loc[:, required].copy()
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True, errors="coerce")
    df[actual_col] = pd.to_numeric(df[actual_col], errors="coerce")
    df[pred_col] = pd.to_numeric(df[pred_col], errors="coerce")
    df = df.dropna(subset=["timestamp_utc"]).copy()
    df["split"] = str(split)
    return df


def _load_reg_point_forecast_predictions_from_final_exports(root: Path) -> pd.DataFrame:
    da_train = _find_first_file(root, "07_", "하루전", "REG가격", "예측결과", suffix=".csv")
    da_test = _find_first_file(root, "08_", "하루전", "REG가격", "예측결과", suffix=".csv")
    rt_train = _find_first_file(root, "09_", "실시간", "REG가격", "예측결과", suffix=".csv")
    rt_test = _find_first_file(root, "10_", "실시간", "REG가격", "예측결과", suffix=".csv")

    da_df = pd.concat(
        [
            _load_reg_export_frame(da_train, "train", "da_reg_actual", "da_reg_pred"),
            _load_reg_export_frame(da_test, "test", "da_reg_actual", "da_reg_pred"),
        ],
        ignore_index=True,
    )
    rt_df = pd.concat(
        [
            _load_reg_export_frame(rt_train, "train", "rt_reg_actual", "rt_reg_pred"),
            _load_reg_export_frame(rt_test, "test", "rt_reg_actual", "rt_reg_pred"),
        ],
        ignore_index=True,
    )
    merged = da_df.merge(
        rt_df,
        on=["timestamp_utc", "split"],
        how="inner",
        validate="one_to_one",
    )
    merged = _coerce_time_columns(merged)
    return merged.sort_values(["operating_date_et", "hour_et", "timestamp_utc"]).reset_index(drop=True)

def load_reg_point_forecast_predictions(root: Path | str, start: str | None = None, end: str | None = None) -> pd.DataFrame:
    root = Path(root)
    if root.is_file():
        path = root
    else:
        try:
            path = _find_first_file(root, "13_DA_REG_RT_REG_원시예측결과_전체", suffix=".csv")
        except FileNotFoundError:
            try:
                path = _find_first_file(root, "DA_REG_RT_REG_원시예측결과_전체", suffix=".csv")
            except FileNotFoundError:
                try:
                    path = _find_first_file(root, "reg_point_forecast_predictions_all", suffix=".csv")
                except FileNotFoundError:
                    try:
                        path = _find_first_file(root, "REG점예측모형", "predictions_all", suffix=".csv")
                    except FileNotFoundError:
                        df = _load_reg_point_forecast_predictions_from_final_exports(root)
                        df = df[_date_mask(df, start, end)].copy()
                        for col in ["da_reg_actual", "rt_reg_actual", "da_reg_pred", "rt_reg_pred"]:
                            if col in df.columns:
                                df[col] = pd.to_numeric(df[col], errors="coerce")
                        return df.sort_values(["operating_date_et", "hour_et", "timestamp_utc"]).reset_index(drop=True)
    df = pd.read_csv(path, encoding="utf-8-sig")
    df = _coerce_time_columns(df)
    df = df[_date_mask(df, start, end)].copy()
    for col in ["da_reg_actual", "rt_reg_actual", "da_reg_pred", "rt_reg_pred"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.sort_values(["operating_date_et", "hour_et", "timestamp_utc"]).reset_index(drop=True)


def load_reg_residual_conditional_model(root: Path | str) -> dict[str, object]:
    root = Path(root)
    if root.is_dir():
        try:
            meta_path = _find_first_file(root, "reg_residual_conditional", "메타정보", suffix=".json")
        except FileNotFoundError:
            meta_path = _find_first_file(root, "reg_residual_conditional_meta", suffix=".json")
    else:
        meta_path = root
    root_dir = meta_path.parent
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    def _maybe(name: str) -> pd.DataFrame:
        if root_dir.is_dir():
            for path in sorted(root_dir.rglob("*")):
                if path.is_file() and name in path.name:
                    return pd.read_csv(path, encoding="utf-8-sig")
        return pd.DataFrame()
    artifacts = {
        "root": root_dir,
        "meta": meta,
        "abs_delta_da_bin_edges": np.asarray(meta.get("abs_delta_da_bin_edges", []), dtype=np.float64),
        "month_hour_delta": _maybe("month_hour_delta_pair_pool.csv"),
        "hour_delta": _maybe("hour_delta_pair_pool.csv"),
        "hour": _maybe("hour_pair_pool.csv"),
        "global": _maybe("global_pair_pool.csv"),
        "cell_stats": _maybe("cell_stats.csv"),
    }
    for key in ["month_hour_delta", "hour_delta", "hour", "global"]:
        df = artifacts[key]
        if not df.empty:
            for col in ["month", "hour_et", "abs_delta_da_bin_idx"]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
            for col in ["eps_da_reg", "eps_rt_reg"]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
            artifacts[key] = df.dropna(subset=["eps_da_reg", "eps_rt_reg"]).reset_index(drop=True)
    return artifacts


def _sample_pair_pool(pool: pd.DataFrame, n_samples: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    if pool.empty:
        return np.zeros(n_samples, dtype=np.float64), np.zeros(n_samples, dtype=np.float64)
    idx = rng.integers(0, len(pool), size=int(n_samples))
    sampled = pool.iloc[idx]
    return sampled["eps_da_reg"].to_numpy(dtype=np.float64), sampled["eps_rt_reg"].to_numpy(dtype=np.float64)


def sample_reg_residual_pair_hour(
    artifacts: dict[str, object],
    operating_date_et: str,
    hour_et: int,
    delta_da_values: np.ndarray,
    base_seed: int = 777,
) -> tuple[np.ndarray, np.ndarray]:
    delta_da_values = np.asarray(delta_da_values, dtype=np.float64)
    n = len(delta_da_values)
    if n == 0:
        return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)
    month = int(pd.Timestamp(str(operating_date_et)).month)
    hour_et = int(hour_et)
    edges = np.asarray(artifacts.get("abs_delta_da_bin_edges", []), dtype=np.float64)
    if edges.size < 2:
        bins = np.zeros(n, dtype=np.int64)
    else:
        bins = np.searchsorted(edges, np.abs(delta_da_values), side="right") - 1
        bins = np.clip(bins, 0, len(edges) - 2).astype(np.int64)
    month_hour_delta = artifacts.get("month_hour_delta", pd.DataFrame())
    hour_delta = artifacts.get("hour_delta", pd.DataFrame())
    hour_pool = artifacts.get("hour", pd.DataFrame())
    global_pool = artifacts.get("global", pd.DataFrame())
    eps_da = np.zeros(n, dtype=np.float64)
    eps_rt = np.zeros(n, dtype=np.float64)
    rng = deterministic_rng(base_seed, operating_date_et, hour_et, "reg_residual_pair")
    for bin_idx in np.unique(bins):
        mask = bins == int(bin_idx)
        pool = pd.DataFrame()
        if not month_hour_delta.empty:
            pool = month_hour_delta.loc[(month_hour_delta["month"] == month) & (month_hour_delta["hour_et"] == hour_et) & (month_hour_delta["abs_delta_da_bin_idx"] == int(bin_idx))]
        if pool.empty and not hour_delta.empty:
            pool = hour_delta.loc[(hour_delta["hour_et"] == hour_et) & (hour_delta["abs_delta_da_bin_idx"] == int(bin_idx))]
        if pool.empty and not hour_pool.empty:
            pool = hour_pool.loc[hour_pool["hour_et"] == hour_et]
        if pool.empty:
            pool = global_pool
        da_s, rt_s = _sample_pair_pool(pool, int(mask.sum()), rng)
        eps_da[mask] = da_s
        eps_rt[mask] = rt_s
    return eps_da, eps_rt


def load_reg_penalty_M_model(root: Path | str) -> dict[str, object]:
    root = Path(root)
    meta_path = _find_first_file(root, "reg_penalty_M", "메타정보", suffix=".json") if root.is_dir() else root
    root_dir = meta_path.parent if meta_path.exists() else root
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    def _maybe(name: str) -> pd.DataFrame:
        if root_dir.is_dir():
            for path in sorted(root_dir.rglob("*")):
                if path.is_file() and name in path.name:
                    return pd.read_csv(path, encoding="utf-8-sig")
        return pd.DataFrame()
    return {
        "root": root_dir,
        "meta": meta,
        "cell_stats": _maybe("cell_stats.csv"),
        "month_hour_body": _maybe("hour_month_body_pool.csv"),
        "month_hour_spike": _maybe("hour_month_spike_pool.csv"),
        "hour_body": _maybe("hour_body_pool.csv"),
        "hour_spike": _maybe("hour_spike_pool.csv"),
        "global_body": _maybe("global_body_pool.csv"),
        "global_spike": _maybe("global_spike_pool.csv"),
    }


def sample_reg_penalty_M_hour(
    artifacts: dict[str, object],
    operating_date_et: str,
    hour_et: int,
    n_samples: int,
    base_seed: int = 777,
) -> np.ndarray:
    month = int(pd.Timestamp(str(operating_date_et)).month)
    hour_et = int(hour_et)
    n_samples = int(n_samples)
    rng = deterministic_rng(base_seed, operating_date_et, hour_et, "reg_penalty_M")
    cell_stats = artifacts.get("cell_stats", pd.DataFrame())
    row = pd.DataFrame()
    if not cell_stats.empty:
        row = cell_stats.loc[(pd.to_numeric(cell_stats.get("month"), errors="coerce") == month) & (pd.to_numeric(cell_stats.get("hour_et"), errors="coerce") == hour_et)]
    pi_spike = 0.05
    if not row.empty and "pi_spike" in row.columns:
        try:
            pi_spike = float(row["pi_spike"].iloc[0])
        except Exception:
            pi_spike = 0.05
    body = artifacts.get("month_hour_body", pd.DataFrame())
    spike = artifacts.get("month_hour_spike", pd.DataFrame())
    if not body.empty:
        body = body.loc[(pd.to_numeric(body.get("month"), errors="coerce") == month) & (pd.to_numeric(body.get("hour_et"), errors="coerce") == hour_et)]
    if not spike.empty:
        spike = spike.loc[(pd.to_numeric(spike.get("month"), errors="coerce") == month) & (pd.to_numeric(spike.get("hour_et"), errors="coerce") == hour_et)]
    if body.empty:
        body = artifacts.get("hour_body", pd.DataFrame())
        if not body.empty:
            body = body.loc[pd.to_numeric(body.get("hour_et"), errors="coerce") == hour_et]
    if spike.empty:
        spike = artifacts.get("hour_spike", pd.DataFrame())
        if not spike.empty:
            spike = spike.loc[pd.to_numeric(spike.get("hour_et"), errors="coerce") == hour_et]
    if body.empty:
        body = artifacts.get("global_body", pd.DataFrame())
    if spike.empty:
        spike = artifacts.get("global_spike", pd.DataFrame())
    body_vals = pd.to_numeric(body.get("m_value", pd.Series(dtype=float)), errors="coerce").dropna().to_numpy(dtype=np.float64)
    spike_vals = pd.to_numeric(spike.get("m_value", pd.Series(dtype=float)), errors="coerce").dropna().to_numpy(dtype=np.float64)
    if len(body_vals) == 0 and len(spike_vals) == 0:
        return np.zeros(n_samples, dtype=np.float64)
    if len(body_vals) == 0:
        body_vals = spike_vals.copy()
    if len(spike_vals) == 0:
        spike_vals = body_vals.copy()
    flags = rng.random(n_samples) < float(np.clip(pi_spike, 0.0, 1.0))
    out = np.empty(n_samples, dtype=np.float64)
    out[~flags] = body_vals[rng.integers(0, len(body_vals), size=int((~flags).sum()))]
    out[flags] = spike_vals[rng.integers(0, len(spike_vals), size=int(flags.sum()))]
    return out


def deterministic_rng(seed: int, *parts: object) -> np.random.Generator:
    key = "|".join([str(seed)] + [str(x) for x in parts]).encode("utf-8")
    digest = hashlib.sha256(key).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "little", signed=False))


def regularize_corr(corr: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    corr = np.asarray(corr, dtype=np.float64)
    if corr.ndim != 2 or corr.shape[0] != corr.shape[1]:
        raise ValueError("corr must be a square matrix")
    sym = 0.5 * (corr + corr.T)
    np.fill_diagonal(sym, 1.0)
    vals, vecs = np.linalg.eigh(sym)
    vals = np.clip(vals, eps, None)
    repaired = vecs @ np.diag(vals) @ vecs.T
    d = np.sqrt(np.diag(repaired))
    repaired = repaired / np.outer(d, d)
    np.fill_diagonal(repaired, 1.0)
    return repaired


def couple_marginals(marginals: list[np.ndarray], corr: np.ndarray | None, rng: np.random.Generator) -> tuple[np.ndarray, ...]:
    arrays = []
    for values in marginals:
        arr = np.asarray(values, dtype=np.float64)
        arr = arr[np.isfinite(arr)]
        if len(arr) == 0:
            raise ValueError("All marginal arrays must contain at least one finite value.")
        arrays.append(arr)
    n = min(len(arr) for arr in arrays)
    sampled = []
    for arr in arrays:
        if len(arr) > n:
            idx = rng.choice(len(arr), size=n, replace=False)
            sampled.append(arr[idx])
        else:
            sampled.append(arr.copy())
    dim = len(sampled)
    if corr is None:
        return tuple(np.asarray(x[:n], dtype=np.float64) for x in sampled)
    corr = regularize_corr(np.asarray(corr, dtype=np.float64))
    z = rng.multivariate_normal(np.zeros(dim, dtype=np.float64), corr, size=n)
    outputs: list[np.ndarray] = []
    for j, arr in enumerate(sampled):
        order = np.argsort(z[:, j], kind="mergesort")
        values = np.sort(np.asarray(arr[:n], dtype=np.float64))
        out = np.empty(n, dtype=np.float64)
        out[order] = values
        outputs.append(out)
    return tuple(outputs)


def make_price_knots_from_samples(values: np.ndarray, n_knots: int = 7) -> np.ndarray:
    vals = np.asarray(values, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return np.linspace(0.0, 1.0, int(n_knots), dtype=np.float64)
    if int(n_knots) <= 1:
        return np.asarray([float(np.median(vals))], dtype=np.float64)
    q = np.quantile(vals, np.linspace(0.0, 1.0, int(n_knots)))
    q = np.asarray(q, dtype=np.float64)
    if np.allclose(q[0], q[-1]):
        centre = float(q[0])
        q = np.linspace(centre - 1e-3, centre + 1e-3, int(n_knots), dtype=np.float64)
    step = max(1e-6, 1e-6 * max(1.0, float(np.nanmax(np.abs(q)))))
    for idx in range(1, len(q)):
        if q[idx] <= q[idx - 1]:
            q[idx] = q[idx - 1] + step
    return q


def _price_interpolation_weights(values: np.ndarray, p_knots: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64)
    p = np.asarray(p_knots, dtype=np.float64)
    n = len(x)
    k = len(p)
    w = np.zeros((n, k), dtype=np.float64)
    if k == 1:
        w[:, 0] = 1.0
        return w
    for i, val in enumerate(x):
        if val <= p[0]:
            w[i, 0] = 1.0
            continue
        if val >= p[-1]:
            w[i, -1] = 1.0
            continue
        right = int(np.searchsorted(p, val, side="right"))
        left = max(0, right - 1)
        denom = p[right] - p[left]
        if denom <= 0:
            w[i, left] = 1.0
            continue
        frac = (val - p[left]) / denom
        w[i, left] = 1.0 - frac
        w[i, right] = frac
    return w


def award_quantity_from_curve(q_knots: np.ndarray, p_knots: np.ndarray, da_price: float) -> float:
    q = np.asarray(q_knots, dtype=np.float64)
    p = np.asarray(p_knots, dtype=np.float64)
    return float(np.interp(float(da_price), p, q, left=q[0], right=q[-1]))


def normalize_penalty_settlement_mode(mode: str | bool | None = "enabled") -> str:
    if isinstance(mode, bool):
        return "enabled" if mode else "none"
    key = "enabled" if mode is None else str(mode).strip().lower()
    aliases_enabled = {"enabled", "enable", "on", "yes", "true", "with_penalty", "penalty"}
    aliases_none = {"none", "disabled", "disable", "off", "no", "false", "no_penalty", "without_penalty"}
    if key in aliases_enabled:
        return "enabled"
    if key in aliases_none:
        return "none"
    raise ValueError(
        f"Unsupported penalty_settlement_mode: {mode!r}. "
        "Use 'enabled' or 'none'."
    )


def settlement_amount(
    award_mw: float,
    actual_wind_mw: float,
    da_energy_price: float,
    rt_energy_price: float,
    da_reg_price: float,
    rt_reg_price: float,
    penalty_band_basis: str = "award",
    capacity_mw: float = CAPACITY_MW,
    penalty_settlement_mode: str | bool | None = "enabled",
) -> dict[str, object]:
    award = float(award_mw)
    actual = float(actual_wind_mw)
    da = float(da_energy_price)
    rt = float(rt_energy_price)
    penalty_mode = normalize_penalty_settlement_mode(penalty_settlement_mode)
    penalty_enabled = penalty_mode == "enabled"
    penalty_price = float(max(0.0, da_reg_price, rt_reg_price)) if penalty_enabled else 0.0
    basis = str(penalty_band_basis).strip().lower()
    if basis == "capacity":
        reference = float(capacity_mw)
        tolerance = 0.03 * float(capacity_mw)
    else:
        reference = float(award)
        tolerance = 0.03 * float(award)
    over_generation = max(actual - award, 0.0)
    under_generation = max(award - actual, 0.0)
    over_penalty_mw = max(actual - award - tolerance, 0.0) if penalty_enabled else 0.0
    under_penalty_mw = max(award - actual - tolerance, 0.0) if penalty_enabled else 0.0
    over_penalty_amount = penalty_price * over_penalty_mw
    under_penalty_amount = penalty_price * under_penalty_mw
    penalty_amount = over_penalty_amount + under_penalty_amount
    da_settle = award * da
    rt_settle = (actual - award) * rt
    penalty_settle = -penalty_amount
    total = da_settle + rt_settle + penalty_settle
    return {
        "settlement_amount": float(total),
        "da_settlement_amount": float(da_settle),
        "rt_settlement_amount": float(rt_settle),
        "penalty_settlement_amount": float(penalty_settle),
        "penalty_trigger": bool(penalty_amount > 0.0),
        "penalty_amount": float(penalty_amount),
        "penalty_price": float(penalty_price),
        "penalty_settlement_mode": penalty_mode,
        "penalty_band_basis": basis,
        "penalty_band_reference_mw": float(reference),
        "operator_output_limit_mw": np.nan,
        "tolerance_mw": float(tolerance),
        "over_generation_mw": float(over_generation),
        "under_generation_mw": float(under_generation),
        "over_penalty_trigger": bool(over_penalty_mw > 0.0),
        "under_penalty_trigger": bool(under_penalty_mw > 0.0),
        "over_penalty_mw": float(over_penalty_mw),
        "under_penalty_mw": float(under_penalty_mw),
        "over_penalty_amount": float(over_penalty_amount),
        "under_penalty_amount": float(under_penalty_amount),
    }

def _evaluate_piecewise_solution(
    q_knots: np.ndarray,
    weights: np.ndarray,
    wind_values: np.ndarray,
    da_values: np.ndarray,
    rt_values: np.ndarray,
    penalty_price_values: np.ndarray,
    tolerance_mw_values: np.ndarray | None,
    under_regularization_weight: float,
) -> dict[str, float]:
    awards = weights @ q_knots
    tolerance = 0.03 * awards if tolerance_mw_values is None else np.asarray(tolerance_mw_values, dtype=np.float64)
    over = np.maximum(wind_values - awards - tolerance, 0.0)
    under = np.maximum(awards - wind_values - tolerance, 0.0)
    profits = awards * da_values + (wind_values - awards) * rt_values - penalty_price_values * (over + under)
    mean_profit = float(np.mean(profits))
    mean_under_mw = float(np.mean(under))
    return {
        "objective": float(mean_profit - float(under_regularization_weight) * mean_under_mw),
        "mean_award_mw": float(np.mean(awards)),
        "mean_tolerance_mw": float(np.mean(tolerance)),
        "mean_profit": mean_profit,
        "mean_under_mw": mean_under_mw,
        "mean_under_cost": float(np.mean(penalty_price_values * under)),
    }


def _optimize_piecewise_offer_curve_direct_linprog(
    *,
    wind: np.ndarray,
    da: np.ndarray,
    rt: np.ndarray,
    pen: np.ndarray,
    tol_values: np.ndarray | None,
    p_knots: np.ndarray,
    weights: np.ndarray,
    objective_key: str,
    cvar_alpha: float,
    under_regularization_weight: float,
    capacity_mw: float,
    min_quantity_gap_mw: float,
    fallback_reason: str,
) -> dict[str, object]:
    """Solve the same direct piecewise LP with SciPy/HiGHS as a CPLEX fallback."""
    from scipy.optimize import linprog
    from scipy.sparse import coo_matrix

    n = int(len(wind))
    n_q = int(len(p_knots))
    q0 = 0
    over0 = n_q
    under0 = over0 + n
    eta_idx = None
    short0 = None
    n_vars = n_q + 2 * n
    if objective_key == "cvar":
        eta_idx = n_vars
        short0 = eta_idx + 1
        n_vars = short0 + n

    c = np.zeros(n_vars, dtype=np.float64)
    spread = da - rt
    if objective_key in {"mean_scenario", "mean_under_regularized"}:
        q_coeff = (weights.T @ spread) / float(n)
        c[q0 : q0 + n_q] = -q_coeff
        c[over0 : over0 + n] = pen / float(n)
        under_pen = pen.copy()
        if objective_key == "mean_under_regularized":
            under_pen = under_pen + float(max(0.0, under_regularization_weight))
        c[under0 : under0 + n] = under_pen / float(n)
    elif objective_key == "cvar":
        alpha = float(max(1e-6, cvar_alpha))
        c[int(eta_idx)] = -1.0
        c[int(short0) : int(short0) + n] = 1.0 / (alpha * float(n))
    else:
        raise ValueError(f"Unsupported objective: {objective_key!r}")

    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    b_ub: list[float] = []

    def add_ub(coeffs: list[tuple[int, float]], rhs: float) -> None:
        row = len(b_ub)
        for col, val in coeffs:
            if val != 0.0:
                rows.append(row)
                cols.append(int(col))
                data.append(float(val))
        b_ub.append(float(rhs))

    gap = max(0.0, float(min_quantity_gap_mw))
    if gap > 0:
        gap = min(gap, float(capacity_mw) / max(1, n_q - 1))
    for idx in range(n_q - 1):
        add_ub([(q0 + idx, 1.0), (q0 + idx + 1, -1.0)], -gap)

    for i in range(n):
        w_terms = [(q0 + k, float(weights[i, k])) for k in range(n_q)]
        if tol_values is None:
            add_ub([(col, -1.03 * val) for col, val in w_terms] + [(over0 + i, -1.0)], -float(wind[i]))
            add_ub([(col, 0.97 * val) for col, val in w_terms] + [(under0 + i, -1.0)], float(wind[i]))
        else:
            tol_i = float(tol_values[i])
            add_ub([(col, -val) for col, val in w_terms] + [(over0 + i, -1.0)], -float(wind[i] - tol_i))
            add_ub(w_terms + [(under0 + i, -1.0)], float(wind[i] + tol_i))

        if objective_key == "cvar":
            # eta - profit_i - shortfall_i <= 0
            # profit_i = wind_i * rt_i + award_i * (da_i - rt_i) - pen_i * (over_i + under_i)
            add_ub(
                [(col, -float(spread[i]) * val) for col, val in w_terms]
                + [
                    (over0 + i, float(pen[i])),
                    (under0 + i, float(pen[i])),
                    (int(eta_idx), 1.0),
                    (int(short0) + i, -1.0),
                ],
                float(wind[i] * rt[i]),
            )

    a_ub = coo_matrix((data, (rows, cols)), shape=(len(b_ub), n_vars)).tocsr()
    a_eq = coo_matrix(
        ([1.0, 1.0], ([0, 1], [q0, q0 + n_q - 1])),
        shape=(2, n_vars),
    ).tocsr()
    b_eq = np.asarray([0.0, float(capacity_mw)], dtype=np.float64)

    bounds: list[tuple[float | None, float | None]] = [(0.0, float(capacity_mw)) for _ in range(n_q)]
    bounds.extend([(0.0, None) for _ in range(2 * n)])
    if objective_key == "cvar":
        bounds.append((None, None))
        bounds.extend([(0.0, None) for _ in range(n)])

    res = linprog(
        c,
        A_ub=a_ub,
        b_ub=np.asarray(b_ub, dtype=np.float64),
        A_eq=a_eq,
        b_eq=b_eq,
        bounds=bounds,
        method="highs",
        options={
            "primal_feasibility_tolerance": 1e-7,
            "dual_feasibility_tolerance": 1e-7,
        },
    )
    if not res.success:
        raise RuntimeError(f"SciPy HiGHS fallback failed after CPLEX status {fallback_reason!r}: {res.message}")

    q_knots = np.asarray(res.x[q0 : q0 + n_q], dtype=np.float64)
    q_knots = np.clip(q_knots, 0.0, float(capacity_mw))
    q_knots[0] = 0.0
    q_knots[-1] = float(capacity_mw)
    q_knots = np.maximum.accumulate(q_knots)
    q_knots[-1] = float(capacity_mw)

    metrics = _evaluate_piecewise_solution(
        q_knots=q_knots,
        weights=weights,
        wind_values=wind,
        da_values=da,
        rt_values=rt,
        penalty_price_values=pen,
        tolerance_mw_values=tol_values,
        under_regularization_weight=float(under_regularization_weight),
    )
    awards = weights @ q_knots
    tolerance = 0.03 * awards if tol_values is None else tol_values
    over = np.maximum(wind - awards - tolerance, 0.0)
    under = np.maximum(awards - wind - tolerance, 0.0)
    profits = awards * da + (wind - awards) * rt - pen * (over + under)
    return {
        "q_knots": q_knots,
        "p_knots": p_knots,
        "objective": float(-res.fun),
        "price_error": 0.0,
        "slope_error": 0.0,
        "max_error": 0.0,
        "success": True,
        "message": f"cplex_fallback_highs_after_{fallback_reason}",
        "nfev": 0,
        "nit": int(getattr(res, "nit", 0) or 0),
        "solver": "scipy_highs_fallback",
        "n_scenarios": int(n),
        "mean_award_mw": float(np.mean(awards)),
        "mean_tolerance_mw": float(np.mean(tolerance)),
        "mean_profit": float(np.mean(profits)),
        "mean_under_mw": float(np.mean(under)),
        "mean_under_cost": float(np.mean(pen * under)),
        "stage_profit_samples": profits,
        **metrics,
    }


def optimize_piecewise_offer_curve_direct_cplex(
    wind_values: np.ndarray,
    da_energy_values: np.ndarray,
    rt_energy_values: np.ndarray,
    penalty_price_values: np.ndarray,
    tolerance_mw_values: np.ndarray | None = None,
    objective: str = "mean_scenario",
    cvar_alpha: float = 0.10,
    under_regularization_weight: float = 0.0,
    capacity_mw: float = CAPACITY_MW,
    n_price_knots: int = 7,
    min_quantity_gap_mw: float = 0.0,
    time_limit_sec: float | None = None,
    cplex_threads: int = 0,
) -> dict[str, object]:
    try:
        from docplex.mp.model import Model
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Direct piecewise optimisation requires docplex.") from exc

    wind = np.asarray(wind_values, dtype=np.float64)
    da = np.asarray(da_energy_values, dtype=np.float64)
    rt = np.asarray(rt_energy_values, dtype=np.float64)
    pen = np.asarray(penalty_price_values, dtype=np.float64)
    n = int(min(len(wind), len(da), len(rt), len(pen)))
    if tolerance_mw_values is not None:
        tol_values = np.asarray(tolerance_mw_values, dtype=np.float64)
        n = int(min(n, len(tol_values)))
    else:
        tol_values = None
    if n <= 0:
        raise ValueError("No scenarios available for direct piecewise optimisation.")
    wind = wind[:n]
    da = da[:n]
    rt = rt[:n]
    pen = pen[:n]
    if tol_values is not None:
        tol_values = tol_values[:n]

    p_knots = make_price_knots_from_samples(da, n_knots=int(n_price_knots))
    weights = _price_interpolation_weights(da, p_knots)

    mdl = Model(name="direct_piecewise_offer")
    mdl.context.solver.log_output = False
    if time_limit_sec is not None and float(time_limit_sec) > 0:
        mdl.parameters.timelimit = float(time_limit_sec)
    if int(cplex_threads) > 0:
        mdl.parameters.threads = int(cplex_threads)
    try:
        mdl.parameters.emphasis.numerical = 1
        mdl.parameters.read.scale = 1
    except Exception:
        pass

    q_vars = [mdl.continuous_var(lb=0.0, ub=float(capacity_mw), name=f"Q{k}") for k in range(len(p_knots))]
    mdl.add_constraint(q_vars[0] == 0.0)
    mdl.add_constraint(q_vars[-1] == float(capacity_mw))
    gap = max(0.0, float(min_quantity_gap_mw))
    if gap > 0:
        gap = min(gap, float(capacity_mw) / max(1, len(q_vars) - 1))
    for idx in range(len(q_vars) - 1):
        mdl.add_constraint(q_vars[idx + 1] >= q_vars[idx] + gap)

    profit_exprs = []
    under_exprs = []
    for i in range(n):
        award_expr = mdl.sum(float(weights[i, k]) * q_vars[k] for k in range(len(q_vars)))
        over_var = mdl.continuous_var(lb=0.0, name=f"over_{i}")
        under_var = mdl.continuous_var(lb=0.0, name=f"under_{i}")
        if tol_values is None:
            mdl.add_constraint(over_var >= float(wind[i]) - 1.03 * award_expr)
            mdl.add_constraint(under_var >= 0.97 * award_expr - float(wind[i]))
        else:
            mdl.add_constraint(over_var >= float(wind[i]) - award_expr - float(tol_values[i]))
            mdl.add_constraint(under_var >= award_expr - float(tol_values[i]) - float(wind[i]))
        profit_expr = award_expr * float(da[i]) + (float(wind[i]) - award_expr) * float(rt[i]) - float(pen[i]) * (over_var + under_var)
        profit_exprs.append(profit_expr)
        under_exprs.append(under_var)

    mean_profit_expr = mdl.sum(profit_exprs) / float(n)
    objective_key = str(objective).strip().lower()
    if objective_key == "mean":
        objective_key = "mean_scenario"
    if objective_key == "mean_scenario":
        mdl.maximize(mean_profit_expr)
    elif objective_key == "cvar":
        alpha = float(max(1e-6, cvar_alpha))
        eta = mdl.continuous_var(name="eta")
        shortfall = [mdl.continuous_var(lb=0.0, name=f"cvar_short_{i}") for i in range(n)]
        for i in range(n):
            mdl.add_constraint(shortfall[i] >= eta - profit_exprs[i])
        mdl.maximize(eta - mdl.sum(shortfall) / (alpha * float(n)))
    elif objective_key == "mean_under_regularized":
        weight = float(max(0.0, under_regularization_weight))
        mdl.maximize(mean_profit_expr - weight * mdl.sum(under_exprs) / float(n))
    else:
        raise ValueError(f"Unsupported objective: {objective!r}")

    solution = mdl.solve(clean_before_solve=True)
    details = mdl.solve_details
    status_text = str(getattr(details, "status", "no solution")).lower()
    if solution is None and "unscaled infeas" in status_text:
        # CPLEX can occasionally report an LP as "optimal with unscaled
        # infeasibilities" for numerically delicate price/quantity scales.  In
        # that case, retry with stronger numerical emphasis before giving up.
        try:
            mdl.parameters.emphasis.numerical = 1
            mdl.parameters.read.scale = 1
            mdl.parameters.simplex.tolerances.feasibility = 1e-7
            mdl.parameters.simplex.tolerances.optimality = 1e-7
        except Exception:
            pass
        solution = mdl.solve(clean_before_solve=True)
        details = mdl.solve_details
        status_text = str(getattr(details, "status", "no solution")).lower()
    if solution is None and "optimal" in status_text:
        candidate_solution = getattr(mdl, "solution", None)
        if candidate_solution is not None:
            solution = candidate_solution
    if solution is None:
        return _optimize_piecewise_offer_curve_direct_linprog(
            wind=wind,
            da=da,
            rt=rt,
            pen=pen,
            tol_values=tol_values,
            p_knots=p_knots,
            weights=weights,
            objective_key=objective_key,
            cvar_alpha=float(cvar_alpha),
            under_regularization_weight=float(under_regularization_weight),
            capacity_mw=float(capacity_mw),
            min_quantity_gap_mw=float(min_quantity_gap_mw),
            fallback_reason=str(getattr(details, "status", "no solution")).replace(" ", "_"),
        )

    q_knots = np.asarray([solution.get_value(var) for var in q_vars], dtype=np.float64)
    metrics = _evaluate_piecewise_solution(
        q_knots=q_knots,
        weights=weights,
        wind_values=wind,
        da_values=da,
        rt_values=rt,
        penalty_price_values=pen,
        tolerance_mw_values=tol_values,
        under_regularization_weight=float(under_regularization_weight),
    )
    awards = weights @ q_knots
    tolerance = 0.03 * awards if tol_values is None else tol_values
    over = np.maximum(wind - awards - tolerance, 0.0)
    under = np.maximum(awards - wind - tolerance, 0.0)
    profits = awards * da + (wind - awards) * rt - pen * (over + under)
    return {
        "q_knots": q_knots,
        "p_knots": p_knots,
        "objective": float(solution.objective_value),
        "price_error": 0.0,
        "slope_error": 0.0,
        "max_error": 0.0,
        "success": True,
        "message": str(getattr(details, "status", "optimal")),
        "nfev": 0,
        "nit": 0,
        "solver": "cplex_direct_piecewise",
        "n_scenarios": int(n),
        "mean_award_mw": float(np.mean(awards)),
        "mean_tolerance_mw": float(np.mean(tolerance)),
        "mean_profit": float(np.mean(profits)),
        "mean_under_mw": float(np.mean(under)),
        "mean_under_cost": float(np.mean(pen * under)),
        "stage_profit_samples": profits,
        **metrics,
    }


def _resolve_scip_executable(scip_executable: str | Path | None = None) -> str | None:
    if scip_executable is not None and str(scip_executable).strip():
        return str(Path(scip_executable))
    env_path = os.environ.get("SCIP_EXECUTABLE", "").strip()
    if env_path:
        return env_path
    try:
        import shutil
        import sys

        found = shutil.which("scip")
        if found:
            return found
        candidates = [
            Path(sys.prefix) / "Library" / "bin" / "scip.exe",
            Path(sys.prefix) / "bin" / "scip",
        ]
        for candidate in candidates:
            if candidate.exists():
                return str(candidate)
    except Exception:
        return None
    return None


def optimize_variable_breakpoint_offer_curve_scip(
    wind_values: np.ndarray,
    da_energy_values: np.ndarray,
    rt_energy_values: np.ndarray,
    penalty_price_values: np.ndarray,
    tolerance_mw_values: np.ndarray | None = None,
    objective: str = "mean_scenario",
    cvar_alpha: float = 0.10,
    under_regularization_weight: float = 0.0,
    capacity_mw: float = CAPACITY_MW,
    n_price_knots: int = 7,
    min_quantity_gap_mw: float = 0.0,
    min_price_gap_fraction: float = 1e-5,
    time_limit_sec: float | None = 120.0,
    scip_executable: str | Path | None = None,
    tee: bool = False,
) -> dict[str, object]:
    """Solve a small exact variable-breakpoint piecewise offer MINLP with SCIP.

    The current CPLEX production model fixes price knots from DA-price
    quantiles and optimizes only quantities.  This SCIP model is the explicit
    counterpart where internal price breakpoints and quantity knots are both
    decision variables.  The first and last price knots are fixed to the
    sampled DA-price min/max so every sampled DA price has a well-defined
    award on the curve.
    """
    try:
        import pyomo.environ as pyo
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Variable-breakpoint MINLP requires Pyomo.") from exc

    wind = np.asarray(wind_values, dtype=np.float64)
    da = np.asarray(da_energy_values, dtype=np.float64)
    rt = np.asarray(rt_energy_values, dtype=np.float64)
    pen = np.asarray(penalty_price_values, dtype=np.float64)
    n = int(min(len(wind), len(da), len(rt), len(pen)))
    if tolerance_mw_values is not None:
        tol_values = np.asarray(tolerance_mw_values, dtype=np.float64)
        n = int(min(n, len(tol_values)))
    else:
        tol_values = None
    if n <= 0:
        raise ValueError("No scenarios available for variable-breakpoint MINLP.")
    if int(n_price_knots) < 2:
        raise ValueError("At least two price knots are required.")

    wind = wind[:n]
    da = da[:n]
    rt = rt[:n]
    pen = pen[:n]
    if tol_values is not None:
        tol_values = tol_values[:n]

    finite_da = da[np.isfinite(da)]
    if finite_da.size == 0:
        raise ValueError("DA price samples are empty or non-finite.")
    p_min = float(np.nanmin(finite_da))
    p_max = float(np.nanmax(finite_da))
    if not np.isfinite(p_min) or not np.isfinite(p_max):
        raise ValueError("DA price samples are non-finite.")
    if abs(p_max - p_min) < 1e-8:
        centre = 0.5 * (p_min + p_max)
        p_min = centre - 1e-3
        p_max = centre + 1e-3
    p_span = float(p_max - p_min)
    da_scaled = np.clip((da - p_min) / p_span, 0.0, 1.0)
    cap = float(capacity_mw)
    wind_scaled = wind / cap
    tol_scaled = None if tol_values is None else tol_values / cap

    n_knots = int(n_price_knots)
    init_p = np.linspace(0.0, 1.0, n_knots, dtype=np.float64)
    q_init = np.linspace(0.0, 1.0, n_knots, dtype=np.float64)
    min_q_gap = max(0.0, float(min_quantity_gap_mw)) / cap
    if min_q_gap > 0:
        min_q_gap = min(min_q_gap, 1.0 / max(1, n_knots - 1))
    min_p_gap = max(0.0, float(min_price_gap_fraction))
    if min_p_gap > 0:
        min_p_gap = min(min_p_gap, 1.0 / max(1, n_knots - 1))

    model = pyo.ConcreteModel(name="variable_breakpoint_offer_curve_scip")
    model.I = pyo.RangeSet(0, n - 1)
    model.K = pyo.RangeSet(0, n_knots - 1)
    model.J = pyo.RangeSet(0, n_knots - 2)
    model.rho = pyo.Var(model.K, bounds=(0.0, 1.0), initialize=lambda _m, k: float(init_p[int(k)]))
    model.q = pyo.Var(model.K, bounds=(0.0, 1.0), initialize=lambda _m, k: float(q_init[int(k)]))
    model.lam = pyo.Var(model.I, model.K, bounds=(0.0, 1.0), initialize=0.0)
    model.segment = pyo.Var(model.I, model.J, within=pyo.Binary, initialize=0)
    model.award = pyo.Var(model.I, bounds=(0.0, 1.0), initialize=0.5)
    model.over = pyo.Var(model.I, bounds=(0.0, None), initialize=0.0)
    model.under = pyo.Var(model.I, bounds=(0.0, None), initialize=0.0)

    model.fix_p_min = pyo.Constraint(expr=model.rho[0] == 0.0)
    model.fix_p_max = pyo.Constraint(expr=model.rho[n_knots - 1] == 1.0)
    model.fix_q_min = pyo.Constraint(expr=model.q[0] == 0.0)
    model.fix_q_max = pyo.Constraint(expr=model.q[n_knots - 1] == 1.0)
    model.price_order = pyo.Constraint(
        model.J,
        rule=lambda m, j: m.rho[int(j) + 1] >= m.rho[int(j)] + min_p_gap,
    )
    model.quantity_order = pyo.Constraint(
        model.J,
        rule=lambda m, j: m.q[int(j) + 1] >= m.q[int(j)] + min_q_gap,
    )
    model.one_segment = pyo.Constraint(model.I, rule=lambda m, i: sum(m.segment[i, j] for j in m.J) == 1)
    model.lambda_sum = pyo.Constraint(model.I, rule=lambda m, i: sum(m.lam[i, k] for k in m.K) == 1)
    model.price_interp = pyo.Constraint(
        model.I,
        rule=lambda m, i: sum(m.lam[i, k] * m.rho[k] for k in m.K) == float(da_scaled[int(i)]),
    )
    model.award_interp = pyo.Constraint(
        model.I,
        rule=lambda m, i: m.award[i] == sum(m.lam[i, k] * m.q[k] for k in m.K),
    )

    def _lambda_adjacency_rule(m, i, k):
        kk = int(k)
        if kk == 0:
            return m.lam[i, k] <= m.segment[i, 0]
        if kk == n_knots - 1:
            return m.lam[i, k] <= m.segment[i, n_knots - 2]
        return m.lam[i, k] <= m.segment[i, kk - 1] + m.segment[i, kk]

    model.lambda_adjacency = pyo.Constraint(model.I, model.K, rule=_lambda_adjacency_rule)

    if tol_scaled is None:
        model.over_def = pyo.Constraint(
            model.I,
            rule=lambda m, i: m.over[i] >= float(wind_scaled[int(i)]) - 1.03 * m.award[i],
        )
        model.under_def = pyo.Constraint(
            model.I,
            rule=lambda m, i: m.under[i] >= 0.97 * m.award[i] - float(wind_scaled[int(i)]),
        )
    else:
        model.over_def = pyo.Constraint(
            model.I,
            rule=lambda m, i: m.over[i] >= float(wind_scaled[int(i)]) - m.award[i] - float(tol_scaled[int(i)]),
        )
        model.under_def = pyo.Constraint(
            model.I,
            rule=lambda m, i: m.under[i] >= m.award[i] - float(wind_scaled[int(i)]) - float(tol_scaled[int(i)]),
        )

    def _profit_expr(m, i):
        ii = int(i)
        return cap * (
            m.award[i] * float(da[ii])
            + (float(wind_scaled[ii]) - m.award[i]) * float(rt[ii])
            - float(pen[ii]) * (m.over[i] + m.under[i])
        )

    objective_key = str(objective).strip().lower()
    if objective_key == "mean":
        objective_key = "mean_scenario"
    if objective_key == "mean_scenario":
        model.obj = pyo.Objective(expr=sum(_profit_expr(model, i) for i in model.I) / float(n), sense=pyo.maximize)
    elif objective_key == "mean_under_regularized":
        weight = float(max(0.0, under_regularization_weight))
        model.obj = pyo.Objective(
            expr=(
                sum(_profit_expr(model, i) for i in model.I) / float(n)
                - cap * weight * sum(model.under[i] for i in model.I) / float(n)
            ),
            sense=pyo.maximize,
        )
    elif objective_key == "cvar":
        alpha = float(max(1e-6, cvar_alpha))
        model.eta = pyo.Var(initialize=0.0)
        model.shortfall = pyo.Var(model.I, bounds=(0.0, None), initialize=0.0)
        model.cvar_shortfall = pyo.Constraint(
            model.I,
            rule=lambda m, i: m.shortfall[i] >= m.eta - _profit_expr(m, i),
        )
        model.obj = pyo.Objective(
            expr=model.eta - sum(model.shortfall[i] for i in model.I) / (alpha * float(n)),
            sense=pyo.maximize,
        )
    else:
        raise ValueError(f"Unsupported objective: {objective!r}")

    for i in range(n):
        right = int(np.searchsorted(init_p, da_scaled[i], side="right"))
        left = max(0, min(n_knots - 2, right - 1))
        denom = max(1e-12, init_p[left + 1] - init_p[left])
        frac = float((da_scaled[i] - init_p[left]) / denom)
        model.segment[i, left].value = 1
        model.lam[i, left].value = max(0.0, min(1.0, 1.0 - frac))
        model.lam[i, left + 1].value = max(0.0, min(1.0, frac))
        model.award[i].value = float(da_scaled[i])

    scip_path = _resolve_scip_executable(scip_executable)
    solver = pyo.SolverFactory("scip", executable=scip_path) if scip_path else pyo.SolverFactory("scip")
    if not solver.available():
        raise RuntimeError("SCIP executable is not available to Pyomo.")
    options: dict[str, object] = {"display/verblevel": 0}
    if time_limit_sec is not None and float(time_limit_sec) > 0:
        options["limits/time"] = float(time_limit_sec)
    results = solver.solve(model, tee=bool(tee), options=options)
    termination = str(results.solver.termination_condition)
    status = str(results.solver.status)
    has_solution = termination.lower() in {
        "optimal",
        "globallyoptimal",
        "locallyoptimal",
        "feasible",
        "maxtimelimit",
        "maxiterations",
    }
    if not has_solution:
        raise RuntimeError(f"SCIP variable-breakpoint MINLP failed: status={status}, termination={termination}")

    rho_knots = np.asarray([pyo.value(model.rho[k]) for k in range(n_knots)], dtype=np.float64)
    q_knots_scaled = np.asarray([pyo.value(model.q[k]) for k in range(n_knots)], dtype=np.float64)
    p_knots = p_min + p_span * rho_knots
    q_knots = np.clip(cap * q_knots_scaled, 0.0, cap)
    awards_scaled = np.asarray([pyo.value(model.award[i]) for i in range(n)], dtype=np.float64)
    awards = np.clip(cap * awards_scaled, 0.0, cap)
    tolerance = 0.03 * awards if tol_values is None else tol_values
    over = np.maximum(wind - awards - tolerance, 0.0)
    under = np.maximum(awards - wind - tolerance, 0.0)
    profits = awards * da + (wind - awards) * rt - pen * (over + under)
    return {
        "q_knots": q_knots,
        "p_knots": p_knots,
        "objective": float(pyo.value(model.obj)),
        "price_error": 0.0,
        "slope_error": 0.0,
        "max_error": 0.0,
        "success": True,
        "certified_optimal": termination.lower() in {"optimal", "globallyoptimal"},
        "message": f"{status}/{termination}",
        "nfev": 0,
        "nit": 0,
        "solver": "pyomo_scip_variable_breakpoint_minlp",
        "n_scenarios": int(n),
        "mean_award_mw": float(np.mean(awards)),
        "mean_tolerance_mw": float(np.mean(tolerance)),
        "mean_profit": float(np.mean(profits)),
        "mean_under_mw": float(np.mean(under)),
        "mean_under_cost": float(np.mean(pen * under)),
        "stage_profit_samples": profits,
        "rho_knots": rho_knots,
        "price_bounds": (float(p_min), float(p_max)),
    }


def summarize_settlement(detail_df: pd.DataFrame) -> pd.DataFrame:
    if detail_df.empty:
        return pd.DataFrame(columns=["strategy_name", "n_rows", "annual_total_settlement", "annual_total_da_settlement", "annual_total_rt_settlement", "annual_total_energy_settlement", "annual_total_penalty_settlement", "hourly_mean_settlement", "hourly_mean_da_settlement", "hourly_mean_rt_settlement", "hourly_mean_energy_settlement", "hourly_mean_penalty_settlement", "hourly_std_settlement", "downside_semideviation", "penalty_event_count", "total_penalty_amount", "mean_tolerance_mw", "mean_penalty_band_reference_mw", "mean_operator_output_limit_mw", "mean_award_mw", "mean_actual_wind_mw"])
    rows = []
    for strategy_name, sub in detail_df.groupby("strategy_name", sort=True):
        settle = pd.to_numeric(sub["settlement_amount"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
        da_settle = pd.to_numeric(sub["da_settlement_amount"], errors="coerce").fillna(0.0)
        rt_settle = pd.to_numeric(sub["rt_settlement_amount"], errors="coerce").fillna(0.0)
        energy_settle = da_settle + rt_settle
        downside = np.minimum(settle, 0.0)
        rows.append({
            "strategy_name": str(strategy_name),
            "n_rows": int(len(sub)),
            "annual_total_settlement": float(np.sum(settle)),
            "annual_total_da_settlement": float(da_settle.sum()),
            "annual_total_rt_settlement": float(rt_settle.sum()),
            "annual_total_energy_settlement": float(energy_settle.sum()),
            "annual_total_penalty_settlement": float(pd.to_numeric(sub["penalty_settlement_amount"], errors="coerce").fillna(0.0).sum()),
            "hourly_mean_settlement": float(np.mean(settle)),
            "hourly_mean_da_settlement": float(da_settle.mean()),
            "hourly_mean_rt_settlement": float(rt_settle.mean()),
            "hourly_mean_energy_settlement": float(energy_settle.mean()),
            "hourly_mean_penalty_settlement": float(pd.to_numeric(sub["penalty_settlement_amount"], errors="coerce").fillna(0.0).mean()),
            "hourly_std_settlement": float(np.std(settle, ddof=0)),
            "downside_semideviation": float(np.sqrt(np.mean(np.square(downside)))),
            "penalty_event_count": int(sub["penalty_trigger"].astype(bool).sum()),
            "total_penalty_amount": float(pd.to_numeric(sub["penalty_amount"], errors="coerce").fillna(0.0).sum()),
            "mean_tolerance_mw": float(pd.to_numeric(sub["tolerance_mw"], errors="coerce").mean()),
            "mean_penalty_band_reference_mw": float(pd.to_numeric(sub["penalty_band_reference_mw"], errors="coerce").mean()),
            "mean_operator_output_limit_mw": float(pd.to_numeric(sub.get("operator_output_limit_mw", np.nan), errors="coerce").mean()),
            "mean_award_mw": float(pd.to_numeric(sub["award_mw"], errors="coerce").mean()),
            "mean_actual_wind_mw": float(pd.to_numeric(sub["actual_wind_mw"], errors="coerce").mean()),
        })
    return pd.DataFrame(rows).sort_values("strategy_name").reset_index(drop=True)


def summarize_penalty(detail_df: pd.DataFrame) -> pd.DataFrame:
    if detail_df.empty:
        return pd.DataFrame(columns=["strategy_name", "penalty_event_count", "penalty_event_rate", "total_penalty_amount", "mean_penalty_amount_when_triggered", "mean_over_generation_mw", "mean_under_generation_mw", "over_penalty_event_count", "under_penalty_event_count", "total_over_penalty_amount", "total_under_penalty_amount", "mean_over_penalty_mw", "mean_under_penalty_mw", "mean_tolerance_mw", "mean_penalty_band_reference_mw"])
    rows = []
    for strategy_name, sub in detail_df.groupby("strategy_name", sort=True):
        penalty_flag = sub["penalty_trigger"].astype(bool)
        over_flag = sub["over_penalty_trigger"].astype(bool)
        under_flag = sub["under_penalty_trigger"].astype(bool)
        penalty_amount = pd.to_numeric(sub["penalty_amount"], errors="coerce").fillna(0.0)
        rows.append({
            "strategy_name": str(strategy_name),
            "penalty_event_count": int(penalty_flag.sum()),
            "penalty_event_rate": float(penalty_flag.mean()),
            "total_penalty_amount": float(penalty_amount.sum()),
            "mean_penalty_amount_when_triggered": float(penalty_amount.loc[penalty_flag].mean()) if penalty_flag.any() else 0.0,
            "mean_over_generation_mw": float(pd.to_numeric(sub["over_generation_mw"], errors="coerce").mean()),
            "mean_under_generation_mw": float(pd.to_numeric(sub["under_generation_mw"], errors="coerce").mean()),
            "over_penalty_event_count": int(over_flag.sum()),
            "under_penalty_event_count": int(under_flag.sum()),
            "total_over_penalty_amount": float(pd.to_numeric(sub["over_penalty_amount"], errors="coerce").fillna(0.0).sum()),
            "total_under_penalty_amount": float(pd.to_numeric(sub["under_penalty_amount"], errors="coerce").fillna(0.0).sum()),
            "mean_over_penalty_mw": float(pd.to_numeric(sub["over_penalty_mw"], errors="coerce").mean()),
            "mean_under_penalty_mw": float(pd.to_numeric(sub["under_penalty_mw"], errors="coerce").mean()),
            "mean_tolerance_mw": float(pd.to_numeric(sub["tolerance_mw"], errors="coerce").mean()),
            "mean_penalty_band_reference_mw": float(pd.to_numeric(sub["penalty_band_reference_mw"], errors="coerce").mean()),
        })
    return pd.DataFrame(rows).sort_values("strategy_name").reset_index(drop=True)


def summarize_distortion(detail_df: pd.DataFrame) -> pd.DataFrame:
    if detail_df.empty or "award_gap_vs_continuous_mw" not in detail_df.columns:
        return pd.DataFrame(columns=["case_name", "mean_abs_award_gap_mw", "mean_award_gap_mw", "under_award_rate", "over_award_rate", "max_abs_award_gap_mw"])
    work = detail_df.loc[detail_df["award_gap_vs_continuous_mw"].notna()].copy()
    if "strategy_label" in work.columns:
        work = work.loc[work["strategy_label"].astype(str).str.contains("piecewise", na=False)].copy()
    if work.empty:
        return pd.DataFrame(columns=["case_name", "mean_abs_award_gap_mw", "mean_award_gap_mw", "under_award_rate", "over_award_rate", "max_abs_award_gap_mw"])
    rows = []
    for case_name, sub in work.groupby("case_name", sort=True):
        gap = pd.to_numeric(sub["award_gap_vs_continuous_mw"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
        rows.append({
            "case_name": str(case_name),
            "mean_abs_award_gap_mw": float(np.mean(np.abs(gap))),
            "mean_award_gap_mw": float(np.mean(gap)),
            "under_award_rate": float(np.mean(gap < 0.0)),
            "over_award_rate": float(np.mean(gap > 0.0)),
            "max_abs_award_gap_mw": float(np.max(np.abs(gap))),
        })
    return pd.DataFrame(rows).sort_values("case_name").reset_index(drop=True)

def validation_output_paths(out_root: Path | str) -> dict[str, Path]:
    out_root = ensure_dir(out_root)
    return {
        "detail_csv": out_root / VALIDATION_DETAIL_CSV,
        "annual_csv": out_root / VALIDATION_ANNUAL_CSV,
        "penalty_csv": out_root / VALIDATION_PENALTY_CSV,
        "distortion_csv": out_root / VALIDATION_DISTORTION_CSV,
        "meta_json": out_root / VALIDATION_META_JSON,
        "cumulative_png": out_root / VALIDATION_CUMULATIVE_PNG,
        "representative_png": out_root / VALIDATION_REPRESENTATIVE_PNG,
    }


def plot_cumulative_profit(detail_df: pd.DataFrame, out_path: Path | str) -> None:
    out_path = Path(out_path)
    ensure_dir(out_path.parent)
    if plt is None or detail_df.empty:
        out_path.write_bytes(b"")
        return
    work = detail_df.copy()
    work["timestamp_utc"] = pd.to_datetime(work["timestamp_utc"], utc=True, errors="coerce")
    work = work.dropna(subset=["timestamp_utc"]).copy()
    fig, ax = plt.subplots(figsize=(10, 6))
    for strategy_name, sub in work.groupby("strategy_name", sort=True):
        sub = sub.sort_values("timestamp_utc")
        cum = pd.to_numeric(sub["settlement_amount"], errors="coerce").fillna(0.0).cumsum()
        ax.plot(sub["timestamp_utc"], cum, label=str(strategy_name), linewidth=1.5)
    ax.set_title("Cumulative settlement")
    ax.set_xlabel("UTC timestamp")
    ax.set_ylabel("USD")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_representative_curves(actual_df: pd.DataFrame, continuous_df: pd.DataFrame, piecewise_df: pd.DataFrame, out_path: Path | str) -> None:
    out_path = Path(out_path)
    ensure_dir(out_path.parent)
    if plt is None or piecewise_df.empty:
        out_path.write_bytes(b"")
        return

    use_continuous = continuous_df is not None and not continuous_df.empty
    if use_continuous:
        cont_row = continuous_df.iloc[0]
        pw_rows = piecewise_df.loc[
            (piecewise_df["case_name"] == cont_row["case_name"])
            & (piecewise_df["operating_date_et"] == cont_row["operating_date_et"])
            & (piecewise_df["hour_et"] == cont_row["hour_et"])
        ]
        if pw_rows.empty:
            use_continuous = False
        else:
            pw_row = pw_rows.iloc[0]
    if not use_continuous:
        pw_row = piecewise_df.iloc[0]

    q_pw = np.asarray([float(pw_row[f"Q{k}"]) for k in range(7)], dtype=np.float64)
    p_pw = np.asarray([float(pw_row[f"P{k}"]) for k in range(7)], dtype=np.float64)
    fig, ax = plt.subplots(figsize=(8, 5))
    if use_continuous:
        q_cont = np.asarray(json.loads(cont_row["q_grid_json"]), dtype=np.float64)
        p_cont = np.asarray(json.loads(cont_row.get("p_smooth_json", cont_row.get("da_price_grid_json", "[]"))), dtype=np.float64)
        ax.plot(p_cont, q_cont, label="continuous", linewidth=2.0)
    ax.plot(p_pw, q_pw, label="piecewise linear (7 points)", linewidth=2.0, marker="o", markersize=4)
    title_suffix = "" if use_continuous else " (piecewise only)"
    ax.set_title(f"Representative curve: {pw_row['case_name']} / {pw_row['operating_date_et']} h{int(pw_row['hour_et'])}{title_suffix}")
    ax.set_xlabel("DA price")
    ax.set_ylabel("Award quantity (MW)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)






