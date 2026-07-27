from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
POINT_ROOT = ROOT / "03_점예측" / "점예측결과"
OUT_ROOT = ROOT / "04_오차데이터" / "오차데이터결과"
CAPACITY_MW = 1985.3
TZ_ET = "America/New_York"

POINT_FILE_NAMES = {
    "wind": {
        "train": "01_학습기간_풍력발전_예측결과.csv",
        "test": "02_테스트기간_풍력발전_예측결과.csv",
    },
    "da_energy": {
        "train": "03_학습기간_DA에너지_예측결과.csv",
        "test": "04_테스트기간_DA에너지_예측결과.csv",
    },
    "rt_energy": {
        "train": "05_학습기간_RT에너지_예측결과.csv",
        "test": "06_테스트기간_RT에너지_예측결과.csv",
    },
    "da_reg": {
        "train": "07_학습기간_하루전_REG가격_예측결과.csv",
        "test": "08_테스트기간_하루전_REG가격_예측결과.csv",
    },
    "rt_reg": {
        "train": "09_학습기간_실시간_REG가격_예측결과.csv",
        "test": "10_테스트기간_실시간_REG가격_예측결과.csv",
    },
}

VALID_POINT_FILE_NAMES = {
    "wind": "16_검증기간_풍력발전_예측결과.csv",
    "da_energy": "17_검증기간_DA에너지_예측결과.csv",
    "rt_energy": "18_검증기간_RT에너지_예측결과.csv",
    "da_reg": "19_검증기간_하루전_REG가격_예측결과.csv",
    "rt_reg": "20_검증기간_실시간_REG가격_예측결과.csv",
}

SERIES_INFO = [
    ("wind", "풍력발전", "01_풍력발전_오차시계열_2018_2021.csv", "11_풍력발전_오차분석요약.csv"),
    ("da_energy", "DA에너지", "02_DA에너지_오차시계열_2018_2021.csv", "12_DA에너지_오차분석요약.csv"),
    ("rt_energy", "RT에너지", "03_RT에너지_오차시계열_2018_2021.csv", "13_RT에너지_오차분석요약.csv"),
    ("da_reg", "DA_REG가격", "04_DA_REG가격_오차시계열_2018_2021.csv", "14_DA_REG가격_오차분석요약.csv"),
    ("rt_reg", "RT_REG가격", "05_RT_REG가격_오차시계열_2018_2021.csv", "15_RT_REG가격_오차분석요약.csv"),
    ("spread", "스프레드가격", "06_스프레드가격_오차시계열_2018_2021.csv", "16_스프레드가격_오차분석요약.csv"),
]


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _dump_json(path: Path, payload: object) -> None:
    _ensure_dir(path.parent)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build error series from point-forecast outputs.")
    parser.add_argument("--point-root", type=Path, default=POINT_ROOT)
    parser.add_argument("--out-root", type=Path, default=OUT_ROOT)
    return parser.parse_args()


def _candidate_path(kind: str, split: str) -> Path:
    if split == "valid":
        path = POINT_ROOT / VALID_POINT_FILE_NAMES[kind]
    else:
        path = POINT_ROOT / POINT_FILE_NAMES[kind][split]
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def _fallback_issue_et(target_et: pd.Series) -> pd.Series:
    midnight = target_et.dt.floor("D")
    return midnight - pd.Timedelta(days=1) + pd.Timedelta(hours=5)


def _safe_logit(x: pd.Series) -> pd.Series:
    eps = 1e-6
    clipped = x.clip(eps, 1.0 - eps)
    return np.log(clipped / (1.0 - clipped))


def _load_point_frame(kind: str, split: str) -> pd.DataFrame:
    path = _candidate_path(kind, split)
    df = pd.read_csv(path, encoding="utf-8-sig")
    target_col = next(col for col in ("시각_UTC", "TimeStamp", "목표시각_UTC") if col in df.columns)
    actual_col = next(col for col in ("실제값", "실측값") if col in df.columns)
    pred_col = "예측값"
    issue_col = "예측기준시각_ET" if "예측기준시각_ET" in df.columns else None
    horizon_col = "예측순번" if "예측순번" in df.columns else None

    target_utc = pd.to_datetime(df[target_col], utc=True, errors="coerce")
    target_et = target_utc.dt.tz_convert(TZ_ET)
    if issue_col:
        issue_utc = pd.to_datetime(df[issue_col], utc=True, errors="coerce")
        if issue_utc.notna().any():
            issue_et = issue_utc.dt.tz_convert(TZ_ET)
        else:
            issue_et = _fallback_issue_et(target_et)
    else:
        issue_et = _fallback_issue_et(target_et)

    out = pd.DataFrame(
        {
            "timestamp_utc": target_utc,
            "timestamp_et": target_et,
            "origin_timestamp_et": issue_et,
            "operating_date_et": target_et.dt.strftime("%Y-%m-%d"),
            "hour_et": target_et.dt.hour.astype("Int64"),
            "split": split,
            "y_true_price": pd.to_numeric(df[actual_col], errors="coerce"),
            "y_pred_price": pd.to_numeric(df[pred_col], errors="coerce"),
        }
    )
    if horizon_col:
        out["horizon_index"] = pd.to_numeric(df[horizon_col], errors="coerce")
    else:
        out["horizon_index"] = (
            out.sort_values("timestamp_utc")
            .groupby(out["origin_timestamp_et"].astype(str))
            .cumcount()
            .astype(float)
        )
    out = out.dropna(subset=["timestamp_utc", "y_true_price", "y_pred_price"]).copy()
    out = out.sort_values(["origin_timestamp_et", "timestamp_utc"]).reset_index(drop=True)
    return out


def _add_common_error_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["error_price"] = df["y_true_price"] - df["y_pred_price"]
    df["error_value"] = df["error_price"]
    df["axis_true_u"] = df["y_true_price"].rank(method="average", pct=True)
    df["axis_pred_u"] = df["y_pred_price"].rank(method="average", pct=True)
    df["delta_true_price"] = df.groupby("origin_timestamp_et")["y_true_price"].diff()
    df["delta_pred_price"] = df.groupby("origin_timestamp_et")["y_pred_price"].diff()
    df["delta_error"] = df["delta_true_price"] - df["delta_pred_price"]
    return df


def _build_wind_frame() -> pd.DataFrame:
    frames = [_load_point_frame("wind", split) for split in ("train", "valid", "test")]
    df = pd.concat(frames, ignore_index=True)
    df = _add_common_error_columns(df)
    df["capacity_mw"] = CAPACITY_MW
    df["y_true_raw_mw"] = df["y_true_price"]
    df["y_pred_raw_mw"] = df["y_pred_price"]
    df["actual_norm"] = df["y_true_raw_mw"] / CAPACITY_MW
    df["pred_norm"] = df["y_pred_raw_mw"] / CAPACITY_MW
    df["actual_clip"] = df["actual_norm"].clip(0.0, 1.0)
    df["pred_clip"] = df["pred_norm"].clip(0.0, 1.0)
    df["actual_logit"] = _safe_logit(df["actual_clip"])
    df["pred_logit"] = _safe_logit(df["pred_clip"])
    df["delta_true_mw"] = df["delta_true_price"]
    df["delta_pred_mw"] = df["delta_pred_price"]
    ordered = [
        "timestamp_utc",
        "split",
        "y_true_raw_mw",
        "y_pred_raw_mw",
        "capacity_mw",
        "timestamp_et",
        "operating_date_et",
        "hour_et",
        "origin_timestamp_et",
        "horizon_index",
        "actual_norm",
        "pred_norm",
        "actual_clip",
        "pred_clip",
        "actual_logit",
        "pred_logit",
        "y_true_price",
        "y_pred_price",
        "error_price",
        "error_value",
        "axis_true_u",
        "axis_pred_u",
        "delta_true_mw",
        "delta_pred_mw",
        "delta_error",
        "delta_true_price",
        "delta_pred_price",
    ]
    return df[ordered].sort_values(["timestamp_utc", "split"]).reset_index(drop=True)


def _build_plain_frame(kind: str) -> pd.DataFrame:
    frames = [_load_point_frame(kind, split) for split in ("train", "valid", "test")]
    df = pd.concat(frames, ignore_index=True)
    df = _add_common_error_columns(df)
    ordered = [
        "timestamp_utc",
        "split",
        "origin_timestamp_et",
        "horizon_index",
        "y_true_price",
        "y_pred_price",
        "timestamp_et",
        "operating_date_et",
        "hour_et",
        "error_price",
        "error_value",
        "axis_true_u",
        "axis_pred_u",
        "delta_true_price",
        "delta_pred_price",
        "delta_error",
    ]
    return df[ordered].sort_values(["timestamp_utc", "split"]).reset_index(drop=True)


def _build_spread_frame(da_df: pd.DataFrame, rt_df: pd.DataFrame) -> pd.DataFrame:
    da_keep = da_df[
        [
            "split",
            "origin_timestamp_et",
            "timestamp_utc",
            "timestamp_et",
            "operating_date_et",
            "hour_et",
            "horizon_index",
            "y_true_price",
            "y_pred_price",
        ]
    ].rename(columns={"y_true_price": "da_true", "y_pred_price": "da_pred"})

    rt_hourly = (
        rt_df.assign(timestamp_utc_hour=rt_df["timestamp_utc"].dt.floor("h"))
        .groupby(["split", "origin_timestamp_et", "operating_date_et", "hour_et", "timestamp_utc_hour"], as_index=False)
        .agg(
            rt_true=("y_true_price", "mean"),
            rt_pred=("y_pred_price", "mean"),
        )
        .rename(columns={"timestamp_utc_hour": "timestamp_utc"})
    )
    rt_hourly["timestamp_et"] = rt_hourly["timestamp_utc"].dt.tz_convert(TZ_ET)
    rt_hourly["horizon_index"] = (
        rt_hourly.sort_values(["origin_timestamp_et", "timestamp_utc"])
        .groupby("origin_timestamp_et")
        .cumcount()
        .astype(float)
    )

    merged = da_keep.merge(
        rt_hourly[
            [
                "split",
                "origin_timestamp_et",
                "timestamp_utc",
                "operating_date_et",
                "hour_et",
                "horizon_index",
                "timestamp_et",
                "rt_true",
                "rt_pred",
            ]
        ],
        on=["split", "origin_timestamp_et", "timestamp_utc", "operating_date_et", "hour_et", "horizon_index", "timestamp_et"],
        how="inner",
    )

    out = pd.DataFrame(
        {
            "timestamp_utc": merged["timestamp_utc"],
            "split": merged["split"],
            "origin_timestamp_et": merged["origin_timestamp_et"],
            "horizon_index": merged["horizon_index"],
            "y_true_price": merged["da_true"] - merged["rt_true"],
            "y_pred_price": merged["da_pred"] - merged["rt_pred"],
            "timestamp_et": merged["timestamp_et"],
            "operating_date_et": merged["operating_date_et"],
            "hour_et": merged["hour_et"],
        }
    )
    out = _add_common_error_columns(out)
    ordered = [
        "timestamp_utc",
        "split",
        "origin_timestamp_et",
        "horizon_index",
        "timestamp_et",
        "operating_date_et",
        "hour_et",
        "y_true_price",
        "y_pred_price",
        "error_price",
        "error_value",
        "axis_true_u",
        "axis_pred_u",
        "delta_true_price",
        "delta_pred_price",
        "delta_error",
    ]
    return out[ordered].sort_values(["timestamp_utc", "split"]).reset_index(drop=True)


def _summary_rows(df: pd.DataFrame, label: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    subsets = [("전체", df)]
    for split, sub_label in (("train", "학습기간"), ("test", "테스트기간")):
        sub = df[df["split"] == split].copy()
        if not sub.empty:
            subsets.append((sub_label, sub))
    for section, sub in subsets:
        err = pd.to_numeric(sub["error_value"], errors="coerce").dropna()
        abs_err = err.abs()
        rows.append(
            {
                "변수": label,
                "구간": section,
                "표본수": int(len(sub)),
                "실제값_평균": float(pd.to_numeric(sub["y_true_price"], errors="coerce").mean()),
                "예측값_평균": float(pd.to_numeric(sub["y_pred_price"], errors="coerce").mean()),
                "오차평균_Bias": float(err.mean()),
                "절대오차_MAE": float(abs_err.mean()),
                "오차_RMSE": float(np.sqrt(np.mean(err**2))),
                "오차표준편차": float(err.std(ddof=0)),
                "절대오차_p50": float(abs_err.quantile(0.50)),
                "절대오차_p90": float(abs_err.quantile(0.90)),
                "절대오차_p95": float(abs_err.quantile(0.95)),
                "최소오차": float(err.min()),
                "최대오차": float(err.max()),
            }
        )
    return rows


def main() -> None:
    args = _parse_args()
    global POINT_ROOT, OUT_ROOT
    POINT_ROOT = args.point_root.resolve()
    OUT_ROOT = args.out_root.resolve()
    if OUT_ROOT.exists() and any(OUT_ROOT.iterdir()):
        raise FileExistsError(f"Output directory must be empty: {OUT_ROOT}")
    _ensure_dir(OUT_ROOT)

    wind_df = _build_wind_frame()
    da_df = _build_plain_frame("da_energy")
    rt_df = _build_plain_frame("rt_energy")
    da_reg_df = _build_plain_frame("da_reg")
    rt_reg_df = _build_plain_frame("rt_reg")
    spread_df = _build_spread_frame(da_df, rt_df)

    dataset_map = {
        "wind": wind_df,
        "da_energy": da_df,
        "rt_energy": rt_df,
        "da_reg": da_reg_df,
        "rt_reg": rt_reg_df,
        "spread": spread_df,
    }

    meta: dict[str, object] = {
        "point_result_root": str(POINT_ROOT),
        "output_root": str(OUT_ROOT),
        "capacity_mw": CAPACITY_MW,
        "generated_files": {},
        "point_result_sources": POINT_FILE_NAMES,
    }

    all_summary_rows: list[dict[str, object]] = []
    for key, label, error_name, summary_name in SERIES_INFO:
        df = dataset_map[key].copy()
        error_path = OUT_ROOT / error_name
        summary_path = OUT_ROOT / summary_name
        df.to_csv(error_path, index=False, encoding="utf-8-sig")
        summary_df = pd.DataFrame(_summary_rows(df, label))
        summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
        all_summary_rows.extend(summary_df.to_dict("records"))
        meta["generated_files"][key] = {
            "error_csv": str(error_path),
            "summary_csv": str(summary_path),
            "rows": int(len(df)),
            "train_rows": int((df["split"] == "train").sum()),
            "test_rows": int((df["split"] == "test").sum()),
        }

    integrated_path = OUT_ROOT / "17_통합_오차분석요약.csv"
    pd.DataFrame(all_summary_rows).to_csv(integrated_path, index=False, encoding="utf-8-sig")
    meta["generated_files"]["all"] = {
        "summary_csv": str(integrated_path),
        "rows": int(len(all_summary_rows)),
    }
    _dump_json(OUT_ROOT / "18_점예측기반_오차데이터_메타정보.json", meta)


if __name__ == "__main__":
    main()
