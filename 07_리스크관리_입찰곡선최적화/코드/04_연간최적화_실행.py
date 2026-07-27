#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable=None, *args, **kwargs):
        return iterable


def _load_module_from_code_dir(module_name: str, pattern: str) -> object:
    code_dir = Path(__file__).resolve().parent
    matches = sorted(code_dir.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No module matching {pattern!r} found in {code_dir}")
    module_path = matches[0]
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load {module_name} from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


if sys.platform == "win32" and __name__ == "__main__":
    import io

    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")


COMMON = _load_module_from_code_dir("flat_offer_curve_common_util", "01_입력정산_공통유틸.py")
MARKET = _load_module_from_code_dir("market_curve_three_methods", "03_M0_M1_M2_최적화.py")
CAPACITY_MW = float(COMMON.CAPACITY_MW)


def _log(message: str) -> None:
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    writer = getattr(tqdm, "write", None)
    if callable(writer):
        writer(f"[{stamp}] {message}")
    else:
        print(f"[{stamp}] {message}", flush=True)


def _case_root(root: Path, case_name: str) -> Path:
    key = str(case_name).strip().lower()
    candidates = [
        root / COMMON.family_case_dir(key),
        root / COMMON.energy_joint_case_dir(key),
        root / COMMON.reg_joint_case_dir(key),
        root / f"joint_case_{key}",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[-1]


def _jsonable(obj: object) -> object:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, pd.Timestamp):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    return obj


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _filter_dates(dates: list[str], start: str, end: str) -> list[str]:
    out = []
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    for date in dates:
        ts = pd.Timestamp(str(date))
        if start_ts <= ts <= end_ts:
            out.append(str(date))
    return out


def _latest_reg_residual_root(opt_root: Path) -> Path | None:
    result_root = opt_root / "최적화결과"
    candidates = sorted(result_root.glob("*/00_REG오차조건부모형/02_조건부샘플링"))
    return candidates[-1] if candidates else None


def _select_hour_scenarios(day_df: pd.DataFrame, hour_et: int, n_scenarios: int) -> pd.DataFrame:
    hour_df = day_df.loc[day_df["hour_et"] == int(hour_et)].dropna(
        subset=["scenario_id", "wind_scn_mw", "da_energy_scn", "rt_energy_scn"]
    ).copy()
    if hour_df.empty:
        return hour_df
    hour_df = hour_df.sort_values(["da_energy_scn", "scenario_id"]).reset_index(drop=True)
    if int(n_scenarios) < len(hour_df):
        idx = np.linspace(0, len(hour_df) - 1, int(n_scenarios), dtype=int)
        hour_df = hour_df.iloc[idx].copy()
    return hour_df.sort_values("scenario_id").reset_index(drop=True)


def _build_penalty_prices(
    *,
    day_df: pd.DataFrame,
    selected_df: pd.DataFrame,
    reg_pred_day: pd.DataFrame,
    residual_artifacts: dict[str, object] | None,
    operating_date_et: str,
    hour_et: int,
    mode: str,
) -> np.ndarray:
    mode_key = str(mode).strip().lower()
    if mode_key == "none":
        return np.zeros(len(selected_df), dtype=np.float64)

    pred_row = reg_pred_day.loc[reg_pred_day["hour_et"] == int(hour_et)]
    if pred_row.empty:
        return np.zeros(len(selected_df), dtype=np.float64)
    da_reg_pred = float(pred_row.iloc[0]["da_reg_pred"])
    rt_reg_pred = float(pred_row.iloc[0]["rt_reg_pred"])
    point_penalty = max(0.0, da_reg_pred, rt_reg_pred)

    if mode_key in {"point", "point_reg", "point_forecast"}:
        return np.full(len(selected_df), point_penalty, dtype=np.float64)
    if mode_key != "conditional_residual":
        raise ValueError(f"Unsupported penalty mode: {mode!r}")
    if residual_artifacts is None:
        return np.full(len(selected_df), point_penalty, dtype=np.float64)

    delta_df = COMMON.extract_hourly_scenario_delta(day_df, int(hour_et), "da_energy_scn")
    eps_da, eps_rt = COMMON.sample_reg_residual_pair_hour(
        residual_artifacts,
        operating_date_et=str(operating_date_et),
        hour_et=int(hour_et),
        delta_da_values=delta_df["delta_value"].to_numpy(dtype=np.float64),
        base_seed=777,
    )
    penalty_df = delta_df.loc[:, ["scenario_id"]].copy()
    penalty_df["penalty_price_scn"] = np.maximum.reduce([
        np.zeros(len(delta_df), dtype=np.float64),
        da_reg_pred + eps_da,
        rt_reg_pred + eps_rt,
    ])
    merged = selected_df.loc[:, ["scenario_id"]].merge(penalty_df, on="scenario_id", how="left")
    return merged["penalty_price_scn"].fillna(point_penalty).to_numpy(dtype=np.float64)


def _market_award_one(segment_quantities: np.ndarray, bid_prices: np.ndarray, da_price: float) -> float:
    prices = np.asarray(bid_prices, dtype=np.float64)
    quantities = np.asarray(segment_quantities, dtype=np.float64)
    return float(np.sum(quantities * (float(da_price) >= prices)))


def _evaluate_realized_market_curve(
    *,
    actual_row: pd.Series,
    method: str,
    result: dict[str, object],
    penalty_band_basis: str,
) -> dict[str, object]:
    award = _market_award_one(result["block_quantities"], result["bid_prices"], float(actual_row["da_energy_actual"]))
    settlement = COMMON.settlement_amount(
        award_mw=award,
        actual_wind_mw=float(actual_row["wind_actual_mw"]),
        da_energy_price=float(actual_row["da_energy_actual"]),
        rt_energy_price=float(actual_row["rt_energy_actual"]),
        da_reg_price=float(actual_row["da_reg_actual"]),
        rt_reg_price=float(actual_row["rt_reg_actual"]),
        penalty_band_basis=penalty_band_basis,
        capacity_mw=CAPACITY_MW,
        penalty_settlement_mode="enabled",
    )
    return {
        "method": method,
        "operating_date_et": str(actual_row["operating_date_et"]),
        "hour_et": int(actual_row["hour_et"]),
        "timestamp_utc": str(actual_row["timestamp_utc"]),
        "award_mw": award,
        "actual_wind_mw": float(actual_row["wind_actual_mw"]),
        "point_forecast_mw": float(actual_row["wind_point_forecast_mw"]),
        "da_energy_actual": float(actual_row["da_energy_actual"]),
        "rt_energy_actual": float(actual_row["rt_energy_actual"]),
        "da_reg_actual": float(actual_row["da_reg_actual"]),
        "rt_reg_actual": float(actual_row["rt_reg_actual"]),
        **settlement,
    }


def _curve_rows(
    *,
    method: str,
    operating_date_et: str,
    hour_et: int,
    result: dict[str, object],
) -> list[dict[str, object]]:
    if bool(result.get("suppress_full_curve_output", False)):
        return []
    rows = []
    for row in MARKET._curve_point_rows(method, result):
        row.update({
            "operating_date_et": str(operating_date_et),
            "hour_et": int(hour_et),
            "solver": str(result["solver"]),
            "status": str(result["status"]),
            "objective": float(result["objective"]),
            "mean_profit": float(result["mean_profit"]),
            "mean_award_mw": float(result["mean_award_mw"]),
            "elapsed_sec": float(result.get("elapsed_sec", np.nan)),
            "n_scenarios": int(result["n_scenarios"]),
            "max_points": int(result["max_points"]),
            "max_segments": int(result["max_segments"]),
            "segment_slots": int(result.get("segment_slots", result["max_segments"])),
            "active_segments": int(result["active_segments"]),
        })
        rows.append(row)
    return rows


def _summary_row(
    *,
    method: str,
    operating_date_et: str,
    hour_et: int,
    result: dict[str, object],
) -> dict[str, object]:
    suppress_curve = bool(result.get("suppress_full_curve_output", False))
    bid_prices_for_json = [] if suppress_curve else result["bid_prices"]
    block_quantities_for_json = [] if suppress_curve else result["block_quantities"]
    cumulative_quantities_for_json = [] if suppress_curve else result["cumulative_quantities"]
    return {
        "method": method,
        "operating_date_et": str(operating_date_et),
        "hour_et": int(hour_et),
        "solver": str(result["solver"]),
        "status": str(result["status"]),
        "objective": float(result["objective"]),
        "mean_profit": float(result["mean_profit"]),
        "mean_award_mw": float(result["mean_award_mw"]),
        "mean_under_mw": float(result["mean_under_mw"]),
        "mean_under_cost": float(result["mean_under_cost"]),
        "elapsed_sec": float(result.get("elapsed_sec", np.nan)),
        "n_scenarios": int(result["n_scenarios"]),
        "max_points": int(result["max_points"]),
        "max_segments": int(result["max_segments"]),
        "segment_slots": int(result.get("segment_slots", result["max_segments"])),
        "active_segments": int(result["active_segments"]),
        "price_support_mode": str(result.get("price_support_mode", "")),
        "projection_price_selection": str(result.get("projection_price_selection", "")),
        "capacity_constraint": str(result.get("capacity_constraint", "")),
        "total_offered_capacity_mw": float(result.get("total_offered_capacity_mw", np.nan)),
        "relaxed_upper_bound": bool(result.get("relaxed_upper_bound", False)),
        "curve_output_suppressed": suppress_curve,
        "best_bound": float(result.get("best_bound", np.nan)),
        "mip_relative_gap": float(result.get("mip_relative_gap", np.nan)),
        "mip_gap_target": float(result.get("mip_gap_target", np.nan)),
        "quantity_variability_lambda": float(result.get("quantity_variability_lambda", 0.0)),
        "quantity_variability_lambda_dollar": float(result.get("quantity_variability_lambda_dollar", 0.0)),
        "quantity_variability_penalty": float(result.get("quantity_variability_penalty", 0.0)),
        "regularization_penalty_amount": float(result.get("regularization_penalty_amount", 0.0)),
        "front_loading_lambda": float(result.get("front_loading_lambda", 0.0)),
        "front_loading_lambda_dollar": float(result.get("front_loading_lambda_dollar", 0.0)),
        "front_loading_penalty": float(result.get("front_loading_penalty", 0.0)),
        "front_loading_penalty_mw": float(result.get("front_loading_penalty_mw", 0.0)),
        "front_loading_penalty_amount": float(result.get("front_loading_penalty_amount", 0.0)),
        "front_loading_max_violation_mw": float(result.get("front_loading_max_violation_mw", 0.0)),
        "objective_unregularized": float(result.get("objective_unregularized", result.get("mean_profit", np.nan))),
        "risk_objective": str(result.get("risk_objective", "mean")),
        "cvar_alpha": float(result.get("cvar_alpha", np.nan)),
        "cvar_weight": float(result.get("cvar_weight", 0.0)),
        "scenario_cvar_profit": float(result.get("scenario_cvar_profit", np.nan)),
        "scenario_cvar_eta": float(result.get("scenario_cvar_eta", np.nan)),
        "scenario_loss_cvar": float(result.get("scenario_loss_cvar", np.nan)),
        "scenario_loss_cvar_eta": float(result.get("scenario_loss_cvar_eta", np.nan)),
        "mean_tail_loss": float(result.get("mean_tail_loss", np.nan)),
        "mean_rt_loss": float(result.get("mean_rt_loss", np.nan)),
        "mean_penalty_loss": float(result.get("mean_penalty_loss", np.nan)),
        "ensemble_count": int(result.get("ensemble_count", 0)),
        "ensemble_risk_weight": float(result.get("ensemble_risk_weight", 0.0)),
        "ensemble_tail_alpha": float(result.get("ensemble_tail_alpha", 0.0)),
        "ensemble_mean_profit": float(result.get("ensemble_mean_profit", np.nan)),
        "ensemble_min_profit": float(result.get("ensemble_min_profit", np.nan)),
        "ensemble_cvar_profit": float(result.get("ensemble_cvar_profit", np.nan)),
        "ensemble_profit_std": float(result.get("ensemble_profit_std", np.nan)),
        "ensemble_mode": str(result.get("ensemble_mode", "")),
        "bagged_fold_mean_profit": float(result.get("bagged_fold_mean_profit", np.nan)),
        "bagged_fold_min_profit": float(result.get("bagged_fold_min_profit", np.nan)),
        "bagged_fold_profit_std": float(result.get("bagged_fold_profit_std", np.nan)),
        "support_candidate_count": int(result.get("support_candidate_count", 0)),
        "support_selected_candidate": int(result.get("support_selected_candidate", -1)),
        "support_train_size": int(result.get("support_train_size", 0)),
        "support_validation_size": int(result.get("support_validation_size", 0)),
        "support_validation_profit": float(result.get("support_validation_profit", np.nan)),
        "support_uniform_validation_profit": float(result.get("support_uniform_validation_profit", np.nan)),
        "support_validation_gain": float(result.get("support_validation_gain", np.nan)),
        "support_validation_score": float(result.get("support_validation_score", np.nan)),
        "support_price_displacement_gamma": float(result.get("support_price_displacement_gamma", np.nan)),
        "support_price_displacement_scale": float(result.get("support_price_displacement_scale", np.nan)),
        "support_price_displacement_norm": float(result.get("support_price_displacement_norm", np.nan)),
        "support_price_displacement_penalty": float(result.get("support_price_displacement_penalty", np.nan)),
        "support_shift_fraction": float(result.get("support_shift_fraction", np.nan)),
        "support_max_gap_multiplier": float(result.get("support_max_gap_multiplier", np.nan)),
        "support_validation_fraction": float(result.get("support_validation_fraction", np.nan)),
        "support_validation_mode": str(result.get("support_validation_mode", "")),
        "support_candidate_mode": str(result.get("support_candidate_mode", "")),
        "support_max_gap": float(result.get("support_max_gap", np.nan)),
        "support_base_gap": float(result.get("support_base_gap", np.nan)),
        "support_max_gap_ratio": float(result.get("support_max_gap_ratio", np.nan)),
        "bid_prices_json": json.dumps(_jsonable(bid_prices_for_json), ensure_ascii=False),
        "segment_quantities_json": json.dumps(_jsonable(block_quantities_for_json), ensure_ascii=False),
        "cumulative_quantities_json": json.dumps(_jsonable(cumulative_quantities_for_json), ensure_ascii=False),
    }


def _parse_json_float_array(text: object) -> np.ndarray:
    if text is None:
        return np.asarray([], dtype=np.float64)
    if isinstance(text, float) and np.isnan(text):
        return np.asarray([], dtype=np.float64)
    raw = str(text).strip()
    if not raw:
        return np.asarray([], dtype=np.float64)
    try:
        vals = json.loads(raw)
    except json.JSONDecodeError:
        return np.asarray([], dtype=np.float64)
    return np.asarray(vals, dtype=np.float64)


def _load_warm_start_prices(root: Path | str | None) -> dict[tuple[str, int], np.ndarray]:
    if root is None:
        return {}
    root_path = Path(root)
    if not str(root_path).strip() or not root_path.exists():
        return {}

    summary_path = root_path / "02_optimization_summary_by_hour.csv"
    if summary_path.exists():
        df = pd.read_csv(summary_path)
        required = {"operating_date_et", "hour_et", "bid_prices_json"}
        if required.issubset(df.columns):
            out: dict[tuple[str, int], np.ndarray] = {}
            for row in df.to_dict("records"):
                prices = _parse_json_float_array(row.get("bid_prices_json"))
                prices = prices[np.isfinite(prices)]
                if len(prices):
                    out[(str(row["operating_date_et"]), int(row["hour_et"]))] = np.sort(prices)
            if out:
                return out

    curve_path = root_path / "01_market_curve_points_by_hour.csv"
    if curve_path.exists():
        df = pd.read_csv(curve_path)
        required = {"operating_date_et", "hour_et", "segment_index", "bid_price"}
        if required.issubset(df.columns):
            work = df.copy()
            if "point_type" in work.columns:
                work = work.loc[work["point_type"].astype(str) != "anchor"].copy()
            out: dict[tuple[str, int], np.ndarray] = {}
            for (date, hour), sub in work.groupby(["operating_date_et", "hour_et"], sort=False):
                sub = sub.sort_values("segment_index")
                prices = pd.to_numeric(sub["bid_price"], errors="coerce").dropna().to_numpy(dtype=np.float64)
                if len(prices):
                    out[(str(date), int(hour))] = np.sort(prices)
            return out
    return {}


def _solve_method(
    *,
    method: str,
    wind: np.ndarray,
    da: np.ndarray,
    rt: np.ndarray,
    pen: np.ndarray,
    max_points: int,
    dense_segments: int,
    price_grid_size: int,
    price_tick: float,
    min_segment_mw: float,
    direct_mip_gap: float,
    cplex_time_limit_sec: float,
    direct_time_limit_sec: float,
    cplex_threads: int,
    warm_start_prices: np.ndarray | None = None,
    projection_price_selection: str = "fixed_quantile",
    price_support_mode: str = "quantile",
    capacity_constraint: str = "eq",
    quantity_variability_lambda: float = 0.0,
    front_loading_lambda: float = 0.0,
    risk_objective: str = "mean",
    cvar_alpha: float = 0.05,
    cvar_weight: float = 0.0,
    ensemble_count: int = 5,
    ensemble_risk_weight: float = 0.5,
    ensemble_tail_alpha: float = 0.2,
    ensemble_mode: str = "da_stratified",
    support_shift_fraction: float = 0.50,
    support_max_gap_multiplier: float = 1.50,
    support_validation_fraction: float = 0.25,
    support_validation_mode: str = "index",
    support_candidate_mode: str = "full",
    support_price_displacement_gamma: float = 0.0,
) -> dict[str, object]:
    tolerance = 0.03 * CAPACITY_MW
    return MARKET.solve_market_curve_method(
        method=method,
        wind=wind,
        da=da,
        rt=rt,
        pen=pen,
        max_points=max_points,
        dense_segments=dense_segments,
        price_grid_size=price_grid_size,
        price_tick=price_tick,
        min_segment_mw=min_segment_mw,
        capacity_mw=CAPACITY_MW,
        tolerance_mw=tolerance,
        cplex_time_limit_sec=cplex_time_limit_sec,
        direct_time_limit_sec=direct_time_limit_sec,
        cplex_threads=cplex_threads,
        solver_log=False,
        direct_mip_gap=direct_mip_gap,
        warm_start_prices=warm_start_prices,
        projection_price_selection=projection_price_selection,
        price_support_mode=price_support_mode,
        capacity_constraint=capacity_constraint,
        quantity_variability_lambda=quantity_variability_lambda,
        front_loading_lambda=front_loading_lambda,
        risk_objective=risk_objective,
        cvar_alpha=cvar_alpha,
        cvar_weight=cvar_weight,
        ensemble_count=ensemble_count,
        ensemble_risk_weight=ensemble_risk_weight,
        ensemble_tail_alpha=ensemble_tail_alpha,
        ensemble_mode=ensemble_mode,
        support_shift_fraction=support_shift_fraction,
        support_max_gap_multiplier=support_max_gap_multiplier,
        support_validation_fraction=support_validation_fraction,
        support_validation_mode=support_validation_mode,
        support_candidate_mode=support_candidate_mode,
        support_price_displacement_gamma=support_price_displacement_gamma,
    )


def _write_outputs(
    out_root: Path,
    curve_rows: list[dict[str, object]],
    summary_rows: list[dict[str, object]],
    detail_rows: list[dict[str, object]],
    error_rows: list[dict[str, object]],
) -> None:
    out_root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(curve_rows).to_csv(out_root / "01_market_curve_points_by_hour.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(summary_rows).to_csv(out_root / "02_optimization_summary_by_hour.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(detail_rows).to_csv(out_root / "03_realized_settlement_by_hour.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(error_rows).to_csv(out_root / "99_errors.csv", index=False, encoding="utf-8-sig")
    if detail_rows:
        annual = COMMON.summarize_settlement(pd.DataFrame(detail_rows).assign(strategy_name=lambda x: x["method"]))
        annual.to_csv(out_root / "04_annual_realized_settlement.csv", index=False, encoding="utf-8-sig")


def _load_existing_outputs(
    out_root: Path,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], set[tuple[str, int]]]:
    curve_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    detail_rows: list[dict[str, object]] = []
    error_rows: list[dict[str, object]] = []
    completed: set[tuple[str, int]] = set()

    curve_path = out_root / "01_market_curve_points_by_hour.csv"
    summary_path = out_root / "02_optimization_summary_by_hour.csv"
    detail_path = out_root / "03_realized_settlement_by_hour.csv"
    error_path = out_root / "99_errors.csv"

    if curve_path.exists():
        curve_rows = pd.read_csv(curve_path).to_dict("records")
    if summary_path.exists():
        summary_rows = pd.read_csv(summary_path).to_dict("records")
        for row in summary_rows:
            completed.add((str(row["operating_date_et"]), int(row["hour_et"])))
    if detail_path.exists():
        detail_rows = pd.read_csv(detail_path).to_dict("records")
    if error_path.exists():
        try:
            error_rows = pd.read_csv(error_path).to_dict("records")
            for row in error_rows:
                completed.add((str(row["operating_date_et"]), int(row["hour_et"])))
        except pd.errors.EmptyDataError:
            error_rows = []

    return curve_rows, summary_rows, detail_rows, error_rows, completed


def run(args: argparse.Namespace) -> None:
    roots = COMMON.default_source_roots(Path(__file__).resolve())
    if str(getattr(args, "scenario_root_name", "")).strip():
        project_roots = COMMON.resolve_project_roots(Path(__file__).resolve())
        scenario_root = project_roots["scenario_root"] / str(args.scenario_root_name).strip()
        roots["wind_scenario_root"] = scenario_root / "__풍력시나리오__"
        roots["energy_joint_scenario_root"] = scenario_root / "__DA_RT에너지결합시나리오__"
        roots["reg_joint_scenario_root"] = scenario_root / "__DA_REG_RT_REG결합시나리오__"
    family = str(args.family).strip().lower()
    method = str(args.method).strip().lower()
    out_root = Path(args.out_root)
    if not out_root.is_absolute():
        out_root = Path.cwd() / out_root
    out_root.mkdir(parents=True, exist_ok=True)
    protected_outputs = [
        out_root / "01_market_curve_points_by_hour.csv",
        out_root / "02_optimization_summary_by_hour.csv",
        out_root / "03_realized_settlement_by_hour.csv",
        out_root / "04_annual_realized_settlement.csv",
    ]
    if any(path.exists() for path in protected_outputs) and not args.resume_existing and not args.overwrite_existing:
        raise FileExistsError(
            f"Output files already exist in {out_root}. Use a new --out-root, --resume-existing, "
            "or explicitly pass --overwrite-existing."
        )

    wind_root = _case_root(roots["wind_scenario_root"], family)
    energy_root = _case_root(roots["energy_joint_scenario_root"], family)
    dates = COMMON.list_available_3block_joint_dates(wind_root, energy_root)
    dates = _filter_dates(dates, args.start, args.end)
    if int(args.max_days) > 0:
        dates = dates[: int(args.max_days)]

    actual = COMMON.build_validation_actual_bundle(
        roots["wind_error_csv"],
        roots["da_energy_error_csv"],
        roots["rt_energy_error_csv"],
        roots["da_reg_error_csv"],
        roots["rt_reg_error_csv"],
        start=args.start,
        end=args.end,
    )
    actual_by_key = {
        (str(row["operating_date_et"]), int(row["hour_et"])): row
        for row in actual.to_dict("records")
    }
    date_hours: list[tuple[str, int]] = []
    for date in dates:
        hours = sorted({int(k[1]) for k in actual_by_key if k[0] == str(date)})
        for hour in hours:
            date_hours.append((str(date), int(hour)))
    if int(args.max_hours) > 0:
        date_hours = date_hours[: int(args.max_hours)]

    explicit_reg_point_file = str(getattr(args, "reg_point_forecast_file", "")).strip()
    reg_point_source = Path(explicit_reg_point_file) if explicit_reg_point_file else roots["reg_point_forecast_root"]
    if explicit_reg_point_file and not reg_point_source.is_file():
        raise FileNotFoundError(f"Explicit REG point-forecast file not found: {reg_point_source}")
    reg_point_source = reg_point_source.resolve()
    reg_pred = COMMON.load_reg_point_forecast_predictions(reg_point_source, args.start, args.end)
    reg_key_counts = reg_pred.groupby(["operating_date_et", "hour_et"], dropna=False).size()
    missing_reg_keys = [key for key in date_hours if key not in reg_key_counts.index]
    duplicate_reg_keys = [key for key in date_hours if int(reg_key_counts.get(key, 0)) != 1]
    if missing_reg_keys or duplicate_reg_keys:
        raise ValueError(
            "REG point-forecast coverage must be exactly one row per optimization hour; "
            f"missing={missing_reg_keys[:5]}, non_unique={duplicate_reg_keys[:5]}"
        )
    required_reg = reg_pred.set_index(["operating_date_et", "hour_et"]).loc[date_hours]
    if not np.isfinite(required_reg[["da_reg_pred", "rt_reg_pred"]].to_numpy(dtype=np.float64)).all():
        raise ValueError("REG point-forecast input contains non-finite predictions for required optimization hours.")
    residual_root = Path(args.reg_residual_root) if args.reg_residual_root else _latest_reg_residual_root(roots["out_root"])
    residual_artifacts = None
    if str(args.penalty_mode).strip().lower() == "conditional_residual" and residual_root is not None:
        residual_artifacts = COMMON.load_reg_residual_conditional_model(residual_root)
    requested_penalty_mode = str(args.penalty_mode).strip().lower()
    effective_penalty_mode = requested_penalty_mode
    if requested_penalty_mode == "conditional_residual" and residual_artifacts is None:
        effective_penalty_mode = "point_reg_fallback"
    warm_start_by_key = _load_warm_start_prices(args.warm_start_root)

    meta = {
        "mode": "market_curve_full_year_runner",
        "family": family,
        "scenario_root_name": str(getattr(args, "scenario_root_name", "")),
        "wind_scenario_root": str(roots["wind_scenario_root"]),
        "energy_joint_scenario_root": str(roots["energy_joint_scenario_root"]),
        "method": method,
        "start": args.start,
        "end": args.end,
        "n_dates": len(dates),
        "n_hour_cases": len(date_hours),
        "n_scenarios": int(args.n_scenarios),
        "max_points": int(args.max_points),
        "max_segments": int(MARKET.max_segments_from_points(int(args.max_points))),
        "dense_segments": int(args.dense_segments),
        "price_grid_size": int(args.price_grid_size),
        "price_tick": float(args.price_tick),
        "min_segment_mw": float(args.min_segment_mw),
        "direct_mip_gap": float(args.direct_mip_gap),
        "projection_price_selection": str(args.projection_price_selection),
        "price_support_mode": str(args.price_support_mode),
        "capacity_constraint": str(args.capacity_constraint),
        "quantity_variability_lambda": float(args.quantity_variability_lambda),
        "front_loading_lambda": float(args.front_loading_lambda),
        "risk_objective": str(args.risk_objective),
        "cvar_alpha": float(args.cvar_alpha),
        "cvar_weight": float(args.cvar_weight),
        "ensemble_count": int(args.ensemble_count),
        "ensemble_risk_weight": float(args.ensemble_risk_weight),
        "ensemble_tail_alpha": float(args.ensemble_tail_alpha),
        "ensemble_mode": str(args.ensemble_mode),
        "support_shift_fraction": float(args.support_shift_fraction),
        "support_max_gap_multiplier": float(args.support_max_gap_multiplier),
        "support_validation_fraction": float(args.support_validation_fraction),
        "support_validation_mode": str(args.support_validation_mode),
        "support_candidate_mode": str(args.support_candidate_mode),
        "support_price_displacement_gamma": float(args.support_price_displacement_gamma),
        "warm_start_root": str(args.warm_start_root),
        "n_warm_start_hour_cases": int(len(warm_start_by_key)),
        "penalty_mode": args.penalty_mode,
        "penalty_mode_effective": effective_penalty_mode,
        "penalty_price_multiplier": float(args.penalty_price_multiplier),
        "reg_point_forecast_source": str(reg_point_source),
        "reg_point_forecast_sha256": _sha256_file(reg_point_source) if reg_point_source.is_file() else "",
        "reg_residual_root_resolved": str(residual_root) if residual_root is not None else "",
        "reg_residual_loaded": bool(residual_artifacts is not None),
        "cplex_time_limit_sec": float(args.cplex_time_limit_sec),
        "direct_time_limit_sec": float(args.direct_time_limit_sec),
        "cplex_threads": int(args.cplex_threads),
        "out_root": str(out_root),
    }
    meta_path = out_root / "00_run_meta.json"
    if bool(args.resume_existing):
        if not meta_path.is_file():
            raise FileNotFoundError(f"Cannot resume without existing run metadata: {meta_path}")
        old_meta = json.loads(meta_path.read_text(encoding="utf-8"))
        compatibility_keys = [
            "family", "scenario_root_name", "method", "start", "end", "n_scenarios",
            "max_points", "dense_segments", "price_grid_size", "projection_price_selection",
            "price_support_mode", "capacity_constraint", "penalty_mode_effective",
            "penalty_price_multiplier", "reg_point_forecast_sha256",
        ]
        incompatible = {
            key: {"existing": old_meta.get(key), "requested": meta.get(key)}
            for key in compatibility_keys
            if old_meta.get(key) != meta.get(key)
        }
        if incompatible:
            raise ValueError(f"Resume metadata is incompatible with the requested run: {incompatible}")
    meta_path.write_text(json.dumps(_jsonable(meta), ensure_ascii=False, indent=2), encoding="utf-8")

    _log(
        f"full-year market-curve start method={method}, family={family}, "
        f"dates={len(dates)}, hour_cases={len(date_hours)}, scenarios={int(args.n_scenarios)}, "
        f"max_points={int(args.max_points)}, warm_starts={len(warm_start_by_key)}"
    )

    if bool(args.resume_existing):
        curve_rows, summary_rows, detail_rows, error_rows, completed_keys = _load_existing_outputs(out_root)
        if completed_keys:
            _log(f"resume enabled: loaded {len(summary_rows)} solved rows and {len(error_rows)} error rows from {out_root}")
    else:
        curve_rows = []
        summary_rows = []
        detail_rows = []
        error_rows = []
        completed_keys: set[tuple[str, int]] = set()

    day_cache: dict[str, pd.DataFrame] = {}
    last_date = None
    progress = tqdm(date_hours, total=len(date_hours), desc=f"{method}_market_curve_full_year", unit="hour-case")
    for idx, (date, hour) in enumerate(progress, start=1):
        if (str(date), int(hour)) in completed_keys:
            progress.set_postfix({"date": date, "hour": hour, "status": "resume-skip"})
            continue
        try:
            if date != last_date:
                _log(f"loading day {date}")
                day_cache = {date: COMMON.load_3block_wind_energy_day(wind_root, energy_root, date)}
                last_date = date
            day_df = day_cache[date]
            selected = _select_hour_scenarios(day_df, hour, int(args.n_scenarios))
            if selected.empty:
                raise ValueError(f"No scenario rows for date={date}, hour={hour}")
            reg_pred_day = reg_pred.loc[reg_pred["operating_date_et"].astype(str) == str(date)].copy()
            pen = _build_penalty_prices(
                day_df=day_df,
                selected_df=selected,
                reg_pred_day=reg_pred_day,
                residual_artifacts=residual_artifacts,
                operating_date_et=date,
                hour_et=hour,
                mode=args.penalty_mode,
            )
            pen = np.maximum(0.0, pen * float(args.penalty_price_multiplier))
            wind = selected["wind_scn_mw"].to_numpy(dtype=np.float64)
            da = selected["da_energy_scn"].to_numpy(dtype=np.float64)
            rt = selected["rt_energy_scn"].to_numpy(dtype=np.float64)
            result = _solve_method(
                method=method,
                wind=wind,
                da=da,
                rt=rt,
                pen=pen,
                max_points=int(args.max_points),
                dense_segments=int(args.dense_segments),
                price_grid_size=int(args.price_grid_size),
                price_tick=float(args.price_tick),
                min_segment_mw=float(args.min_segment_mw),
                direct_mip_gap=float(args.direct_mip_gap),
                cplex_time_limit_sec=float(args.cplex_time_limit_sec),
                direct_time_limit_sec=float(args.direct_time_limit_sec),
                cplex_threads=int(args.cplex_threads),
                warm_start_prices=warm_start_by_key.get((str(date), int(hour))),
                projection_price_selection=str(args.projection_price_selection),
                price_support_mode=str(args.price_support_mode),
                capacity_constraint=str(args.capacity_constraint),
                quantity_variability_lambda=float(args.quantity_variability_lambda),
                front_loading_lambda=float(args.front_loading_lambda),
                risk_objective=str(args.risk_objective),
                cvar_alpha=float(args.cvar_alpha),
                cvar_weight=float(args.cvar_weight),
                ensemble_count=int(args.ensemble_count),
                ensemble_risk_weight=float(args.ensemble_risk_weight),
                ensemble_tail_alpha=float(args.ensemble_tail_alpha),
                ensemble_mode=str(args.ensemble_mode),
                support_shift_fraction=float(args.support_shift_fraction),
                support_max_gap_multiplier=float(args.support_max_gap_multiplier),
                support_validation_fraction=float(args.support_validation_fraction),
                support_validation_mode=str(args.support_validation_mode),
                support_candidate_mode=str(args.support_candidate_mode),
                support_price_displacement_gamma=float(args.support_price_displacement_gamma),
            )
            curve_rows.extend(_curve_rows(method=method, operating_date_et=date, hour_et=hour, result=result))
            summary_rows.append(_summary_row(method=method, operating_date_et=date, hour_et=hour, result=result))
            actual_row = pd.Series(actual_by_key[(date, hour)])
            detail_rows.append(_evaluate_realized_market_curve(
                actual_row=actual_row,
                method=method,
                result=result,
                penalty_band_basis=args.penalty_band_basis,
            ))
            progress.set_postfix({
                "date": date,
                "hour": hour,
                "status": str(result["status"])[:18],
                "profit": f"{float(result['mean_profit']):.1f}",
                "seg": int(result["active_segments"]),
            })
        except Exception as exc:
            error_rows.append({
                "operating_date_et": date,
                "hour_et": int(hour),
                "method": method,
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            })
            progress.set_postfix({"date": date, "hour": hour, "error": type(exc).__name__})

        if int(args.save_every) > 0 and idx % int(args.save_every) == 0:
            _write_outputs(out_root, curve_rows, summary_rows, detail_rows, error_rows)
            _log(f"checkpoint saved {idx}/{len(date_hours)} hour-cases to {out_root}")

    _write_outputs(out_root, curve_rows, summary_rows, detail_rows, error_rows)
    _log(f"finished method={method}; solved={len(summary_rows)}, errors={len(error_rows)}, out={out_root}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run full-year market-compatible offer-curve optimization.")
    parser.add_argument("--family", default="laplace")
    parser.add_argument("--scenario-root-name", default="", help="Optional scenario result folder under 06_시나리오생성, e.g. 시나리오생성결과_s5000_laplace.")
    parser.add_argument("--method", choices=["fixed", "relaxed", "projection"], required=True)
    parser.add_argument("--start", default="2021-01-01")
    parser.add_argument("--end", default="2021-12-31")
    parser.add_argument("--n-scenarios", type=int, default=3000)
    parser.add_argument("--max-points", type=int, default=11)
    parser.add_argument("--dense-segments", type=int, default=51)
    parser.add_argument("--price-grid-size", type=int, default=51)
    parser.add_argument("--price-tick", type=float, default=0.0)
    parser.add_argument("--min-segment-mw", type=float, default=0.0)
    parser.add_argument("--capacity-constraint", choices=["eq", "le"], default="eq")
    parser.add_argument("--quantity-variability-lambda", type=float, default=0.0)
    parser.add_argument("--front-loading-lambda", type=float, default=0.0)
    parser.add_argument("--risk-objective", choices=["mean", "mean_cvar", "cvar", "mean_loss_cvar", "loss_cvar"], default="mean")
    parser.add_argument("--cvar-alpha", type=float, default=0.05)
    parser.add_argument("--cvar-weight", type=float, default=0.0)
    parser.add_argument("--ensemble-count", type=int, default=5)
    parser.add_argument("--ensemble-risk-weight", type=float, default=0.5)
    parser.add_argument("--ensemble-tail-alpha", type=float, default=0.2)
    parser.add_argument("--ensemble-mode", choices=["da_stratified", "index"], default="da_stratified")
    parser.add_argument("--support-shift-fraction", type=float, default=0.50)
    parser.add_argument("--support-max-gap-multiplier", type=float, default=1.50)
    parser.add_argument("--support-validation-fraction", type=float, default=0.25)
    parser.add_argument("--support-validation-mode", choices=["da_stratified", "index"], default="index")
    parser.add_argument("--support-candidate-mode", choices=["full", "upper_tail_plus_tilt", "tilt_only"], default="full")
    parser.add_argument("--support-price-displacement-gamma", type=float, default=0.0)
    parser.add_argument("--direct-mip-gap", type=float, default=1e-3)
    parser.add_argument(
        "--projection-price-selection",
        choices=["fixed_quantile", "frequency_weighted", "shape_preserving", "uniform_shape"],
        default="fixed_quantile",
    )
    parser.add_argument("--price-support-mode", choices=["quantile", "uniform"], default="quantile")
    parser.add_argument("--penalty-mode", choices=["conditional_residual", "point_reg", "none"], default="conditional_residual")
    parser.add_argument("--penalty-price-multiplier", type=float, default=1.0)
    parser.add_argument("--penalty-band-basis", choices=["capacity", "award"], default="capacity")
    parser.add_argument("--cplex-time-limit-sec", type=float, default=300.0)
    parser.add_argument("--direct-time-limit-sec", type=float, default=600.0)
    parser.add_argument("--cplex-threads", type=int, default=4)
    parser.add_argument("--save-every", type=int, default=24)
    parser.add_argument("--max-days", type=int, default=0)
    parser.add_argument("--max-hours", type=int, default=0)
    parser.add_argument("--reg-residual-root", default="")
    parser.add_argument("--reg-point-forecast-file", default="")
    parser.add_argument("--warm-start-root", default="")
    parser.add_argument("--resume-existing", action="store_true")
    parser.add_argument("--overwrite-existing", action="store_true")
    parser.add_argument("--out-root", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
