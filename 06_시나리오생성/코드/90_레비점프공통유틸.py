#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path(__file__).resolve().parents[2]
FIT_ROOT = ROOT / "05_오차증분_분포적합" / "오차증분_분포적합결과"
JUMP_SUMMARY_CSV = FIT_ROOT / "51_통합_레비점프_점프요약.csv"
JUMP_STEP_CSV = FIT_ROOT / "52_통합_레비점프_점프단계요약.csv"

DEFAULT_JUMP_Q = 0.95
DEFAULT_MIN_BODY_N = 100


@dataclass
class LevyJumpModel:
    dataset_key: str
    step_col: str
    threshold_abs: float
    body_family: str
    body_param_1: float
    body_param_2: float
    global_jump_prob: float
    global_jump_values: np.ndarray
    step_jump_prob: dict[int, float]
    step_jump_values: dict[int, np.ndarray]

    def to_meta(self) -> dict[str, object]:
        return {
            "dataset_key": self.dataset_key,
            "step_col": self.step_col,
            "threshold_abs": float(self.threshold_abs),
            "body_family": self.body_family,
            "body_param_1": float(self.body_param_1),
            "body_param_2": float(self.body_param_2),
            "global_jump_prob": float(self.global_jump_prob),
            "global_jump_pool_n": int(len(self.global_jump_values)),
            "step_count": int(len(self.step_jump_prob)),
        }


def _select_train(df: pd.DataFrame) -> pd.DataFrame:
    train = df.loc[df["split"].astype(str).str.lower() == "train"].copy() if "split" in df.columns else pd.DataFrame()
    if train.empty:
        train = df.copy()
    train["delta_error"] = pd.to_numeric(train["delta_error"], errors="coerce")
    train = train.dropna(subset=["delta_error"]).copy()
    return train[np.isfinite(train["delta_error"].to_numpy(dtype=float))].copy()


def _step_column(df: pd.DataFrame) -> str:
    if "horizon_index" in df.columns:
        hz = pd.to_numeric(df["horizon_index"], errors="coerce")
        if hz.notna().sum() > 0:
            return "horizon_index"
    return "hour_et"


def _fit_body_candidate(body_values: np.ndarray) -> tuple[str, float, float]:
    gauss_mu, gauss_sigma = stats.norm.fit(body_values)
    laplace_loc, laplace_scale = stats.laplace.fit(body_values)
    gauss_sigma = max(float(gauss_sigma), 1e-8)
    laplace_scale = max(float(laplace_scale), 1e-8)
    ll_gauss = float(np.sum(stats.norm.logpdf(body_values, loc=gauss_mu, scale=gauss_sigma)))
    ll_laplace = float(np.sum(stats.laplace.logpdf(body_values, loc=laplace_loc, scale=laplace_scale)))
    if ll_laplace >= ll_gauss:
        return "laplace", float(laplace_loc), laplace_scale
    return "gaussian", float(gauss_mu), gauss_sigma


def _fit_specific_body(body_values: np.ndarray, family: str) -> tuple[float, float]:
    fam = str(family).strip().lower()
    if fam == "gaussian":
        mu, sigma = stats.norm.fit(body_values)
        return float(mu), max(float(sigma), 1e-8)
    if fam == "laplace":
        loc, scale = stats.laplace.fit(body_values)
        return float(loc), max(float(scale), 1e-8)
    raise ValueError(f"Unsupported levy-jump body family override: {family}")


def _load_jump_artifacts() -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    summary_df = pd.read_csv(JUMP_SUMMARY_CSV, encoding="utf-8-sig") if JUMP_SUMMARY_CSV.exists() else None
    step_df = pd.read_csv(JUMP_STEP_CSV, encoding="utf-8-sig") if JUMP_STEP_CSV.exists() else None
    return summary_df, step_df


def build_levy_jump_model(
    dataset_key: str,
    error_df: pd.DataFrame,
    body_family_override: str | None = None,
) -> LevyJumpModel:
    summary_df, step_df = _load_jump_artifacts()
    train = _select_train(error_df)
    if train.empty:
        raise ValueError(f"No usable rows for levy_jump model: {dataset_key}")
    override_family = str(body_family_override).strip().lower() if body_family_override else None
    if override_family and override_family not in {"gaussian", "laplace"}:
        raise ValueError(f"Unsupported body family override: {body_family_override}")

    if summary_df is not None and (summary_df["dataset_key"].astype(str) == dataset_key).any():
        row = summary_df.loc[summary_df["dataset_key"].astype(str) == dataset_key].iloc[0]
        step_col = str(row.get("step_col", "horizon_index"))
        threshold_abs = float(row.get("threshold_abs", np.nan))
        body_family = override_family or str(row.get("preferred_body_family", "laplace")).lower()
        if body_family == "gaussian":
            body_param_1 = float(row.get("body_gaussian_mu", 0.0))
            body_param_2 = max(float(row.get("body_gaussian_sigma", 1.0)), 1e-8)
        else:
            body_family = "laplace"
            body_param_1 = float(row.get("body_laplace_loc", 0.0))
            body_param_2 = max(float(row.get("body_laplace_scale", 1.0)), 1e-8)
        global_jump_prob = float(row.get("jump_rate", np.nan))
    else:
        step_col = _step_column(train)
        delta = train["delta_error"].to_numpy(dtype=float)
        threshold_abs = float(np.quantile(np.abs(delta), DEFAULT_JUMP_Q))
        jump_mask = np.abs(delta) > threshold_abs
        body_values = delta[~jump_mask]
        if len(body_values) < DEFAULT_MIN_BODY_N:
            body_values = delta
        if override_family:
            body_family = override_family
            body_param_1, body_param_2 = _fit_specific_body(body_values, body_family)
        else:
            body_family, body_param_1, body_param_2 = _fit_body_candidate(body_values)
        global_jump_prob = float(jump_mask.mean())

    train[step_col] = pd.to_numeric(train[step_col], errors="coerce")
    train = train.dropna(subset=[step_col]).copy()
    train["step_index"] = train[step_col].astype(int)
    jump_mask = np.abs(train["delta_error"].to_numpy(dtype=float)) > threshold_abs
    jump_values = train.loc[jump_mask, "delta_error"].to_numpy(dtype=float)
    if len(jump_values) == 0:
        jump_values = np.array([0.0], dtype=np.float64)

    step_jump_prob: dict[int, float] = {}
    step_jump_values: dict[int, np.ndarray] = {}
    if step_df is not None and (step_df["dataset_key"].astype(str) == dataset_key).any():
        sub = step_df.loc[step_df["dataset_key"].astype(str) == dataset_key].copy()
        for row in sub.itertuples(index=False):
            step_idx = int(row.step_index)
            step_jump_prob[step_idx] = float(row.jump_prob)
    for step_idx, sub in train.groupby("step_index", sort=True):
        vals = sub.loc[np.abs(sub["delta_error"].to_numpy(dtype=float)) > threshold_abs, "delta_error"].to_numpy(dtype=float)
        if len(vals) > 0:
            step_jump_values[int(step_idx)] = vals
        step_jump_prob.setdefault(int(step_idx), float(len(vals) / max(len(sub), 1)))

    return LevyJumpModel(
        dataset_key=dataset_key,
        step_col=step_col,
        threshold_abs=float(threshold_abs),
        body_family=body_family,
        body_param_1=float(body_param_1),
        body_param_2=float(body_param_2),
        global_jump_prob=float(global_jump_prob),
        global_jump_values=np.asarray(jump_values, dtype=np.float64),
        step_jump_prob=step_jump_prob,
        step_jump_values=step_jump_values,
    )


def sample_levy_jump_increments(
    rng: np.random.Generator,
    model: LevyJumpModel,
    size: tuple[int, int],
) -> np.ndarray:
    n_scenarios, n_steps = size
    if model.body_family == "gaussian":
        body = rng.normal(loc=model.body_param_1, scale=model.body_param_2, size=size)
    else:
        body = rng.laplace(loc=model.body_param_1, scale=model.body_param_2, size=size)
    increments = np.asarray(body, dtype=np.float64)
    for j, step_index in enumerate(range(1, n_steps + 1)):
        jump_prob = float(model.step_jump_prob.get(step_index, model.global_jump_prob))
        jump_prob = min(max(jump_prob, 0.0), 0.95)
        if jump_prob <= 0:
            continue
        flags = rng.random(n_scenarios) < jump_prob
        if not np.any(flags):
            continue
        pool = model.step_jump_values.get(step_index, model.global_jump_values)
        if len(pool) == 0:
            continue
        increments[flags, j] = rng.choice(pool, size=int(flags.sum()), replace=True).astype(np.float64)
    return increments
