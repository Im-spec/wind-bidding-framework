#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


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
BASE = _load_module_from_code_dir("step_offer_base", "02_입찰곡선_기초연산.py")
CAPACITY_MW = float(COMMON.CAPACITY_MW)


def _curve_point_rows(method: str, result: dict[str, object]) -> list[dict[str, object]]:
    """Serialize an optimized step curve into its anchor and segment-end points."""
    prices = np.asarray(result["bid_prices"], dtype=np.float64)
    blocks = np.asarray(result["block_quantities"], dtype=np.float64)
    cumulative = np.cumsum(blocks)
    if len(prices) == 0:
        return []
    rows: list[dict[str, object]] = [{
        "method": method,
        "point_index": 1,
        "segment_index": 0,
        "point_type": "anchor",
        "bid_price": float(prices[0]),
        "segment_quantity_mw": 0.0,
        "cumulative_quantity_mw": 0.0,
    }]
    for index, price in enumerate(prices):
        rows.append({
            "method": method,
            "point_index": int(index + 2),
            "segment_index": int(index + 1),
            "point_type": "segment_end",
            "bid_price": float(price),
            "segment_quantity_mw": float(blocks[index]),
            "cumulative_quantity_mw": float(cumulative[index]),
        })
    return rows


def _log(message: str) -> None:
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{stamp}] {message}", flush=True)


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


def max_segments_from_points(max_points: int) -> int:
    """Number of submitted step-offer price/MW pairs.

    Earlier experiments used a piecewise-linear convention where seven knots
    implied six segments.  The current NYISO-style step-offer experiments use
    the submitted MW-price pairs directly, so ``max_points`` is the number of
    step-offer slots.
    """
    points = int(max_points)
    if points < 1:
        raise ValueError("max_points must be at least 1.")
    return points


def price_quantile_grid(values: np.ndarray, n_prices: int, price_tick: float = 0.0) -> np.ndarray:
    vals = np.asarray(values, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        raise ValueError("No finite DA prices for price grid.")
    n = int(n_prices)
    if n < 1:
        raise ValueError("n_prices must be positive.")

    grid = np.quantile(vals, np.linspace(0.0, 1.0, n))
    tick = float(price_tick)
    if tick > 0:
        grid = np.round(grid / tick) * tick
    return np.sort(np.asarray(grid, dtype=np.float64))


def price_uniform_grid(values: np.ndarray, n_prices: int, price_tick: float = 0.0) -> np.ndarray:
    vals = np.asarray(values, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        raise ValueError("No finite DA prices for price grid.")
    n = int(n_prices)
    if n < 1:
        raise ValueError("n_prices must be positive.")

    lo = float(np.min(vals))
    hi = float(np.max(vals))
    if np.isclose(lo, hi):
        lo -= 1.0
        hi += 1.0
    grid = np.linspace(lo, hi, n)
    tick = float(price_tick)
    if tick > 0:
        grid = np.round(grid / tick) * tick
    return np.sort(np.asarray(grid, dtype=np.float64))


def _make_price_grid(values: np.ndarray, n_prices: int, price_tick: float, mode: str) -> np.ndarray:
    key = str(mode).strip().lower()
    if key in {"uniform", "linear", "linspace", "range"}:
        return price_uniform_grid(values, n_prices, price_tick=price_tick)
    if key in {"quantile", "scenario_quantile", "da_quantile"}:
        return price_quantile_grid(values, n_prices, price_tick=price_tick)
    raise ValueError(f"Unknown price grid mode: {mode!r}")


def _add_capacity_constraint(mdl, total_expr, capacity_mw: float, capacity_constraint: str) -> None:
    key = str(capacity_constraint).strip().lower()
    if key in {"le", "<=", "at_most", "upper_bound", "partial"}:
        mdl.add_constraint(total_expr <= float(capacity_mw))
    elif key in {"eq", "=", "==", "full", "full_capacity"}:
        mdl.add_constraint(total_expr == float(capacity_mw))
    else:
        raise ValueError(f"Unknown capacity_constraint: {capacity_constraint!r}")

def solve_fixed_price_market_curve_lp(
    wind: np.ndarray,
    da: np.ndarray,
    rt: np.ndarray,
    pen: np.ndarray,
    bid_prices: np.ndarray,
    capacity_mw: float,
    tolerance_mw: float,
    time_limit_sec: float,
    cplex_threads: int,
    solver_log: bool = False,
    min_segment_mw: float = 0.0,
    capacity_constraint: str = "eq",
    quantity_variability_lambda: float = 0.0,
    front_loading_lambda: float = 0.0,
    risk_objective: str = "mean",
    cvar_alpha: float = 0.05,
    cvar_weight: float = 0.0,
) -> dict[str, object]:
    from docplex.mp.model import Model

    wind = np.asarray(wind, dtype=np.float64)
    da = np.asarray(da, dtype=np.float64)
    rt = np.asarray(rt, dtype=np.float64)
    pen = np.asarray(pen, dtype=np.float64)
    prices = np.asarray(bid_prices, dtype=np.float64)
    n = min(len(wind), len(da), len(rt), len(pen))
    wind, da, rt, pen = wind[:n], da[:n], rt[:n], pen[:n]
    accept = BASE.step_acceptance_matrix(da, prices)
    cap = float(capacity_mw)
    min_mw = max(0.0, float(min_segment_mw))
    if min_mw * len(prices) > cap + 1e-9:
        raise ValueError("min_segment_mw is too large for the requested number of fixed segments.")

    mdl = Model(name=f"fixed_price_market_curve_lp_{len(prices)}seg")
    mdl.context.solver.log_output = bool(solver_log)
    if time_limit_sec > 0:
        mdl.parameters.timelimit = float(time_limit_sec)
    if cplex_threads > 0:
        mdl.parameters.threads = int(cplex_threads)

    segments = [
        mdl.continuous_var(lb=min_mw, ub=cap, name=f"d_{k}")
        for k in range(len(prices))
    ]
    _add_capacity_constraint(mdl, mdl.sum(segments), cap, capacity_constraint)

    profit_exprs = []
    award_exprs = []
    loss_exprs = []
    tol = float(tolerance_mw)
    for i in range(n):
        award = mdl.sum(float(accept[i, k]) * segments[k] for k in range(len(prices)))
        award_exprs.append(award)
        over = mdl.continuous_var(lb=0.0, name=f"over_{i}")
        under = mdl.continuous_var(lb=0.0, name=f"under_{i}")
        mdl.add_constraint(over >= float(wind[i]) - award - tol)
        mdl.add_constraint(under >= award - float(wind[i]) - tol)
        rt_settlement = (float(wind[i]) - award) * float(rt[i])
        penalty_loss = float(pen[i]) * (over + under)
        rt_loss = mdl.continuous_var(lb=0.0, name=f"rt_loss_{i}")
        mdl.add_constraint(rt_loss >= -rt_settlement)
        loss_exprs.append(rt_loss + penalty_loss)
        profit_exprs.append(
            award * float(da[i])
            + rt_settlement
            - penalty_loss
        )

    mean_profit_expr = mdl.sum(profit_exprs) / float(n)
    risk_key = str(risk_objective).strip().lower()
    alpha = float(np.clip(float(cvar_alpha), 1.0 / max(1, n), 1.0))
    rho = float(np.clip(float(cvar_weight), 0.0, 1.0))
    cvar_expr = None
    eta = None
    if risk_key in {"mean_cvar", "mean-cvar", "blended_cvar", "cvar_mean", "cvar"}:
        eta = mdl.continuous_var(lb=-mdl.infinity, name="scenario_cvar_eta")
        shortfalls = [mdl.continuous_var(lb=0.0, name=f"scenario_cvar_shortfall_{i}") for i in range(n)]
        for i, profit in enumerate(profit_exprs):
            mdl.add_constraint(shortfalls[i] >= eta - profit)
        cvar_expr = eta - mdl.sum(shortfalls) / (alpha * float(n))

    loss_cvar_expr = None
    loss_eta = None
    if risk_key in {"mean_loss_cvar", "mean-loss-cvar", "loss_cvar_mean", "loss_cvar"}:
        loss_eta = mdl.continuous_var(lb=0.0, name="loss_cvar_eta")
        loss_excess = [mdl.continuous_var(lb=0.0, name=f"loss_cvar_excess_{i}") for i in range(n)]
        for i, loss in enumerate(loss_exprs):
            mdl.add_constraint(loss_excess[i] >= loss - loss_eta)
        loss_cvar_expr = loss_eta + mdl.sum(loss_excess) / (alpha * float(n))

    if risk_key in {"mean", "expected", "expected_value", "ev"}:
        base_objective_expr = mean_profit_expr
        objective_label = "mean"
    elif risk_key in {"mean_cvar", "mean-cvar", "blended_cvar", "cvar_mean"}:
        if cvar_expr is None:
            raise RuntimeError("CVaR expression was not created.")
        base_objective_expr = (1.0 - rho) * mean_profit_expr + rho * cvar_expr
        objective_label = "mean_cvar"
    elif risk_key == "cvar":
        if cvar_expr is None:
            raise RuntimeError("CVaR expression was not created.")
        base_objective_expr = cvar_expr
        objective_label = "cvar"
        rho = 1.0
    elif risk_key in {"mean_loss_cvar", "mean-loss-cvar", "loss_cvar_mean"}:
        if loss_cvar_expr is None:
            raise RuntimeError("Loss-CVaR expression was not created.")
        base_objective_expr = mean_profit_expr - rho * loss_cvar_expr
        objective_label = "mean_loss_cvar"
    elif risk_key == "loss_cvar":
        if loss_cvar_expr is None:
            raise RuntimeError("Loss-CVaR expression was not created.")
        base_objective_expr = -loss_cvar_expr
        objective_label = "loss_cvar"
        rho = 1.0
    else:
        raise ValueError(f"Unknown risk_objective: {risk_objective!r}")

    lambda_norm = max(0.0, float(quantity_variability_lambda))
    j_scale = cap * max(1.0, float(np.mean(np.abs(da))))
    lambda_dollar = lambda_norm * j_scale
    front_lambda_norm = max(0.0, float(front_loading_lambda))
    front_lambda_dollar = front_lambda_norm * j_scale
    objective_expr = base_objective_expr
    if lambda_dollar > 0.0 and award_exprs:
        sum_awards = mdl.sum(award_exprs)
        mean_award_expr = sum_awards / float(n)
        variance_expr = mdl.sum((award - mean_award_expr) * (award - mean_award_expr) for award in award_exprs) / float(n)
        normalized_variance_expr = variance_expr / max(cap * cap, 1e-9)
        objective_expr = objective_expr - lambda_dollar * normalized_variance_expr

    front_loading_expr = 0.0
    if front_lambda_dollar > 0.0 and len(segments) > 1:
        front_vars = []
        for k in range(len(segments) - 1):
            h = mdl.continuous_var(lb=0.0, name=f"frontload_{k}")
            mdl.add_constraint(h >= segments[k] - segments[k + 1])
            front_vars.append(h)
        front_loading_expr = mdl.sum(front_vars) / max(cap, 1e-9)
        objective_expr = objective_expr - front_lambda_dollar * front_loading_expr

    mdl.maximize(objective_expr)
    sol = mdl.solve(clean_before_solve=True)
    details = mdl.solve_details
    if sol is None:
        raise RuntimeError(f"Fixed-price market curve LP failed: {getattr(details, 'status', 'no solution')}")

    segment_q = np.asarray([sol.get_value(v) for v in segments], dtype=np.float64)
    out = BASE._metrics(
        segment_q,
        prices,
        wind,
        da,
        rt,
        pen,
        tolerance_mw=tol,
        objective_value=float(sol.objective_value),
        solver="cplex_fixed_quantile_price_market_curve_lp",
        status=str(getattr(details, "status", "unknown")),
    )
    out["min_segment_mw"] = float(min_mw)
    out["capacity_constraint"] = str(capacity_constraint)
    out["total_offered_capacity_mw"] = float(np.sum(segment_q))
    out["segment_slots"] = int(len(prices))
    out["active_segments"] = int(np.sum(segment_q > 1e-7))
    awards = BASE.award_from_step_offer(segment_q, prices, da)
    profits = BASE._profit_samples(awards, wind, da, rt, pen, tol)
    tail_count = int(max(1, np.ceil(alpha * len(profits)))) if len(profits) else 0
    scenario_cvar_profit = float(np.mean(np.sort(profits)[:tail_count])) if tail_count else float("nan")
    rt_settlements = (wind - awards) * rt
    over_samples = np.maximum(wind - awards - tol, 0.0)
    under_samples = np.maximum(awards - wind - tol, 0.0)
    penalty_loss_samples = pen * (over_samples + under_samples)
    loss_samples = np.maximum(-rt_settlements, 0.0) + penalty_loss_samples
    loss_tail_count = int(max(1, np.ceil(alpha * len(loss_samples)))) if len(loss_samples) else 0
    scenario_loss_cvar = float(np.mean(np.sort(loss_samples)[-loss_tail_count:])) if loss_tail_count else float("nan")
    normalized_variance = float(np.mean(((awards - np.mean(awards)) / max(cap, 1e-9)) ** 2)) if len(awards) else 0.0
    front_violations = np.maximum(segment_q[:-1] - segment_q[1:], 0.0) if len(segment_q) > 1 else np.asarray([], dtype=np.float64)
    front_loading_penalty = float(np.sum(front_violations) / max(cap, 1e-9)) if len(front_violations) else 0.0
    out["risk_objective"] = str(objective_label)
    out["cvar_alpha"] = float(alpha)
    out["cvar_weight"] = float(rho)
    out["scenario_cvar_profit"] = float(scenario_cvar_profit)
    out["scenario_cvar_eta"] = float(sol.get_value(eta)) if eta is not None else float("nan")
    out["scenario_loss_cvar"] = float(scenario_loss_cvar)
    out["scenario_loss_cvar_eta"] = float(sol.get_value(loss_eta)) if loss_eta is not None else float("nan")
    out["mean_tail_loss"] = float(np.mean(loss_samples)) if len(loss_samples) else float("nan")
    out["mean_rt_loss"] = float(np.mean(np.maximum(-rt_settlements, 0.0))) if len(rt_settlements) else float("nan")
    out["mean_penalty_loss"] = float(np.mean(penalty_loss_samples)) if len(penalty_loss_samples) else float("nan")
    out["quantity_variability_lambda"] = float(lambda_norm)
    out["quantity_variability_lambda_dollar"] = float(lambda_dollar)
    out["quantity_variability_penalty"] = float(normalized_variance)
    out["regularization_penalty_amount"] = float(lambda_dollar * normalized_variance)
    out["front_loading_lambda"] = float(front_lambda_norm)
    out["front_loading_lambda_dollar"] = float(front_lambda_dollar)
    out["front_loading_penalty"] = float(front_loading_penalty)
    out["front_loading_penalty_mw"] = float(np.sum(front_violations)) if len(front_violations) else 0.0
    out["front_loading_penalty_amount"] = float(front_lambda_dollar * front_loading_penalty)
    out["front_loading_max_violation_mw"] = float(np.max(front_violations)) if len(front_violations) else 0.0
    out["objective_unregularized"] = float(out["mean_profit"])
    return out

def solve_relaxed_scenario_price_market_curve_lp(
    wind: np.ndarray,
    da: np.ndarray,
    rt: np.ndarray,
    pen: np.ndarray,
    capacity_mw: float,
    tolerance_mw: float,
    time_limit_sec: float,
    cplex_threads: int,
    solver_log: bool = False,
    capacity_constraint: str = "le",
) -> dict[str, object]:
    """Relax the submitted point-count limit.

    This is the S-model style benchmark: every distinct DA scenario price can
    have its own cleared quantity.  It is therefore not a market-submittable
    limited step offer; it is an upper-bound benchmark for the value of a rich
    scenario-price bidding curve.
    """
    from docplex.mp.model import Model

    wind = np.asarray(wind, dtype=np.float64)
    da = np.asarray(da, dtype=np.float64)
    rt = np.asarray(rt, dtype=np.float64)
    pen = np.asarray(pen, dtype=np.float64)
    n = min(len(wind), len(da), len(rt), len(pen))
    wind, da, rt, pen = wind[:n], da[:n], rt[:n], pen[:n]
    if n == 0:
        raise ValueError("At least one scenario is required.")

    prices, inverse = np.unique(np.asarray(da, dtype=np.float64), return_inverse=True)
    order = np.argsort(prices)
    prices = prices[order]
    remap = np.empty_like(order)
    remap[order] = np.arange(len(order), dtype=np.int64)
    inverse = remap[inverse]

    cap = float(capacity_mw)
    tol = float(tolerance_mw)
    mdl = Model(name=f"relaxed_scenario_price_market_curve_lp_{len(prices)}prices")
    mdl.context.solver.log_output = bool(solver_log)
    if time_limit_sec > 0:
        mdl.parameters.timelimit = float(time_limit_sec)
    if cplex_threads > 0:
        mdl.parameters.threads = int(cplex_threads)

    awards_by_price = [
        mdl.continuous_var(lb=0.0, ub=cap, name=f"q_at_p_{k}")
        for k in range(len(prices))
    ]
    for k in range(1, len(awards_by_price)):
        mdl.add_constraint(awards_by_price[k - 1] <= awards_by_price[k], ctname=f"mono_{k}")

    _add_capacity_constraint(mdl, awards_by_price[-1], cap, capacity_constraint)

    profit_exprs = []
    for i in range(n):
        award = awards_by_price[int(inverse[i])]
        over = mdl.continuous_var(lb=0.0, name=f"over_{i}")
        under = mdl.continuous_var(lb=0.0, name=f"under_{i}")
        mdl.add_constraint(over >= float(wind[i]) - award - tol)
        mdl.add_constraint(under >= award - float(wind[i]) - tol)
        profit_exprs.append(
            award * float(da[i])
            + (float(wind[i]) - award) * float(rt[i])
            - float(pen[i]) * (over + under)
        )

    mdl.maximize(mdl.sum(profit_exprs) / float(n))
    sol = mdl.solve(clean_before_solve=True)
    details = mdl.solve_details
    if sol is None:
        raise RuntimeError(f"Relaxed scenario-price LP failed: {getattr(details, 'status', 'no solution')}")

    cumulative_q = np.asarray([sol.get_value(v) for v in awards_by_price], dtype=np.float64)
    block_q = np.diff(np.concatenate([[0.0], cumulative_q]))
    out = BASE._metrics(
        block_q,
        prices,
        wind,
        da,
        rt,
        pen,
        tolerance_mw=tol,
        objective_value=float(sol.objective_value),
        solver="cplex_relaxed_scenario_price_curve_lp",
        status=str(getattr(details, "status", "unknown")),
    )
    out["capacity_constraint"] = str(capacity_constraint)
    out["total_offered_capacity_mw"] = float(cumulative_q[-1]) if len(cumulative_q) else 0.0
    out["segment_slots"] = int(len(prices))
    out["active_segments"] = int(np.sum(block_q > 1e-7))
    out["relaxed_upper_bound"] = True
    out["suppress_full_curve_output"] = True
    return out


def project_dense_step_to_market_curve_qp(
    dense_block_quantities: np.ndarray,
    dense_prices: np.ndarray,
    target_prices: np.ndarray,
    projection_da: np.ndarray,
    capacity_mw: float,
    time_limit_sec: float,
    cplex_threads: int,
    solver_log: bool = False,
    min_segment_mw: float = 0.0,
    capacity_constraint: str = "eq",
) -> np.ndarray:
    from docplex.mp.model import Model

    target_prices = np.asarray(target_prices, dtype=np.float64)
    target_accept = BASE.step_acceptance_matrix(projection_da, target_prices)
    dense_awards = BASE.award_from_step_offer(dense_block_quantities, dense_prices, projection_da)
    cap = float(capacity_mw)
    min_mw = max(0.0, float(min_segment_mw))
    if min_mw * len(target_prices) > cap + 1e-9:
        raise ValueError("min_segment_mw is too large for the requested projected segments.")

    mdl = Model(name="project_dense_to_market_curve_qp")
    mdl.context.solver.log_output = bool(solver_log)
    if time_limit_sec > 0:
        mdl.parameters.timelimit = float(time_limit_sec)
    if cplex_threads > 0:
        mdl.parameters.threads = int(cplex_threads)

    segments = [
        mdl.continuous_var(lb=min_mw, ub=cap, name=f"proj_d_{k}")
        for k in range(len(target_prices))
    ]
    _add_capacity_constraint(mdl, mdl.sum(segments), cap, capacity_constraint)

    sq_errors = []
    for i in range(len(projection_da)):
        award = mdl.sum(float(target_accept[i, k]) * segments[k] for k in range(len(target_prices)))
        err = award - float(dense_awards[i])
        sq_errors.append(err * err)
    mdl.minimize(mdl.sum(sq_errors) / max(1, len(sq_errors)))
    sol = mdl.solve(clean_before_solve=True)
    if sol is None:
        raise RuntimeError(f"Dense-to-market-curve projection QP failed: {mdl.solve_details.status}")
    return np.asarray([sol.get_value(v) for v in segments], dtype=np.float64)


def project_award_curve_to_market_curve_qp(
    curve_prices: np.ndarray,
    curve_awards: np.ndarray,
    target_prices: np.ndarray,
    capacity_mw: float,
    time_limit_sec: float,
    cplex_threads: int,
    solver_log: bool = False,
    min_segment_mw: float = 0.0,
    capacity_constraint: str = "eq",
) -> np.ndarray:
    from docplex.mp.model import Model

    price_axis = np.asarray(curve_prices, dtype=np.float64)
    awards = np.asarray(curve_awards, dtype=np.float64)
    target_prices = np.asarray(target_prices, dtype=np.float64)
    if len(price_axis) != len(awards):
        raise ValueError("curve_prices and curve_awards must have the same length.")
    if len(price_axis) == 0:
        raise ValueError("curve price grid cannot be empty.")

    target_accept = BASE.step_acceptance_matrix(price_axis, target_prices)
    cap = float(capacity_mw)
    min_mw = max(0.0, float(min_segment_mw))
    if min_mw * len(target_prices) > cap + 1e-9:
        raise ValueError("min_segment_mw is too large for the requested projected segments.")

    mdl = Model(name="project_continuous_curve_to_market_curve_qp")
    mdl.context.solver.log_output = bool(solver_log)
    if time_limit_sec > 0:
        mdl.parameters.timelimit = float(time_limit_sec)
    if cplex_threads > 0:
        mdl.parameters.threads = int(cplex_threads)

    segments = [
        mdl.continuous_var(lb=min_mw, ub=cap, name=f"shape_proj_d_{k}")
        for k in range(len(target_prices))
    ]
    _add_capacity_constraint(mdl, mdl.sum(segments), cap, capacity_constraint)

    sq_errors = []
    for i in range(len(price_axis)):
        award = mdl.sum(float(target_accept[i, k]) * segments[k] for k in range(len(target_prices)))
        err = award - float(awards[i])
        sq_errors.append(err * err)
    mdl.minimize(mdl.sum(sq_errors) / max(1, len(sq_errors)))
    sol = mdl.solve(clean_before_solve=True)
    if sol is None:
        raise RuntimeError(f"Continuous-curve projection QP failed: {mdl.solve_details.status}")
    return np.asarray([sol.get_value(v) for v in segments], dtype=np.float64)


def _continuous_expected_settlement_awards(
    wind: np.ndarray,
    rt: np.ndarray,
    pen: np.ndarray,
    price_axis: np.ndarray,
    capacity_mw: float,
    tolerance_mw: float,
) -> np.ndarray:
    """Pointwise continuous offer response q*(p) for exogenous DA price p.

    This creates a continuous-curve benchmark that does not use the empirical
    DA clearing frequency when deciding where the six market segments should
    be placed.  For each price on a uniform axis, it solves the one-dimensional
    expected-settlement problem implied by the wind, RT, and penalty scenarios.
    """
    wind = np.asarray(wind, dtype=np.float64)
    rt = np.asarray(rt, dtype=np.float64)
    pen = np.asarray(pen, dtype=np.float64)
    prices = np.asarray(price_axis, dtype=np.float64)
    n = min(len(wind), len(rt), len(pen))
    if n == 0:
        raise ValueError("At least one scenario is required.")
    wind, rt, pen = wind[:n], rt[:n], np.maximum(pen[:n], 0.0)

    cap = float(capacity_mw)
    tol = float(tolerance_mw)
    lower_band = wind - tol
    upper_band = wind + tol

    event_values = np.concatenate([lower_band, upper_band])
    event_weights = -np.concatenate([pen, pen]) / float(n)
    event_mask = np.isfinite(event_values) & (event_values >= 0.0) & (event_values <= cap)
    event_values = event_values[event_mask]
    event_weights = event_weights[event_mask]
    order = np.argsort(event_values)
    event_values = event_values[order]
    event_weights = event_weights[order]

    mean_rt = float(np.mean(rt))
    start_plus = np.mean(pen * (0.0 < lower_band))
    start_minus = np.mean(pen * (0.0 > upper_band))

    awards = np.zeros(len(prices), dtype=np.float64)
    for i, price in enumerate(prices):
        derivative = float(price) - mean_rt + float(start_plus) - float(start_minus)
        if derivative <= 0.0:
            awards[i] = 0.0
            continue
        award = cap
        for value, weight in zip(event_values, event_weights):
            derivative += float(weight)
            if derivative <= 0.0:
                award = float(np.clip(value, 0.0, cap))
                break
        awards[i] = award
    return np.maximum.accumulate(np.clip(awards, 0.0, cap))


def _select_shape_preserving_prices(
    curve_prices: np.ndarray,
    curve_awards: np.ndarray,
    max_segments: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    prices = np.asarray(curve_prices, dtype=np.float64)
    awards = np.asarray(curve_awards, dtype=np.float64)
    if len(prices) != len(awards):
        raise ValueError("curve_prices and curve_awards must have the same length.")
    if len(prices) == 0:
        raise ValueError("curve price grid cannot be empty.")

    max_count = int(max_segments)
    if max_count < 1:
        raise ValueError("max_segments must be positive.")
    if len(prices) <= max_count:
        indices = np.arange(len(prices), dtype=np.int64)
        if len(indices) < max_count:
            pad = int(indices[-1]) if len(indices) else 0
            indices = np.concatenate([indices, np.full(max_count - len(indices), pad, dtype=np.int64)])
        return np.sort(prices[indices]), indices, np.ones(len(prices), dtype=np.float64)

    increments = np.abs(np.diff(awards, prepend=awards[0]))
    increments[0] = max(increments[0], abs(awards[0]))
    if float(np.sum(increments)) <= 1e-9:
        indices = np.linspace(0, len(prices) - 1, max_count).round().astype(np.int64)
        return np.sort(prices[indices]), indices, increments

    cumulative = np.cumsum(increments)
    levels = np.linspace(cumulative[0], cumulative[-1], max_count)
    indices = np.searchsorted(cumulative, levels, side="left")
    indices = np.clip(indices, 0, len(prices) - 1).astype(np.int64)
    if len(np.unique(indices)) < max_count:
        ranked = np.argsort(increments)[::-1]
        chosen = list(dict.fromkeys(int(x) for x in indices))
        for idx in ranked:
            if int(idx) not in chosen:
                chosen.append(int(idx))
            if len(chosen) >= max_count:
                break
        indices = np.asarray(sorted(chosen[:max_count]), dtype=np.int64)
    return np.sort(prices[indices]), indices, increments


def _select_frequency_weighted_projection_prices(
    *,
    dense_prices: np.ndarray,
    dense_quantities: np.ndarray,
    da_values: np.ndarray,
    max_segments: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    prices = np.asarray(dense_prices, dtype=np.float64)
    quantities = np.asarray(dense_quantities, dtype=np.float64)
    da = np.asarray(da_values, dtype=np.float64)
    if len(prices) != len(quantities):
        raise ValueError("dense price and quantity arrays must have the same length.")
    if len(prices) == 0:
        raise ValueError("dense price grid cannot be empty.")

    # A segment price matters only when it changes awards in scenarios that
    # actually clear around that threshold.  The empirical acceptance drop
    # between adjacent dense prices is the clearing-frequency proxy.
    accept_counts = np.asarray([np.sum(da >= p) for p in prices], dtype=np.float64)
    if len(prices) == 1:
        importance = np.asarray([float(abs(quantities[0]))], dtype=np.float64)
    else:
        clearing_mass = np.maximum(accept_counts[:-1] - accept_counts[1:], 0.0)
        importance = np.zeros(len(prices), dtype=np.float64)
        importance[:-1] += np.abs(quantities[:-1]) * clearing_mass
        importance[1:] += np.abs(quantities[:-1]) * clearing_mass
        importance += 1e-9 * np.maximum(np.abs(quantities), 1.0)

    selected = np.flatnonzero(quantities > 1e-7)
    if len(selected) == 0:
        selected = np.asarray([int(np.argmax(importance))], dtype=np.int64)
    if len(selected) > int(max_segments):
        selected = selected[np.argsort(importance[selected])[-int(max_segments):]]
    selected = np.sort(selected)
    if len(selected) < int(max_segments):
        pool = np.argsort(importance)[::-1]
        additions = []
        selected_set = set(int(x) for x in selected)
        for idx in pool:
            if int(idx) not in selected_set:
                additions.append(int(idx))
                selected_set.add(int(idx))
            if len(selected) + len(additions) >= int(max_segments):
                break
        if additions:
            selected = np.sort(np.concatenate([selected, np.asarray(additions, dtype=np.int64)]))
    if len(selected) < int(max_segments):
        pad = int(selected[-1]) if len(selected) else int(np.argmax(importance))
        selected = np.concatenate([
            selected,
            np.full(int(max_segments) - len(selected), pad, dtype=np.int64),
        ])
    elif len(selected) > int(max_segments):
        selected = np.sort(selected[np.argsort(importance[selected])[-int(max_segments):]])
    return np.sort(prices[selected]), selected, importance


def solve_continuous_projection_market_curve(
    wind: np.ndarray,
    da: np.ndarray,
    rt: np.ndarray,
    pen: np.ndarray,
    target_prices: np.ndarray,
    dense_prices: np.ndarray,
    capacity_mw: float,
    tolerance_mw: float,
    time_limit_sec: float,
    cplex_threads: int,
    solver_log: bool = False,
    min_segment_mw: float = 0.0,
    price_selection: str = "fixed_quantile",
    capacity_constraint: str = "eq",
) -> dict[str, object]:
    selection_key = str(price_selection).strip().lower()
    if selection_key in {"shape", "shape_preserving", "continuous_shape", "uniform_shape"}:
        curve_prices = np.asarray(dense_prices, dtype=np.float64)
        curve_awards = _continuous_expected_settlement_awards(
            wind=np.asarray(wind, dtype=np.float64),
            rt=np.asarray(rt, dtype=np.float64),
            pen=np.asarray(pen, dtype=np.float64),
            price_axis=curve_prices,
            capacity_mw=capacity_mw,
            tolerance_mw=tolerance_mw,
        )
        if selection_key == "uniform_shape":
            target_prices = np.asarray(target_prices, dtype=np.float64)
            selected_dense_indices = np.searchsorted(curve_prices, target_prices, side="left")
            selected_dense_indices = np.clip(selected_dense_indices, 0, len(curve_prices) - 1).astype(np.int64)
            shape_importance = np.abs(np.diff(curve_awards, prepend=curve_awards[0]))
        else:
            target_prices, selected_dense_indices, shape_importance = _select_shape_preserving_prices(
                curve_prices=curve_prices,
                curve_awards=curve_awards,
                max_segments=len(target_prices),
            )
        projected_segments = project_award_curve_to_market_curve_qp(
            curve_prices=curve_prices,
            curve_awards=curve_awards,
            target_prices=np.asarray(target_prices, dtype=np.float64),
            capacity_mw=capacity_mw,
            time_limit_sec=time_limit_sec,
            cplex_threads=cplex_threads,
            solver_log=solver_log,
            min_segment_mw=min_segment_mw,
            capacity_constraint=capacity_constraint,
        )
        out = BASE._metrics(
            projected_segments,
            target_prices,
            np.asarray(wind, dtype=np.float64),
            np.asarray(da, dtype=np.float64),
            np.asarray(rt, dtype=np.float64),
            np.asarray(pen, dtype=np.float64),
            tolerance_mw=float(tolerance_mw),
            objective_value=0.0,
            solver="continuous_curve_shape_projection_qp",
            status="continuous:pointwise; shape_projection:optimal",
        )
        out["objective"] = float(out["mean_profit"])
        out["dense_objective"] = np.nan
        out["dense_bid_prices"] = curve_prices
        out["dense_block_quantities"] = curve_awards
        out["projection_price_selection"] = selection_key
        out["selected_dense_indices"] = np.asarray(selected_dense_indices, dtype=np.int64)
        out["dense_price_importance"] = np.asarray(shape_importance, dtype=np.float64)
        out["min_segment_mw"] = float(max(0.0, min_segment_mw))
        out["capacity_constraint"] = str(capacity_constraint)
        out["total_offered_capacity_mw"] = float(np.sum(projected_segments))
        out["segment_slots"] = int(len(target_prices))
        out["active_segments"] = int(np.sum(np.asarray(projected_segments, dtype=np.float64) > 1e-7))
        return out

    dense = BASE.solve_fixed_price_step_lp(
        wind,
        da,
        rt,
        pen,
        bid_prices=dense_prices,
        capacity_mw=capacity_mw,
        tolerance_mw=tolerance_mw,
        time_limit_sec=time_limit_sec,
        cplex_threads=cplex_threads,
        solver_log=solver_log,
        capacity_constraint=capacity_constraint,
    )
    selected_dense_indices = np.asarray([], dtype=np.int64)
    price_importance = np.asarray([], dtype=np.float64)
    if selection_key in {"frequency", "frequency_weighted", "clearing_frequency", "clearing_weighted"}:
        target_prices, selected_dense_indices, price_importance = _select_frequency_weighted_projection_prices(
            dense_prices=np.asarray(dense["bid_prices"], dtype=np.float64),
            dense_quantities=np.asarray(dense["block_quantities"], dtype=np.float64),
            da_values=np.asarray(da, dtype=np.float64),
            max_segments=len(target_prices),
        )
    projected_segments = project_dense_step_to_market_curve_qp(
        dense_block_quantities=np.asarray(dense["block_quantities"], dtype=np.float64),
        dense_prices=np.asarray(dense["bid_prices"], dtype=np.float64),
        target_prices=np.asarray(target_prices, dtype=np.float64),
        projection_da=np.asarray(da, dtype=np.float64),
        capacity_mw=capacity_mw,
        time_limit_sec=time_limit_sec,
        cplex_threads=cplex_threads,
        solver_log=solver_log,
        min_segment_mw=min_segment_mw,
        capacity_constraint=capacity_constraint,
    )
    out = BASE._metrics(
        projected_segments,
        target_prices,
        np.asarray(wind, dtype=np.float64),
        np.asarray(da, dtype=np.float64),
        np.asarray(rt, dtype=np.float64),
        np.asarray(pen, dtype=np.float64),
        tolerance_mw=tolerance_mw,
        objective_value=0.0,
        solver="cplex_dense_curve_then_market_curve_projection_qp",
        status=f"dense:{dense['status']}; projection:optimal",
    )
    out["objective"] = float(out["mean_profit"])
    out["dense_objective"] = float(dense["objective"])
    out["dense_bid_prices"] = np.asarray(dense["bid_prices"], dtype=np.float64)
    out["dense_block_quantities"] = np.asarray(dense["block_quantities"], dtype=np.float64)
    out["projection_price_selection"] = selection_key
    out["selected_dense_indices"] = np.asarray(selected_dense_indices, dtype=np.int64)
    out["dense_price_importance"] = np.asarray(price_importance, dtype=np.float64)
    out["min_segment_mw"] = float(max(0.0, min_segment_mw))
    out["capacity_constraint"] = str(capacity_constraint)
    out["total_offered_capacity_mw"] = float(np.sum(projected_segments))
    out["segment_slots"] = int(len(target_prices))
    out["active_segments"] = int(np.sum(np.asarray(projected_segments, dtype=np.float64) > 1e-7))
    return out

def solve_market_curve_method(
    *,
    method: str,
    wind: np.ndarray,
    da: np.ndarray,
    rt: np.ndarray,
    pen: np.ndarray,
    max_points: int,
    dense_segments: int,
    price_tick: float,
    min_segment_mw: float,
    capacity_mw: float,
    tolerance_mw: float,
    cplex_time_limit_sec: float,
    cplex_threads: int,
    solver_log: bool = False,
    projection_price_selection: str = "fixed_quantile",
    price_support_mode: str = "quantile",
    capacity_constraint: str = "eq",
    quantity_variability_lambda: float = 0.0,
    front_loading_lambda: float = 0.0,
    risk_objective: str = "mean",
    cvar_alpha: float = 0.05,
    cvar_weight: float = 0.0,
    **_unused: object,
) -> dict[str, object]:
    """Dispatch only the three methods used in the final paper experiments."""
    method_key = str(method).strip().lower()
    max_segments = max_segments_from_points(max_points)
    fixed_prices = _make_price_grid(
        da, max_segments, price_tick=price_tick, mode=price_support_mode
    )
    dense_prices = _make_price_grid(
        da, int(dense_segments), price_tick=price_tick, mode=price_support_mode
    )

    start = time.perf_counter()
    if method_key == "relaxed":  # M0
        result = solve_relaxed_scenario_price_market_curve_lp(
            wind,
            da,
            rt,
            pen,
            capacity_mw=capacity_mw,
            tolerance_mw=tolerance_mw,
            time_limit_sec=cplex_time_limit_sec,
            cplex_threads=cplex_threads,
            solver_log=solver_log,
            capacity_constraint=capacity_constraint,
        )
    elif method_key == "fixed":  # M1
        result = solve_fixed_price_market_curve_lp(
            wind,
            da,
            rt,
            pen,
            bid_prices=fixed_prices,
            capacity_mw=capacity_mw,
            tolerance_mw=tolerance_mw,
            time_limit_sec=cplex_time_limit_sec,
            cplex_threads=cplex_threads,
            solver_log=solver_log,
            min_segment_mw=min_segment_mw,
            capacity_constraint=capacity_constraint,
            quantity_variability_lambda=quantity_variability_lambda,
            front_loading_lambda=front_loading_lambda,
            risk_objective=risk_objective,
            cvar_alpha=cvar_alpha,
            cvar_weight=cvar_weight,
        )
    elif method_key == "projection":  # M2
        result = solve_continuous_projection_market_curve(
            wind,
            da,
            rt,
            pen,
            target_prices=fixed_prices,
            dense_prices=dense_prices,
            capacity_mw=capacity_mw,
            tolerance_mw=tolerance_mw,
            time_limit_sec=cplex_time_limit_sec,
            cplex_threads=cplex_threads,
            solver_log=solver_log,
            min_segment_mw=min_segment_mw,
            price_selection=projection_price_selection,
            capacity_constraint=capacity_constraint,
        )
    else:
        raise ValueError(f"Unsupported final-paper method: {method!r}")

    result["elapsed_sec"] = float(time.perf_counter() - start)
    result["max_points"] = int(max_points)
    result["max_segments"] = int(max_segments)
    result["segment_slots"] = int(max_segments)
    result["active_segments"] = int(
        np.sum(np.asarray(result["block_quantities"], dtype=np.float64) > 1e-7)
    )
    result["price_tick"] = float(price_tick)
    result["price_support_mode"] = str(
        result.get("price_support_mode", price_support_mode)
    )
    result["capacity_constraint"] = str(capacity_constraint)
    result["quantity_variability_lambda"] = float(
        max(0.0, float(quantity_variability_lambda))
    )
    result["front_loading_lambda"] = float(max(0.0, float(front_loading_lambda)))
    result["total_offered_capacity_mw"] = float(
        np.sum(np.asarray(result["block_quantities"], dtype=np.float64))
    )
    return result
