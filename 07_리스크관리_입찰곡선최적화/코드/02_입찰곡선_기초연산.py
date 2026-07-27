#!/usr/bin/env python
"""Offer-curve acceptance and scenario-profit primitives used by M0, M1, and M2."""

from __future__ import annotations

import numpy as np


def step_acceptance_matrix(da: np.ndarray, bid_prices: np.ndarray) -> np.ndarray:
    da = np.asarray(da, dtype=np.float64)
    prices = np.asarray(bid_prices, dtype=np.float64)
    return (da[:, None] >= prices[None, :]).astype(np.float64)


def award_from_step_offer(
    block_quantities: np.ndarray,
    bid_prices: np.ndarray,
    da_values: np.ndarray,
) -> np.ndarray:
    acceptance = step_acceptance_matrix(np.asarray(da_values, dtype=np.float64), bid_prices)
    return acceptance @ np.asarray(block_quantities, dtype=np.float64)


def profit_samples(
    awards: np.ndarray,
    wind: np.ndarray,
    da: np.ndarray,
    rt: np.ndarray,
    penalty_price: np.ndarray,
    tolerance_mw: float,
) -> np.ndarray:
    over = np.maximum(wind - awards - float(tolerance_mw), 0.0)
    under = np.maximum(awards - wind - float(tolerance_mw), 0.0)
    return awards * da + (wind - awards) * rt - penalty_price * (over + under)


# Internal alias retained so the preserved optimization equations call the
# same symbol as the result-verified implementation.
_profit_samples = profit_samples


def _add_capacity_constraint(model, total_quantity, capacity_mw: float, mode: str) -> None:
    if mode == "eq":
        model.add_constraint(total_quantity == float(capacity_mw))
    elif mode == "le":
        model.add_constraint(total_quantity <= float(capacity_mw))
    else:
        raise ValueError(f"Unknown capacity_constraint: {mode!r}")


def _metrics(
    block_quantities: np.ndarray,
    bid_prices: np.ndarray,
    wind: np.ndarray,
    da: np.ndarray,
    rt: np.ndarray,
    pen: np.ndarray,
    tolerance_mw: float,
    objective_value: float,
    solver: str,
    status: str,
) -> dict[str, object]:
    awards = award_from_step_offer(block_quantities, bid_prices, da)
    profits = _profit_samples(awards, wind, da, rt, pen, tolerance_mw)
    under = np.maximum(awards - wind - float(tolerance_mw), 0.0)
    return {
        "solver": solver,
        "status": status,
        "objective": float(objective_value),
        "n_scenarios": int(len(wind)),
        "bid_prices": np.asarray(bid_prices, dtype=np.float64),
        "block_quantities": np.asarray(block_quantities, dtype=np.float64),
        "cumulative_quantities": np.cumsum(np.asarray(block_quantities, dtype=np.float64)),
        "mean_award_mw": float(np.mean(awards)),
        "mean_profit": float(np.mean(profits)),
        "mean_under_mw": float(np.mean(under)),
        "mean_under_cost": float(np.mean(pen * under)),
        "profit_samples": profits,
    }


def solve_fixed_price_step_lp(
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
    capacity_constraint: str = "eq",
) -> dict[str, object]:
    """Solve the fixed-grid LP used directly by M1 and internally by M2."""
    from docplex.mp.model import Model

    wind = np.asarray(wind, dtype=np.float64)
    da = np.asarray(da, dtype=np.float64)
    rt = np.asarray(rt, dtype=np.float64)
    pen = np.asarray(pen, dtype=np.float64)
    prices = np.asarray(bid_prices, dtype=np.float64)
    n = min(len(wind), len(da), len(rt), len(pen))
    wind, da, rt, pen = wind[:n], da[:n], rt[:n], pen[:n]
    accept = step_acceptance_matrix(da, prices)

    model = Model(name=f"fixed_price_step_lp_{len(prices)}")
    model.context.solver.log_output = bool(solver_log)
    if time_limit_sec > 0:
        model.parameters.timelimit = float(time_limit_sec)
    if cplex_threads > 0:
        model.parameters.threads = int(cplex_threads)

    blocks = [
        model.continuous_var(lb=0.0, ub=float(capacity_mw), name=f"d_{k}")
        for k in range(len(prices))
    ]
    _add_capacity_constraint(model, model.sum(blocks), float(capacity_mw), capacity_constraint)

    profit_exprs = []
    tol = float(tolerance_mw)
    for i in range(n):
        award = model.sum(float(accept[i, k]) * blocks[k] for k in range(len(prices)))
        over = model.continuous_var(lb=0.0, name=f"over_{i}")
        under = model.continuous_var(lb=0.0, name=f"under_{i}")
        model.add_constraint(over >= float(wind[i]) - award - tol)
        model.add_constraint(under >= award - float(wind[i]) - tol)
        profit_exprs.append(
            award * float(da[i])
            + (float(wind[i]) - award) * float(rt[i])
            - float(pen[i]) * (over + under)
        )

    model.maximize(model.sum(profit_exprs) / float(n))
    solution = model.solve(clean_before_solve=True)
    details = model.solve_details
    if solution is None:
        raise RuntimeError(f"Fixed-price step LP failed: {getattr(details, 'status', 'no solution')}")

    block_q = np.asarray([solution.get_value(value) for value in blocks], dtype=np.float64)
    result = _metrics(
        block_q,
        prices,
        wind,
        da,
        rt,
        pen,
        tolerance_mw=tol,
        objective_value=float(solution.objective_value),
        solver="cplex_fixed_price_step_lp",
        status=str(getattr(details, "status", "unknown")),
    )
    result["capacity_constraint"] = str(capacity_constraint)
    result["total_offered_capacity_mw"] = float(np.sum(block_q))
    return result
