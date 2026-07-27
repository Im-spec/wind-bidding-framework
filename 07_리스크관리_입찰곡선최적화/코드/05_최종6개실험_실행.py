#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Run the paper's six final 2021 experiments from one entry point."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

FULL_YEAR_RUNNER = Path(__file__).resolve().parent / "04_연간최적화_실행.py"
RESULT_ROOT = PROJECT_ROOT / "07_리스크관리_입찰곡선최적화" / "reproduced"
LOG_ROOT = PROJECT_ROOT / "logs"
STATE_PATH = RESULT_ROOT / "98_s5000_six_cases_progress.json"
REG_POINT_FORECAST_ROOT = PROJECT_ROOT / "03_점예측" / "점예측결과"

FAMILIES = ("laplace", "gaussian")
SCENARIO_ROOT_NAMES = {
    "laplace": "시나리오생성결과_s5000_laplace",
    "gaussian": "시나리오생성결과_s5000_gaussian",
}
METHODS = (
    ("m0_relaxed_continuous", "relaxed", None),
    ("m1_fixed_11slot", "fixed", None),
    ("m2_frequency_projection_11slot", "projection", "frequency_weighted"),
)


def _scenario_root(family: str) -> Path:
    return PROJECT_ROOT / "06_시나리오생성" / "reproduced" / SCENARIO_ROOT_NAMES[family]


def _scenario_complete(family: str) -> bool:
    root = _scenario_root(family)
    wind = list((root / "__풍력시나리오__" / f"joint_case_{family}" / "joint_day_parts").glob("operating_date_et=*/*.parquet"))
    energy = list((root / "__DA_RT에너지결합시나리오__" / f"joint_case_{family}" / "joint_day_parts").glob("operating_date_et=*/*.parquet"))
    return bool(
        len(wind) == 308
        and len(energy) == 308
        and (root / f"99_s5000_{family}_meta.json").is_file()
    )


def _write_state(**updates: object) -> None:
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    state: dict[str, object] = {}
    if STATE_PATH.is_file():
        state = json.loads(STATE_PATH.read_text(encoding="utf-8"))
    state.update(updates)
    state["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    STATE_PATH.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def _run(label: str, args: list[str]) -> None:
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    with (LOG_ROOT / f"{label}.out.log").open("a", encoding="utf-8") as out_fh, (
        LOG_ROOT / f"{label}.err.log"
    ).open("a", encoding="utf-8") as err_fh:
        completed = subprocess.run(
            [sys.executable, *args], cwd=PROJECT_ROOT,
            stdout=out_fh, stderr=err_fh, text=True, check=False,
        )
    if completed.returncode:
        raise RuntimeError(f"{label} failed with exit code {completed.returncode}; see {LOG_ROOT}")


def _ensure_scenarios(family: str) -> None:
    if _scenario_complete(family):
        return
    raise RuntimeError(
        f"{family}: complete S5000 scenarios are missing: "
        f"{_scenario_root(family)}"
    )


def _result_dir(family: str, method_label: str) -> Path:
    prefix = "130" if family == "gaussian" else ("128" if method_label.startswith("m0") else "126")
    return RESULT_ROOT / f"{prefix}_true_s5000_{family}_{method_label}_full_year"


def _run_case(family: str, method_label: str, method: str, projection: str | None) -> None:
    if not REG_POINT_FORECAST_ROOT.is_dir():
        raise FileNotFoundError(f"REG point-forecast outputs are missing: {REG_POINT_FORECAST_ROOT}")
    args = [
        str(FULL_YEAR_RUNNER),
        "--scenario-root-name", str(Path("reproduced") / SCENARIO_ROOT_NAMES[family]),
        "--family", family, "--start", "2021-01-01", "--end", "2021-12-31",
        "--n-scenarios", "5000", "--max-points", "11",
        "--dense-segments", "201", "--price-grid-size", "201",
        "--price-support-mode", "uniform", "--capacity-constraint", "le",
        "--penalty-mode", "point_reg",
        "--reg-point-forecast-file", str(REG_POINT_FORECAST_ROOT),
        "--cplex-time-limit-sec", "300", "--cplex-threads", "4",
        "--save-every", "24", "--method", method,
        "--out-root", str(_result_dir(family, method_label)),
    ]
    if any(_result_dir(family, method_label).glob("0*_*.csv")):
        args.append("--resume-existing")
    if projection:
        args.extend(["--projection-price-selection", projection])
    _run(f"optimize_{family}_{method_label}", args)


def main() -> int:
    cases = [
        (family, method_label, method, projection)
        for family in FAMILIES
        for method_label, method, projection in METHODS
    ]
    _write_state(status="running", completed_cases=0, total_cases=len(cases), current_case=None)
    try:
        for family in FAMILIES:
            _ensure_scenarios(family)
        for index, (family, method_label, method, projection) in enumerate(cases, start=1):
            case_name = f"{family}_{method_label}"
            _write_state(status="running", completed_cases=index - 1, current_case=case_name)
            _run_case(family, method_label, method, projection)
            _write_state(status="running", completed_cases=index, current_case=case_name)
    except Exception as exc:
        _write_state(status="failed", error_type=type(exc).__name__, error_message=str(exc))
        raise
    _write_state(status="completed", completed_cases=len(cases), current_case=None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
