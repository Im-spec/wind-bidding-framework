#!/usr/bin/env python
"""Run the paper pipeline from preprocessing through the six final optimizations."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
STAGES = ("preprocess", "forecast", "errors", "fit", "scenarios", "optimization")


def run(script: str, *args: str) -> None:
    command = [sys.executable, str(ROOT / script), *map(str, args)]
    print("\n[RUN]", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def run_preprocessing() -> None:
    code = "02_입력데이터가공/코드"
    out = ROOT / "02_입력데이터가공" / "가공데이터"
    integrated = ROOT / "01_입력데이터" / "정확재현_통합원본"
    run(f"{code}/01_풍력발전_가공데이터생성.py", "--out-root", out)
    run(f"{code}/02_하루전가격_가공데이터생성.py", "--out-root", out)
    run(f"{code}/03_실시간가격_가공데이터생성.py", "--out-root", out)
    run(
        f"{code}/04_DA_REG가격_가공데이터생성.py",
        "--source-csv", integrated / "05_DA_REG가격_학습데이터.csv",
        "--out-root", out,
    )
    run(
        f"{code}/05_RT_REG가격_가공데이터생성.py",
        "--source-csv", integrated / "06_RT_REG가격_학습데이터.csv",
        "--out-root", out,
    )


def run_point_forecasts() -> None:
    code = "03_점예측/코드"
    prep = ROOT / "02_입력데이터가공" / "가공데이터"
    internal = ROOT / "03_점예측" / "점예측내부산출물"
    results = ROOT / "03_점예측" / "점예측결과"
    cases = (
        ("08_풍력발전_본체모형학습.py", "11_풍력발전_결과내보내기.py", "01_풍력발전_가공데이터_전체.csv"),
        ("09_DA에너지_본체모형학습.py", "12_DA에너지_결과내보내기.py", "05_DA에너지_가공데이터_전체.csv"),
        ("10_RT에너지_본체모형학습.py", "13_RT에너지_결과내보내기.py", "09_RT에너지_가공데이터_전체.csv"),
        ("14_DA_REG가격_본체모형학습.py", "16_DA_REG가격_결과내보내기.py", "13_DA_REG가격_가공데이터_전체.csv"),
        ("15_RT_REG가격_본체모형학습.py", "17_RT_REG가격_결과내보내기.py", "17_RT_REG가격_가공데이터_전체.csv"),
    )
    for trainer, exporter, dataset in cases:
        run(f"{code}/{trainer}", "--dataset-csv", prep / dataset, "--out-root", internal)
        run(f"{code}/{exporter}", "--pred-root", internal, "--user-out-root", results)


def run_error_series() -> None:
    run(
        "04_오차데이터/코드/01_점예측기반_오차데이터생성.py",
        "--point-root", ROOT / "03_점예측" / "점예측결과",
        "--out-root", ROOT / "04_오차데이터" / "오차데이터결과",
    )


def run_distribution_fit() -> None:
    run(
        "05_오차증분_분포적합/코드/01_오차증분기반_분포적합생성.py",
        "--error-root", ROOT / "04_오차데이터" / "오차데이터결과",
        "--out-root", ROOT / "05_오차증분_분포적합" / "오차증분_분포적합결과",
        "--distributions", "gaussian", "laplace",
    )


def run_scenarios() -> None:
    script = "06_시나리오생성/코드/01_점예측오차증분기반_시나리오생성.py"
    for family in ("gaussian", "laplace"):
        run(
            script,
            "--families", family,
            "--date-start", "2021-01-01",
            "--date-end", "2021-12-31",
            "--n-scenarios", "5000",
            "--seed", "777",
            "--matching-mode", "variable_pc12_morton",
            "--scenario-root-name", f"reproduced/시나리오생성결과_s5000_{family}",
            "--progress-name", f"98_s5000_{family}_progress.json",
            "--meta-name", f"99_s5000_{family}_meta.json",
        )


def run_optimization() -> None:
    run("07_리스크관리_입찰곡선최적화/코드/05_최종6개실험_실행.py")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-stage", choices=STAGES, default=STAGES[0])
    parser.add_argument("--end-stage", choices=STAGES, default=STAGES[-1])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    start = STAGES.index(args.start_stage)
    end = STAGES.index(args.end_stage)
    if start > end:
        raise ValueError("--start-stage must not come after --end-stage")
    actions = {
        "preprocess": run_preprocessing,
        "forecast": run_point_forecasts,
        "errors": run_error_series,
        "fit": run_distribution_fit,
        "scenarios": run_scenarios,
        "optimization": run_optimization,
    }
    for stage in STAGES[start : end + 1]:
        print(f"\n===== {stage} =====", flush=True)
        actions[stage]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())