from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path(__file__).resolve().parents[2]
ERROR_ROOT = ROOT / "04_오차데이터" / "오차데이터결과"
OUT_ROOT = ROOT / "05_오차증분_분포적합" / "오차증분_분포적합결과"
BASE_PLOT_DIR = OUT_ROOT / "01_메인_기본분포비교"
TABLE_DIR = OUT_ROOT / "03_참고_기본분포_상세표"
GUIDE_MD = OUT_ROOT / "00_안내_결과폴더.md"

plt.style.use("seaborn-v0_8-whitegrid")

DATASETS = [
    ("wind", "풍력발전", "Wind", ERROR_ROOT / "01_풍력발전_오차시계열_2018_2021.csv"),
    ("da_energy", "DA에너지", "DA Energy", ERROR_ROOT / "02_DA에너지_오차시계열_2018_2021.csv"),
    ("rt_energy", "RT에너지", "RT Energy", ERROR_ROOT / "03_RT에너지_오차시계열_2018_2021.csv"),
    ("da_reg", "DA_REG가격", "DA REG", ERROR_ROOT / "04_DA_REG가격_오차시계열_2018_2021.csv"),
    ("rt_reg", "RT_REG가격", "RT REG", ERROR_ROOT / "05_RT_REG가격_오차시계열_2018_2021.csv"),
]

DIST_ORDER = ["Gaussian", "Laplace"]
DIST_NAME_ALIASES = {
    "gaussian": "Gaussian",
    "normal": "Gaussian",
    "laplace": "Laplace",
    "skewed_t": "SkewedT",
    "skewedt": "SkewedT",
    "skew-t": "SkewedT",
    "jf_skew_t": "SkewedT",
    "jones_faddy_skew_t": "SkewedT",
    "stable": "Stable",
    "levy_stable": "Stable",
    "levy-stable": "Stable",
}
RNG = np.random.default_rng(20260415)
DIST_STYLES = {
    "Gaussian": {"color": "#4C78A8", "linestyle": "-"},
    "Laplace": {"color": "#F58518", "linestyle": "--"},
    "SkewedT": {"color": "#B279A2", "linestyle": "-"},
    "Stable": {"color": "#54A24B", "linestyle": "-."},
}

DISPLAY_Q_LOW = 0.005
DISPLAY_Q_HIGH = 0.995
QQ_Q_LOW = 0.01
QQ_Q_HIGH = 0.99
QQ_POINTS = 220
EVAL_MAX_N = 20000
PROGRESS_JSON = OUT_ROOT / "98_오차증분_분포적합_진행상황.json"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit error-increment distributions.")
    parser.add_argument(
        "--error-root",
        type=Path,
        default=ERROR_ROOT,
        help="Directory containing the five error-series CSV inputs.",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=OUT_ROOT,
        help="Separate output directory. Existing canonical outputs are not overwritten.",
    )
    parser.add_argument(
        "--include-stable",
        action="store_true",
        help="Also fit Levy alpha-stable distribution. This can be very slow.",
    )
    parser.add_argument(
        "--distributions",
        nargs="+",
        default=None,
        help="Explicit distribution list: gaussian laplace skewed_t stable.",
    )
    parser.add_argument(
        "--only-datasets",
        nargs="+",
        default=None,
        help="Optional dataset keys to run: wind da_energy rt_energy da_reg rt_reg.",
    )
    parser.add_argument(
        "--stable-fit-max-n",
        type=int,
        default=0,
        help="Maximum sample size for Stable fit. 0 means use all training increments.",
    )
    parser.add_argument(
        "--skewed-t-fit-max-n",
        type=int,
        default=50000,
        help="Maximum sample size for SkewedT fit. 0 means use all training increments.",
    )
    parser.add_argument(
        "--stable-fit-method",
        choices=["MLE", "MM"],
        default="MLE",
        help="SciPy levy_stable.fit method.",
    )
    parser.add_argument(
        "--stable-pdf-method",
        choices=["piecewise", "dni", "fft-simpson"],
        default="piecewise",
        help="SciPy levy_stable pdf/cdf method used during fit/evaluation.",
    )
    return parser.parse_args()


def _resolve_distribution_order(args: argparse.Namespace) -> list[str]:
    if args.distributions:
        names = args.distributions
    else:
        names = ["gaussian", "laplace"]
        if args.include_stable:
            names.append("stable")

    resolved: list[str] = []
    for raw_name in names:
        key = str(raw_name).strip().lower()
        if key not in DIST_NAME_ALIASES:
            raise ValueError(f"Unsupported distribution: {raw_name}")
        canonical = DIST_NAME_ALIASES[key]
        if canonical not in resolved:
            resolved.append(canonical)
    return resolved


def _write_progress(payload: dict[str, object]) -> None:
    PROGRESS_JSON.parent.mkdir(parents=True, exist_ok=True)
    PROGRESS_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _ensure_dirs() -> None:
    (ROOT / "05_오차증분_분포적합" / "코드").mkdir(parents=True, exist_ok=True)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    BASE_PLOT_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    for folder in (BASE_PLOT_DIR, TABLE_DIR):
        for item in folder.iterdir():
            if item.is_file():
                item.unlink()


def _read_increments(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    for col in ("error_value", "error_price", "delta_error"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "error_value" not in df.columns and "error_price" in df.columns:
        df["error_value"] = df["error_price"]
    if "delta_error" not in df.columns:
        df["delta_error"] = df["error_value"].diff()
    if "split" not in df.columns:
        df["split"] = "all"
    return df


def _extract_series(df: pd.DataFrame, split: str) -> np.ndarray:
    if split == "all":
        sub = df
    else:
        sub = df[df["split"] == split]
    values = pd.to_numeric(sub["delta_error"], errors="coerce").dropna().to_numpy(dtype=float)
    return values[np.isfinite(values)]


def _downsample(values: np.ndarray, max_n: int) -> np.ndarray:
    if len(values) <= max_n:
        return values.copy()
    idx = RNG.choice(len(values), size=max_n, replace=False)
    return np.sort(values[idx])


def _safe_stats(values: np.ndarray) -> dict[str, float]:
    return {
        "n": int(len(values)),
        "mean": float(np.mean(values)) if len(values) else math.nan,
        "std": float(np.std(values, ddof=0)) if len(values) else math.nan,
        "q01": float(np.quantile(values, 0.01)) if len(values) else math.nan,
        "q50": float(np.quantile(values, 0.50)) if len(values) else math.nan,
        "q99": float(np.quantile(values, 0.99)) if len(values) else math.nan,
    }


def _fit_reference(values: np.ndarray) -> tuple[float, float]:
    mean = float(np.mean(values)) if len(values) else 0.0
    std = float(np.std(values, ddof=0)) if len(values) else 1.0
    if not np.isfinite(std) or std <= 0:
        std = 1.0
    return mean, std


def _to_standardized(values: np.ndarray, ref_mean: float, ref_std: float) -> np.ndarray:
    std = ref_std if np.isfinite(ref_std) and ref_std > 0 else 1.0
    return (np.asarray(values, dtype=float) - ref_mean) / std


def _prepare_fit_values(
    name: str,
    train_values: np.ndarray,
    stable_fit_max_n: int,
    skewed_t_fit_max_n: int,
) -> np.ndarray:
    if name == "Stable" and stable_fit_max_n > 0:
        return _downsample(train_values, stable_fit_max_n)
    if name == "SkewedT" and skewed_t_fit_max_n > 0:
        return _downsample(train_values, skewed_t_fit_max_n)
    return train_values.copy()


def _fit_stable_once(fit_values: np.ndarray, stable_fit_method: str, stable_pdf_method: str) -> tuple[float, ...]:
    stats.levy_stable.parameterization = "S1"
    stats.levy_stable.pdf_default_method = stable_pdf_method
    stats.levy_stable.cdf_default_method = "piecewise" if stable_pdf_method == "fft-simpson" else stable_pdf_method
    alpha, beta, loc, scale = stats.levy_stable.fit(fit_values, method=stable_fit_method)
    alpha = float(alpha)
    beta = float(beta)
    loc = float(loc)
    scale = float(scale)
    if not (0 < alpha <= 2 and -1 <= beta <= 1 and np.isfinite(scale) and scale > 0):
        raise ValueError(f"invalid stable params: alpha={alpha}, beta={beta}, scale={scale}")
    return (alpha, beta, loc, max(scale, 1e-8))


def _fit_distribution(
    name: str,
    fit_values: np.ndarray,
    stable_fit_method: str = "MLE",
    stable_pdf_method: str = "piecewise",
) -> tuple[tuple[float, ...], str | None]:
    try:
        if name == "Gaussian":
            mu, sigma = stats.norm.fit(fit_values)
            sigma = max(float(sigma), 1e-8)
            return (float(mu), sigma), None
        if name == "Laplace":
            loc, scale = stats.laplace.fit(fit_values)
            scale = max(float(scale), 1e-8)
            return (float(loc), scale), None
        if name == "SkewedT":
            a, b, loc, scale = stats.jf_skew_t.fit(fit_values)
            a = max(float(a), 1e-8)
            b = max(float(b), 1e-8)
            scale = max(float(scale), 1e-8)
            return (a, b, float(loc), scale), None
        if name == "Stable":
            try:
                return _fit_stable_once(fit_values, stable_fit_method, stable_pdf_method), None
            except Exception as first_exc:  # noqa: BLE001
                if stable_pdf_method != "piecewise":
                    try:
                        return _fit_stable_once(fit_values, stable_fit_method, "piecewise"), f"fallback_from_{stable_pdf_method}: {first_exc}"
                    except Exception as fallback_exc:  # noqa: BLE001
                        return tuple(), f"{first_exc}; piecewise fallback failed: {fallback_exc}"
                return tuple(), str(first_exc)
        return tuple(), f"unsupported distribution: {name}"
    except Exception as exc:  # noqa: BLE001
        return tuple(), str(exc)


def _dist_obj(name: str):
    if name == "Gaussian":
        return stats.norm
    if name == "Laplace":
        return stats.laplace
    if name == "SkewedT":
        return stats.jf_skew_t
    if name == "Stable":
        return stats.levy_stable
    raise ValueError(name)


def _safe_loglik(name: str, params: tuple[float, ...], values: np.ndarray) -> float:
    try:
        logpdf = _dist_obj(name).logpdf(values, *params)
        logpdf = np.asarray(logpdf, dtype=float)
        logpdf = logpdf[np.isfinite(logpdf)]
        if len(logpdf) == 0:
            return math.nan
        return float(np.sum(logpdf))
    except Exception:  # noqa: BLE001
        return math.nan


def _safe_ks(name: str, params: tuple[float, ...], values: np.ndarray) -> float:
    try:
        sample = _downsample(values, 4000)
        if len(sample) == 0:
            return math.nan
        stat = stats.kstest(sample, lambda x: _dist_obj(name).cdf(x, *params)).statistic
        return float(stat)
    except Exception:  # noqa: BLE001
        return math.nan


def _aic_bic(loglik: float, k: int, n: int) -> tuple[float, float]:
    if not np.isfinite(loglik) or n <= 0:
        return math.nan, math.nan
    return float(2 * k - 2 * loglik), float(k * math.log(n) - 2 * loglik)


def _theoretical_quantile(name: str, params: tuple[float, ...], probs: np.ndarray) -> np.ndarray:
    try:
        q = _dist_obj(name).ppf(probs, *params)
        return np.asarray(q, dtype=float)
    except Exception:  # noqa: BLE001
        return np.full_like(probs, np.nan, dtype=float)


def _central_range(values: np.ndarray, q_low: float = DISPLAY_Q_LOW, q_high: float = DISPLAY_Q_HIGH) -> tuple[float, float]:
    low, high = np.quantile(values, [q_low, q_high])
    if not np.isfinite(low) or not np.isfinite(high) or low >= high:
        low = float(np.min(values))
        high = float(np.max(values))
    if low == high:
        pad = max(abs(low) * 0.05, 1.0)
        return low - pad, high + pad
    return float(low), float(high)


def _qq_error_metrics(name: str, params: tuple[float, ...], values: np.ndarray) -> tuple[float, float]:
    if len(values) == 0 or not params:
        return math.nan, math.nan
    probs = np.linspace(QQ_Q_LOW, QQ_Q_HIGH, QQ_POINTS)
    empirical = np.quantile(values, probs)
    theoretical = _theoretical_quantile(name, params, probs)
    mask = np.isfinite(empirical) & np.isfinite(theoretical)
    if mask.sum() == 0:
        return math.nan, math.nan
    diff = theoretical[mask] - empirical[mask]
    return float(np.sqrt(np.mean(diff**2))), float(np.mean(np.abs(diff)))


def _plot_density(
    variable_en: str,
    values: np.ndarray,
    fit_results: dict[str, tuple[float, ...]],
    out_path: Path,
    ref_mean: float | None = None,
    ref_std: float | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 5.0))
    if ref_mean is None or ref_std is None:
        ref_mean, ref_std = _fit_reference(values)
    standardized = _to_standardized(values, ref_mean, ref_std)
    x_low, x_high = _central_range(standardized)
    display_values = standardized[(standardized >= x_low) & (standardized <= x_high)]
    if len(display_values) == 0:
        display_values = standardized
    bins = max(24, min(60, int(np.sqrt(len(display_values)))))
    ax.hist(
        display_values,
        bins=bins,
        density=True,
        color="#d4d4d8",
        alpha=0.65,
        label="Histogram",
        edgecolor="none",
    )
    if len(display_values) > 10:
        try:
            kde = stats.gaussian_kde(display_values)
            x_kde = np.linspace(x_low, x_high, 500)
            ax.plot(x_kde, kde(x_kde), color="#111827", lw=1.8, linestyle=":", label="Empirical KDE")
        except Exception:  # noqa: BLE001
            pass
    z_grid = np.linspace(x_low, x_high, 600)
    raw_grid = z_grid * ref_std + ref_mean
    for name in DIST_ORDER:
        params = fit_results.get(name)
        if not params:
            continue
        try:
            y = _dist_obj(name).pdf(raw_grid, *params) * ref_std
            y = np.asarray(y, dtype=float)
            style = DIST_STYLES[name]
            ax.plot(
                z_grid,
                y,
                lw=2.4,
                color=style["color"],
                linestyle=style["linestyle"],
                label=name,
            )
        except Exception:  # noqa: BLE001
            continue
    ax.set_xlim(float(x_low), float(x_high))
    ax.set_xlabel("Standardized Increment (z)")
    ax.set_ylabel("Density")
    ax.legend(frameon=False, ncol=4, loc="upper right", handlelength=2.8)
    ax.grid(axis="y", color="#e5e7eb", linewidth=0.8)
    ax.grid(axis="x", visible=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_qq(
    values: np.ndarray,
    fit_results: dict[str, tuple[float, ...]],
    out_path: Path,
    ref_mean: float | None = None,
    ref_std: float | None = None,
) -> None:
    sample = np.asarray(values, dtype=float)
    sample = sample[np.isfinite(sample)]
    if len(sample) < 20:
        return
    if ref_mean is None or ref_std is None:
        ref_mean, ref_std = _fit_reference(sample)

    probs = np.linspace(QQ_Q_LOW, QQ_Q_HIGH, QQ_POINTS)
    empirical = _to_standardized(np.quantile(sample, probs), ref_mean, ref_std)
    empirical = np.asarray(empirical, dtype=float)
    q_low, q_high = _central_range(empirical, QQ_Q_LOW, QQ_Q_HIGH)

    fig, ax = plt.subplots(figsize=(6.2, 6.0))
    ax.plot([q_low, q_high], [q_low, q_high], color="#9ca3af", lw=1.2, linestyle="--", zorder=1)
    for name in DIST_ORDER:
        params = fit_results.get(name)
        if not params:
            continue
        theo = _to_standardized(_theoretical_quantile(name, params, probs), ref_mean, ref_std)
        mask = np.isfinite(theo) & np.isfinite(empirical)
        style = DIST_STYLES[name]
        ax.plot(
            empirical[mask],
            theo[mask],
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=2.0,
            label=name,
            zorder=2,
        )
    ax.set_xlim(q_low, q_high)
    ax.set_ylim(q_low, q_high)
    ax.grid(color="#e5e7eb", linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlabel("Empirical Quantile (standardized)")
    ax.set_ylabel("Theoretical Quantile (standardized)")
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = _parse_args()
    global ERROR_ROOT, OUT_ROOT, BASE_PLOT_DIR, TABLE_DIR, GUIDE_MD, PROGRESS_JSON, DATASETS, DIST_ORDER
    ERROR_ROOT = args.error_root.resolve()
    OUT_ROOT = args.out_root.resolve()
    BASE_PLOT_DIR = OUT_ROOT / "01_메인_기본분포비교"
    TABLE_DIR = OUT_ROOT / "03_참고_기본분포_상세표"
    GUIDE_MD = OUT_ROOT / "00_안내_결과폴더.md"
    PROGRESS_JSON = OUT_ROOT / "98_오차증분_분포적합_진행상황.json"
    DATASETS = [
        ("wind", "풍력발전", "Wind", ERROR_ROOT / "01_풍력발전_오차시계열_2018_2021.csv"),
        ("da_energy", "DA에너지", "DA Energy", ERROR_ROOT / "02_DA에너지_오차시계열_2018_2021.csv"),
        ("rt_energy", "RT에너지", "RT Energy", ERROR_ROOT / "03_RT에너지_오차시계열_2018_2021.csv"),
        ("da_reg", "DA_REG가격", "DA REG", ERROR_ROOT / "04_DA_REG가격_오차시계열_2018_2021.csv"),
        ("rt_reg", "RT_REG가격", "RT REG", ERROR_ROOT / "05_RT_REG가격_오차시계열_2018_2021.csv"),
    ]
    DIST_ORDER = _resolve_distribution_order(args)
    only_datasets = {str(x).strip().lower() for x in args.only_datasets} if args.only_datasets else None
    selected_datasets = [item for item in DATASETS if only_datasets is None or item[0] in only_datasets]
    if not selected_datasets:
        raise ValueError(f"No datasets selected. only_datasets={args.only_datasets}")

    _ensure_dirs()
    started_at = time.time()
    _write_progress(
        {
            "status": "started",
            "distributions": DIST_ORDER,
            "selected_datasets": [item[0] for item in selected_datasets],
            "stable_fit_max_n": int(args.stable_fit_max_n),
            "skewed_t_fit_max_n": int(args.skewed_t_fit_max_n),
            "stable_fit_method": args.stable_fit_method,
            "stable_pdf_method": args.stable_pdf_method,
            "elapsed_sec": 0.0,
        }
    )

    comparison_frames: list[pd.DataFrame] = []
    best_rows: list[dict[str, object]] = []
    meta_rows: list[dict[str, object]] = []

    for idx, (key, ko, en, path) in enumerate(selected_datasets, start=1):
        print(f"[dataset {idx}/{len(selected_datasets)}] {key} loading", flush=True)
        df = _read_increments(path)
        train_values = _extract_series(df, "train")
        test_values = _extract_series(df, "test")
        if len(train_values) == 0:
            train_values = _extract_series(df, "all")
        if len(test_values) == 0:
            test_values = train_values.copy()

        ref_mean, ref_std = _fit_reference(train_values)
        eval_train = _downsample(train_values, EVAL_MAX_N)
        eval_test = _downsample(test_values, EVAL_MAX_N)

        rows: list[dict[str, object]] = []
        param_rows: list[dict[str, object]] = []
        fit_results: dict[str, tuple[float, ...]] = {}

        for dist_name in DIST_ORDER:
            fit_values = _prepare_fit_values(
                dist_name,
                train_values,
                int(args.stable_fit_max_n),
                int(args.skewed_t_fit_max_n),
            )
            print(
                f"[dataset {idx}/{len(selected_datasets)}] {key} fitting {dist_name} "
                f"(fit_n={len(fit_values)}, train_n={len(train_values)})",
                flush=True,
            )
            fit_started_at = time.time()
            _write_progress(
                {
                    "status": "fitting",
                    "dataset_key": key,
                    "distribution": dist_name,
                    "fit_n": int(len(fit_values)),
                    "train_n": int(len(train_values)),
                    "elapsed_sec": round(time.time() - started_at, 3),
                }
            )
            params, error = _fit_distribution(
                dist_name,
                fit_values,
                stable_fit_method=args.stable_fit_method,
                stable_pdf_method=args.stable_pdf_method,
            )
            fit_elapsed = time.time() - fit_started_at
            print(
                f"[dataset {idx}/{len(selected_datasets)}] {key} {dist_name} done "
                f"(elapsed_sec={fit_elapsed:.1f}, error={error or 'none'})",
                flush=True,
            )
            if params:
                fit_results[dist_name] = params
            loglik_train = _safe_loglik(dist_name, params, eval_train) if params else math.nan
            loglik_test = _safe_loglik(dist_name, params, eval_test) if params and len(eval_test) > 0 else math.nan
            mean_nll_test = (
                float(-loglik_test / len(eval_test))
                if params and len(eval_test) > 0 and np.isfinite(loglik_test)
                else math.nan
            )
            ks_train = _safe_ks(dist_name, params, eval_train) if params else math.nan
            qq_rmse, qq_mae = _qq_error_metrics(dist_name, params, eval_test if len(eval_test) else eval_train)
            aic, bic = _aic_bic(loglik_train, len(params), len(eval_train))
            rows.append(
                {
                    "dataset_key": key,
                    "dataset_ko": ko,
                    "dataset_en": en,
                    "distribution": dist_name,
                    "fit_n": int(len(fit_values)),
                    "train_eval_n": int(len(eval_train)),
                    "test_eval_n": int(len(eval_test)),
                    "train_loglik": loglik_train,
                    "train_aic": aic,
                    "train_bic": bic,
                    "test_mean_nll": mean_nll_test,
                    "train_ks_stat": ks_train,
                    "qq_rmse_central": qq_rmse,
                    "qq_mae_central": qq_mae,
                    "fit_error": error or "",
                    "fit_elapsed_sec": fit_elapsed,
                }
            )
            if params:
                if dist_name == "Gaussian":
                    param_names = ["mu", "sigma"]
                elif dist_name == "Laplace":
                    param_names = ["loc", "scale"]
                elif dist_name == "SkewedT":
                    param_names = ["a", "b", "loc", "scale"]
                else:
                    param_names = ["alpha", "beta", "loc", "scale"]
                row = {
                    "dataset_key": key,
                    "dataset_ko": ko,
                    "dataset_en": en,
                    "distribution": dist_name,
                }
                row.update({name: value for name, value in zip(param_names, params)})
                row["fit_elapsed_sec"] = fit_elapsed
                row["fit_n"] = int(len(fit_values))
                param_rows.append(row)
            _write_progress(
                {
                    "status": "fit_done",
                    "dataset_key": key,
                    "distribution": dist_name,
                    "fit_n": int(len(fit_values)),
                    "train_n": int(len(train_values)),
                    "fit_elapsed_sec": round(fit_elapsed, 3),
                    "fit_error": error or "",
                    "elapsed_sec": round(time.time() - started_at, 3),
                }
            )

        comp_df = pd.DataFrame(rows)
        params_df = pd.DataFrame(param_rows)
        comp_df.to_csv(TABLE_DIR / f"{idx:02d}_{ko}_오차증분_분포비교.csv", index=False, encoding="utf-8-sig")
        params_df.to_csv(TABLE_DIR / f"{10 + idx:02d}_{ko}_오차증분_적합모수.csv", index=False, encoding="utf-8-sig")

        _plot_density(
            en,
            eval_test if len(eval_test) else eval_train,
            fit_results,
            BASE_PLOT_DIR / f"{20 + idx:02d}_{ko}_오차증분_분포적합_밀도비교.png",
        )
        _plot_qq(
            eval_test if len(eval_test) else eval_train,
            fit_results,
            BASE_PLOT_DIR / f"{30 + idx:02d}_{ko}_오차증분_분포적합_QQ비교.png",
        )

        comparison_frames.append(comp_df)
        best_row = (
            comp_df.sort_values(["train_aic", "test_mean_nll"], ascending=[True, True])
            .iloc[0]
            .to_dict()
        )
        best_rows.append(best_row)
        meta_rows.append(
            {
                "dataset_key": key,
                "dataset_ko": ko,
                "dataset_en": en,
                "source_path": str(path.relative_to(ROOT)),
                "train_stats": _safe_stats(train_values),
                "test_stats": _safe_stats(test_values),
            }
        )

    all_comp = pd.concat(comparison_frames, ignore_index=True)
    all_comp.to_csv(OUT_ROOT / "41_통합_오차증분_분포비교요약.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(best_rows).to_csv(OUT_ROOT / "42_통합_오차증분_최적분포요약.csv", index=False, encoding="utf-8-sig")
    GUIDE_MD.write_text(
        "\n".join([
            "# Result Folder Guide",
            "",
            "- Root: summary tables, metadata, and explanation files.",
            "- `01_메인_기본분포비교`: Gaussian/Laplace/SkewedT density and QQ plots; Stable is included when the script is run with `--include-stable` or `--distributions stable`.",
            "- `02_메인_레비점프_전체비교`: Levy-jump full-increment density and QQ plots.",
            "- `03_참고_기본분포_상세표`: per-variable distribution comparison and fitted parameter tables.",
            "- `04_참고_레비점프_구성요소`: levy-jump component diagnostics.",
            "- `90_레거시_레비점프_보관`: superseded levy-jump plots.",
        ]),
        encoding="utf-8",
    )
    (OUT_ROOT / "99_오차증분_분포적합_메타정보.json").write_text(
        json.dumps(
            {
                "datasets": meta_rows,
                "distributions": DIST_ORDER,
                "stable_fit_max_n": int(args.stable_fit_max_n),
                "skewed_t_fit_max_n": int(args.skewed_t_fit_max_n),
                "stable_fit_method": args.stable_fit_method,
                "stable_pdf_method": args.stable_pdf_method,
                "source_root": str(ERROR_ROOT.relative_to(ROOT)),
                "output_root": str(OUT_ROOT.relative_to(ROOT)),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    _write_progress(
        {
            "status": "completed",
            "distributions": DIST_ORDER,
            "selected_datasets": [item[0] for item in selected_datasets],
            "elapsed_sec": round(time.time() - started_at, 3),
            "summary_csv": str((OUT_ROOT / "41_통합_오차증분_분포비교요약.csv").relative_to(ROOT)),
        }
    )


if __name__ == "__main__":
    main()



