#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

SERIES_KEY = "wind"


def _load_common() -> object:
    module_path = Path(__file__).resolve().parent / "90_점예측_공통유틸.py"
    spec = importlib.util.spec_from_file_location("point_common", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load point helper from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_COMMON = _load_common()


def main() -> None:
    roots = _COMMON.resolve_project_roots(__file__)
    spec = _COMMON.SERIES_SPECS[SERIES_KEY]
    ap = argparse.ArgumentParser(description="Train richer HistGBR wind point forecast model.")
    ap.add_argument("--dataset-csv", default=str(roots["prep_root"] / str(spec["dataset_name"])))
    ap.add_argument("--out-root", default=str(roots["internal_root"]))
    args = ap.parse_args()

    df = _COMMON.load_series_dataset(Path(args.dataset_csv), spec)
    pred_df, metrics, coef_df, model_bundle = _COMMON.train_direct_model(df, spec)
    outputs = _COMMON.write_model_outputs(pred_df, metrics, coef_df, model_bundle, Path(args.out_root), str(spec["prefix"]))
    print(json.dumps({"series_key": SERIES_KEY, "dataset_csv": args.dataset_csv, "outputs": outputs, "metrics": metrics}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
