#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

SERIES_KEY = "rt_reg"


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
    ap = argparse.ArgumentParser(description="Export RT_REG point forecast results.")
    ap.add_argument("--pred-root", default=str(roots["internal_root"]))
    ap.add_argument("--user-out-root", default=str(roots["result_root"]))
    ap.add_argument("--meta-out-root", default="")
    args = ap.parse_args()

    series_out = _COMMON.export_nonreg_series(SERIES_KEY, Path(args.pred_root), Path(args.user_out_root))
    summary_out = _COMMON.rebuild_nonreg_export_summary(
        Path(args.pred_root),
        Path(args.user_out_root),
        Path(args.meta_out_root) if str(args.meta_out_root).strip() else Path(args.pred_root),
    )
    print(json.dumps({"series_output": series_out, "summary_refresh": summary_out}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
