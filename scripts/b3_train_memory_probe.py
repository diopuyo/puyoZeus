"""実データは最初のHGBビン化までで必ず停止する。予測・採点・モデル保存は禁止。"""
from __future__ import annotations

import os

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import argparse
import json
import resource
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from src.b3_column_spec import COLUMN_NAMES, EXCLUDED_FAST_COLUMNS
from src.b3_eval_cv import candidate_trainer, run_cv
from src.b3_train_data import list_complete_parquets, load_training_rows, precheck_rows
from src.b3_train_folds import games_per_group_from_rows, outer_assignment
from src.b3_train_matrix import MATRIX_ATTR
from src.b3_train_model import feature_columns

GIB = 1024 ** 3
KIB = 1024
SEED = 1003
FOLD = 0
# 木成長時の勾配・Hessian・行索引・ヒストグラム・予測配列等の保守的な追加枠。
GROWTH_RESERVE_GIB = 0.5
BIN_SUBSAMPLE_ROWS = 200_000  # sklearn 1.8 _BinMapper の既定上限


class ProbeComplete(Exception):
    """ビン化後、最初の木を作る前の計測専用停止。"""


def memory() -> dict[str, float]:
    resident_pages = int(Path("/proc/self/statm").read_text().split()[1])
    return {"rss_gib": resident_pages * os.sysconf("SC_PAGE_SIZE") / GIB,
            "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * KIB / GIB}


def measure(df: pd.DataFrame, gmap: dict[str, str], cols: list[str]) -> dict:
    report = {"loaded": memory(), "rows": len(df), "features": len(cols),
              "matrix_gib": df.attrs[MATRIX_ATTR].values.nbytes / GIB}
    original = HistGradientBoostingClassifier._bin_data

    def stop_after_bins(model: object, X: np.ndarray, is_training_data: bool) -> None:
        report["fit_input"] = {**memory(), "shape": list(X.shape), "dtype": str(X.dtype)}
        bins = original(model, X, is_training_data)
        report["binned"] = {**memory(), "binned_gib": bins.nbytes / GIB}
        raise ProbeComplete()

    with patch.object(HistGradientBoostingClassifier, "_bin_data", stop_after_bins):
        try:
            run_cv("memory_only", df, candidate_trainer(), gmap, [SEED], cols, folds=[FOLD])
        except ProbeComplete:
            report["stopped_before_first_tree"] = True
        else:
            raise AssertionError("計測専用停止が発動していない")
    outer = outer_assignment(games_per_group_from_rows(df, gmap))
    fold = df.video_id.map(gmap).map(outer).to_numpy()
    mode = df.mode_exchange.to_numpy()
    max_rows = max(int(((fold != k) & (mode == m)).sum()) for k in set(fold) for m in (0, 1))
    base = report["loaded"]["rss_gib"]
    peak = report["binned"]["peak_rss_gib"]
    fit_bytes = int(np.prod(report["fit_input"]["shape"])) * np.dtype(np.float64).itemsize
    overhead = max(0.0, report["fit_input"]["rss_gib"] - base - fit_bytes / GIB)
    # 全列が有効な最大状態を仮定し、float64入力+uint8ビンとビン化標本を積み上げる。
    arrays = max_rows * len(cols) * (np.dtype(np.float64).itemsize + np.dtype(np.uint8).itemsize) / GIB
    sample = min(max_rows, BIN_SUBSAMPLE_ROWS) * len(cols) * np.dtype(np.float64).itemsize / GIB
    report.update(max_outer_state_rows=max_rows, fit_metadata_overhead_gib=overhead,
                  full_input_and_bins_gib=arrays, bin_subsample_gib=sample,
                  estimated_full_peak_gib=max(peak, base + overhead + arrays + sample + GROWTH_RESERVE_GIB),
                  growth_reserve_gib=GROWTH_RESERVE_GIB)
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rows-dir", type=Path, required=True)
    ap.add_argument("--assignment", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    assignment = pd.read_csv(a.assignment, sep="\t")
    gmap = {v: row.group for row in assignment.itertuples() for v in row.videos.split(",")}
    cols = feature_columns(COLUMN_NAMES)
    df = load_training_rows(list_complete_parquets(a.rows_dir), cols)
    pre = precheck_rows(df, gmap)
    if pre["violations_total"]:
        raise ValueError("前提検査に違反あり")
    report = measure(df, gmap, cols)
    report.update(seed=SEED, fold=FOLD, hgb_threads=os.environ.get("B3_HGB_THREADS", "1"),
                  excluded_fast_count=len(EXCLUDED_FAST_COLUMNS), predictions_saved=False)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
