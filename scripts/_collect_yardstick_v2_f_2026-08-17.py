"""W23根治 (2026-08-17) の統一測定 構成F 収集。

`scripts/_collect_yardstick_v2_r2w10_2026-08-17.py` と同じ28チャンクを、
構成E (本番採用構成 + R2 + W10ガード + override_color_guard + ojama_column_stack_fix)
にさらに `--enable-next-history-starvation-fix` (W23根治) を足した構成Fで
再収集する:

    構成F: 構成E + --enable-next-history-starvation-fix

使い方:
    PYTHONPATH=. ./venv/bin/python -m scripts._collect_yardstick_v2_f_2026-08-17
"""
from __future__ import annotations

import importlib
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent

# ファイル名にハイフンを含むため import 文でなく importlib で動的 import する。
_bc = importlib.import_module("scripts._collect_yardstick_v2_bc_2026-08-15")
_e = importlib.import_module("scripts._collect_yardstick_v2_e_2026-08-17")

FLAG_F: str = f"{_e.FLAG_E} --enable-next-history-starvation-fix"
OUT_DIR_NAME: str = "yardstick_v2_boards_f_2026-08-17"

SHARED_MAX_PARALLEL_WORKERS: int = 14


def build_jobs() -> list:
    out_dir = _ROOT / "data" / "indicators_v2" / OUT_DIR_NAME
    log_dir = _ROOT / "logs" / "yardstick_v2_collect_f_2026-08-17"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    return _bc.build_jobs(out_dir, log_dir, FLAG_F)


def main() -> None:
    jobs = build_jobs()
    print(f"[f] ジョブ数: {len(jobs)} (workers={SHARED_MAX_PARALLEL_WORKERS})")
    t0 = time.monotonic()
    results = []
    with ThreadPoolExecutor(max_workers=SHARED_MAX_PARALLEL_WORKERS) as ex:
        for r in ex.map(_bc.run_job, jobs):
            results.append(r)
    n_ok = sum(1 for _, ok, _ in results if ok)
    print(f"完了: {n_ok}/{len(results)} 成功 ({time.monotonic() - t0:.1f}s)")
    if n_ok < len(results):
        print("失敗ジョブ:")
        for name, ok, _ in results:
            if not ok:
                print(f"  {name}")


if __name__ == "__main__":
    main()
