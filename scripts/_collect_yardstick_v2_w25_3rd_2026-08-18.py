"""W25根治 第3弾・最終 (2026-08-18) の統一測定 構成F+第3弾フラグ 収集。

`scripts/_collect_yardstick_v2_f_2026-08-17.py` と同じ28チャンクを、
構成F にさらに `--enable-ojama-write-accounting-guard` (第3弾、CNN観測
入力段の会計整合フィルタ) を足した構成で再収集する:

    構成F+第3弾: 構成F + --enable-ojama-write-accounting-guard

比較対象の 構成F ベースラインは既に
data/indicators_v2/yardstick_v2_boards_f_2026-08-17/ に収集済 (再収集不要)。

使い方:
    PYTHONPATH=. ./venv/bin/python -m scripts._collect_yardstick_v2_w25_3rd_2026-08-18
"""
from __future__ import annotations

import importlib
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent

_bc = importlib.import_module("scripts._collect_yardstick_v2_bc_2026-08-15")
_f = importlib.import_module("scripts._collect_yardstick_v2_f_2026-08-17")

FLAG_W25_3RD: str = f"{_f.FLAG_F} --enable-ojama-write-accounting-guard"
OUT_DIR_NAME: str = "yardstick_v2_boards_w25_3rd_2026-08-18"

SHARED_MAX_PARALLEL_WORKERS: int = 14


def build_jobs() -> list:
    out_dir = _ROOT / "data" / "indicators_v2" / OUT_DIR_NAME
    log_dir = _ROOT / "logs" / "yardstick_v2_collect_w25_3rd_2026-08-18"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    return _bc.build_jobs(out_dir, log_dir, FLAG_W25_3RD)


def main() -> None:
    jobs = build_jobs()
    print(f"[w25_3rd] ジョブ数: {len(jobs)} (workers={SHARED_MAX_PARALLEL_WORKERS})")
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
