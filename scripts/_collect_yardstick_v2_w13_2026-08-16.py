"""W13根治 (highlight override) 単体の28チャンク再収集 (2026-08-16)。

`scripts/_collect_yardstick_v2_bc_2026-08-15.py` と全く同じ28チャンク
(NEEDED_CHUNKS, CHUNK_SEC, CHUNK_OFFSET_FRACTIONS) を、本番採用構成
(collect_flags(), placement_override込み) + --enable-highlight-override
の単体構成で再収集する。

比較対象 (既存npz、再収集不要):
    data/indicators_v2/yardstick_v2_boards_c1p_fix_2026-08-15
    (= 現行本番採用構成そのもの、物差しv2 95.63%、
    src/production_config.py RECOGNITION_ADOPTED 参照)

使い方:
    PYTHONPATH=. ./venv/bin/python -m scripts._collect_yardstick_v2_w13_2026-08-16
"""
from __future__ import annotations

import importlib
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent

# ファイル名にハイフンを含むため import 文でなく importlib で動的 import する。
_bc = importlib.import_module("scripts._collect_yardstick_v2_bc_2026-08-15")

FLAG_HIGHLIGHT_OVERRIDE: str = "--enable-highlight-override"

OUT_DIR_NAME: str = "yardstick_v2_boards_w13_2026-08-16"
LOG_DIR: Path = _ROOT / "logs" / "yardstick_v2_collect_w13_2026-08-16"


def main() -> None:
    out_dir = _ROOT / "data" / "indicators_v2" / OUT_DIR_NAME
    out_dir.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    jobs = _bc.build_jobs(out_dir, LOG_DIR, FLAG_HIGHLIGHT_OVERRIDE)
    print(f"[w13] ジョブ数: {len(jobs)} extra_flags={FLAG_HIGHLIGHT_OVERRIDE!r}")
    t0 = time.monotonic()
    results = []
    with ThreadPoolExecutor(max_workers=_bc.MAX_PARALLEL_WORKERS) as ex:
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
