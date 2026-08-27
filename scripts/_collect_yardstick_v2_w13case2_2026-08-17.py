"""W13根治 案2 (patch-NCC HSV ANDガード) の28チャンク再収集 (2026-08-17)。

`scripts/_collect_yardstick_v2_bc_2026-08-15.py` と全く同じ28チャンク
(NEEDED_CHUNKS, CHUNK_SEC, CHUNK_OFFSET_FRACTIONS) を、本番採用構成
(collect_flags(), placement_override込み) + --enable-patch-fp-hsv-guard
の単体構成、および案1+2併用構成 (--enable-highlight-override も同時ON) の
2構成で再収集する。

比較対象 (既存npz、再収集不要):
    data/indicators_v2/yardstick_v2_boards_c1p_fix_2026-08-15
    (= 現行本番採用構成そのもの、物差しv2 95.63%、タグ "a")
    data/indicators_v2/yardstick_v2_boards_w13_2026-08-16
    (= 案1 単体、タグ "w13")

使い方:
    PYTHONPATH=. ./venv/bin/python -m scripts._collect_yardstick_v2_w13case2_2026-08-17 --config p2
    PYTHONPATH=. ./venv/bin/python -m scripts._collect_yardstick_v2_w13case2_2026-08-17 --config both
"""
from __future__ import annotations

import argparse
import importlib
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent

# ファイル名にハイフンを含むため import 文でなく importlib で動的 import する。
_bc = importlib.import_module("scripts._collect_yardstick_v2_bc_2026-08-15")

FLAG_PATCH_FP_HSV_GUARD: str = "--enable-patch-fp-hsv-guard"
FLAG_BOTH: str = "--enable-highlight-override --enable-patch-fp-hsv-guard"

OUT_DIRS: dict[str, str] = {
    "p2": "yardstick_v2_boards_w13p2_2026-08-17",
    "both": "yardstick_v2_boards_w13both_2026-08-17",
}
EXTRA_FLAGS: dict[str, str] = {
    "p2": FLAG_PATCH_FP_HSV_GUARD,
    "both": FLAG_BOTH,
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", choices=["p2", "both"], required=True)
    args = ap.parse_args()

    out_dir = _ROOT / "data" / "indicators_v2" / OUT_DIRS[args.config]
    extra_flags = EXTRA_FLAGS[args.config]
    log_dir = _ROOT / "logs" / "yardstick_v2_collect_w13case2_2026-08-17" / args.config
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    jobs = _bc.build_jobs(out_dir, log_dir, extra_flags)
    print(f"[{args.config}] ジョブ数: {len(jobs)} extra_flags={extra_flags!r}")
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
