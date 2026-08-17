"""認識強化統一測定 (2026-08-17): R2 (浮きぷよ復元) / W10ガード (着地色継続監視)
の物差しv2再収集。

`scripts/_collect_yardstick_v2_bc_2026-08-15.py` と同じ28チャンク
(NEEDED_CHUNKS, CHUNK_SEC, CHUNK_OFFSET_FRACTIONS) を、以下3構成で再収集する
(構成A=現行本番採用構成そのものは data/verify/yardstick_v2_2026-08-14/
scoring_ablation/score_w13p2.json が既に採点済みのため再収集不要):

    構成B: 本番採用構成 + --enable-floating-gap-restore (R2)
    構成C: 本番採用構成 + --enable-landing-color-guard (W10ガード、
           collect_boards_lean.py の CLI 配線漏れを本タスクで是正済み)
    構成D: 本番採用構成 + 両方

使い方:
    PYTHONPATH=. ./venv/bin/python -m scripts._collect_yardstick_v2_r2w10_2026-08-17 --config all
    PYTHONPATH=. ./venv/bin/python -m scripts._collect_yardstick_v2_r2w10_2026-08-17 --config b
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

FLAG_R2: str = "--enable-floating-gap-restore"
FLAG_W10: str = "--enable-landing-color-guard"
FLAG_BOTH: str = f"{FLAG_R2} {FLAG_W10}"

OUT_DIRS: dict[str, str] = {
    "b": "yardstick_v2_boards_r2_2026-08-17",
    "c": "yardstick_v2_boards_w10guard_2026-08-17",
    "d": "yardstick_v2_boards_r2w10_2026-08-17",
}
EXTRA_FLAGS: dict[str, str] = {
    "b": FLAG_R2,
    "c": FLAG_W10,
    "d": FLAG_BOTH,
}

# CPU が空いているため 3 構成 x 28 チャンク = 84 ジョブを共有プールで並列化する
# (feedback_speed_cost_priority: 並列最大化、cv2 は _collect_lean_1t.py 側で
# 1 スレッド固定済みのためプロセス数を増やしても競合しない)。
SHARED_MAX_PARALLEL_WORKERS: int = 14


def build_all_jobs(configs: list[str]) -> list:
    jobs = []
    for cfg in configs:
        out_dir = _ROOT / "data" / "indicators_v2" / OUT_DIRS[cfg]
        log_dir = _ROOT / "logs" / "yardstick_v2_collect_r2w10_2026-08-17" / cfg
        out_dir.mkdir(parents=True, exist_ok=True)
        log_dir.mkdir(parents=True, exist_ok=True)
        jobs.extend(_bc.build_jobs(out_dir, log_dir, EXTRA_FLAGS[cfg]))
    return jobs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", choices=["b", "c", "d", "all"], default="all")
    args = ap.parse_args()

    configs = ["b", "c", "d"] if args.config == "all" else [args.config]
    jobs = build_all_jobs(configs)
    print(f"[{configs}] ジョブ数: {len(jobs)} (workers={SHARED_MAX_PARALLEL_WORKERS})")
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
