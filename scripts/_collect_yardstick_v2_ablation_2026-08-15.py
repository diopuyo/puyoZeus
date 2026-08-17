"""構成c (OJAMA_FALL系3フラグ同時ON) の単体切り分けアブレーション収集 (2026-08-15)。

構成c = --enable-ojama-fall-placement-override + entry_hardening + scoped_exit
の3枚同時ONで net +1.19pt だが満杯帯-4.11pt/終盤-2.16pt/新規誤り57セルが
併発した (data/verify/yardstick_v2_2026-08-14/scoring/ 参照)。どのフラグが
改善を担い、どのフラグが悪化を持ち込んでいるかをフラグ単体で切り分けるため、
`scripts/_collect_yardstick_v2_bc_2026-08-15.py` と全く同じ28チャンク
(NEEDED_CHUNKS, CHUNK_SEC, CHUNK_OFFSET_FRACTIONS) を単体構成で再収集する。
チャンク切り出しロジックの重複を避けるため bc スクリプトを
importlib で動的 import して build_jobs/run_job をそのまま再利用する
(ファイル名にハイフンを含むため通常の import 文は使えない)。

    c1 : --enable-ojama-fall-placement-override のみ
    c2 : --enable-ojama-fall-entry-hardening のみ
    c3 : --enable-ojama-fall-scoped-exit のみ
    c12/c13/c23 : 2枚combo (単体で「改善維持+悪化なし」が見えた場合の追試用、
                  最大2構成まで追加可という手順4の指示に対応)

使い方:
    PYTHONPATH=. ./venv/bin/python -m scripts._collect_yardstick_v2_ablation_2026-08-15 --config c1
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

FLAG_PLACEMENT_OVERRIDE: str = "--enable-ojama-fall-placement-override"
FLAG_ENTRY_HARDENING: str = "--enable-ojama-fall-entry-hardening"
FLAG_SCOPED_EXIT: str = "--enable-ojama-fall-scoped-exit"

# config名 -> (extra_flags, npz出力ディレクトリ名)
CONFIGS: dict[str, tuple[str, str]] = {
    "c1": (FLAG_PLACEMENT_OVERRIDE, "yardstick_v2_boards_c1_placement_2026-08-15"),
    "c2": (FLAG_ENTRY_HARDENING, "yardstick_v2_boards_c2_entryhard_2026-08-15"),
    "c3": (FLAG_SCOPED_EXIT, "yardstick_v2_boards_c3_scopedexit_2026-08-15"),
    "c12": (f"{FLAG_PLACEMENT_OVERRIDE} {FLAG_ENTRY_HARDENING}",
            "yardstick_v2_boards_c12_2026-08-15"),
    "c13": (f"{FLAG_PLACEMENT_OVERRIDE} {FLAG_SCOPED_EXIT}",
            "yardstick_v2_boards_c13_2026-08-15"),
    "c23": (f"{FLAG_ENTRY_HARDENING} {FLAG_SCOPED_EXIT}",
            "yardstick_v2_boards_c23_2026-08-15"),
}

LOG_DIR_ROOT: Path = _ROOT / "logs" / "yardstick_v2_collect_ablation_2026-08-15"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", choices=sorted(CONFIGS), required=True)
    args = ap.parse_args()

    extra_flags, out_dir_name = CONFIGS[args.config]
    out_dir = _ROOT / "data" / "indicators_v2" / out_dir_name
    log_dir = LOG_DIR_ROOT / args.config
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    jobs = _bc.build_jobs(out_dir, log_dir, extra_flags)
    print(f"[config={args.config}] ジョブ数: {len(jobs)} extra_flags={extra_flags!r}")
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
