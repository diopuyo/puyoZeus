"""c1' (placement_override 修正版) 単体の28チャンク再収集 (2026-08-15)。

src/ojama_visual_detector.py の `_confirm_placement_evidence` 修正版
(evidence一発判定の欠陥修正: own_score_delta の自chain roll-up除外のみ、
実時間ヒステリシス案は55盤面物差しで有意悪化と実測し不採用に確定 —
`_confirm_placement_evidence` の docstring 参照) を適用した状態で、
`scripts/_collect_yardstick_v2_bc_2026-08-15.py` と全く同じ28チャンクを
--enable-ojama-fall-placement-override 単体で再収集する。

旧 c1 (data/indicators_v2/yardstick_v2_boards_c1_placement_2026-08-15) は
欠陥版のまま保持し、 本スクリプトの出力 (c1') と対比できるようにする。

使い方:
    PYTHONPATH=. ./venv/bin/python -m scripts._collect_yardstick_v2_c1p_2026-08-15
"""
from __future__ import annotations

import importlib
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent

# ファイル名にハイフンを含むため import 文でなく importlib で動的 import する。
_bc = importlib.import_module("scripts._collect_yardstick_v2_bc_2026-08-15")

FLAG_PLACEMENT_OVERRIDE: str = "--enable-ojama-fall-placement-override"

OUT_DIR_NAME: str = "yardstick_v2_boards_c1p_fix_2026-08-15"
LOG_DIR: Path = _ROOT / "logs" / "yardstick_v2_collect_c1p_2026-08-15"


def main() -> None:
    out_dir = _ROOT / "data" / "indicators_v2" / OUT_DIR_NAME
    out_dir.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    jobs = _bc.build_jobs(out_dir, LOG_DIR, FLAG_PLACEMENT_OVERRIDE)
    print(f"[c1'] ジョブ数: {len(jobs)} extra_flags={FLAG_PLACEMENT_OVERRIDE!r}")
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
