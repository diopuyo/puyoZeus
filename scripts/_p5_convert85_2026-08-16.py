"""W12根治 P5先行検証: 真値npz85本のみでlabeled_win CSVを再生成する。

## 背景 (2026-08-16、63本再収集ジョブと同時並行)
63本の再収集 (`scripts/_regen148_w12_recollect_2026-08-16.py`) はWSLで独立
走行中で完了に半日以上かかる。待たずに、既に真値
(`ojama_forecast`/`ojama_net_balance` 列)を持つ85本だけで新5列
(`ojama_forecast_uncapped`/`ojama_net_balance_uncapped`/`ojama_forecast_log`/
`ojama_forecast_progress_interaction`/`color_forecast_ratio_own`) の効果を
先行判定する。既存の148本用ディレクトリ
(`data/indicators_v2/boards_lean_phase_l_2026-08-11/`) を一切変更せず、
真値を持つ85本だけをsymlinkした3バッチディレクトリ (`_p5_split_batches`で
事前作成済み) に対して `scripts.build_labeled_win_from_npz` を並列3プロセス
で起動し、完了後にCSVを結合する。

## 並列数=3の理由 (user指示)
63本再収集ジョブが10並列でCPUを使用中 (`_regen148_w12_recollect_2026-08-16.py`
のオーケストレータ)。新規に重い `--profile full` 変換 (連鎖シミュ含む)
を大量並列で追加投入すると競合し、優先度の高い63本再収集が遅延する。
このため本スクリプトは3並列 (既存3バッチディレクトリと対応) に固定し、
`nice` で優先度も下げる (`_NICE_LEVEL`)。

## 実行方法 (WSL detach)
    wsl -d Ubuntu -- bash -c "cd /mnt/c/.../puyo_analyzer && \\
        setsid -f bash -c 'PYTHONPATH=. ./venv/bin/python \\
        -m scripts._p5_convert85_2026-08-16 \\
        > logs/p5_convert85_2026-08-16.log 2>&1 < /dev/null'"
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

_PROJ_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJ_ROOT))

# 事前に作成済みの3バッチディレクトリ (真値npz85本を28-29本ずつsymlink、
# `_gen_w12_recollect_targets_2026-08-16.py` とは無関係の別作業)。
BATCH_DIRS: tuple[str, ...] = (
    "data/indicators_v2/boards_lean_phase_l_85_batch0_2026-08-16",
    "data/indicators_v2/boards_lean_phase_l_85_batch1_2026-08-16",
    "data/indicators_v2/boards_lean_phase_l_85_batch2_2026-08-16",
)
OUT_DIR = Path("data/verify/labeled_win_w12_85_2026-08-16")
OUT_MERGED_CSV = OUT_DIR / "labeled_win_w12_85.csv"

_NICE_LEVEL: int = 10  # 63本再収集ジョブより優先度を下げる (nice値を大きく)


def _launch_batch(idx: int, npz_dir: str) -> subprocess.Popen:
    out_csv = OUT_DIR / f"batch{idx}.csv"
    log_path = OUT_DIR / f"batch{idx}.log"
    cmd = [
        "nice", "-n", str(_NICE_LEVEL),
        "./venv/bin/python", "-m", "scripts.build_labeled_win_from_npz",
        "--npz-dir", npz_dir, "--out", str(out_csv), "--profile", "full",
    ]
    log_f = open(log_path, "w", encoding="utf-8")
    print(f"[launch] batch{idx}: {npz_dir} -> {out_csv}")
    return subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    procs = [_launch_batch(i, d) for i, d in enumerate(BATCH_DIRS)]
    print(f"[wait] {len(procs)}バッチの完了待ち (nice={_NICE_LEVEL})...")
    rcs = [p.wait() for p in procs]
    elapsed = time.time() - t0
    print(f"[done] 全バッチ変換完了 ({elapsed/60:.1f}分)、リターンコード={rcs}")
    if any(rc != 0 for rc in rcs):
        print("[ERROR] 一部バッチが異常終了。マージを中止する")
        return 1

    print("[merge] CSV結合開始")
    dfs = [pd.read_csv(OUT_DIR / f"batch{i}.csv") for i in range(len(BATCH_DIRS))]
    merged = pd.concat(dfs, ignore_index=True)
    merged.to_csv(OUT_MERGED_CSV, index=False)
    print(f"[merge] {len(merged)}行 -> {OUT_MERGED_CSV}"
          f" (動画数={merged['video_id'].nunique()})")
    print("P5_CONVERT85_DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
