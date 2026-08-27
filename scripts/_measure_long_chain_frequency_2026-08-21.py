"""実データ (学習データ62本のnpz) で「長い連鎖の場面が全体の何%か」を測る
(2026-08-21、user指示③、`exact_shallow`/`beam` の振り分け頻度の実測)。

方法: 62本の npz (`data/indicators_v2/boards_lean_model50v2_2026-08-20/`) の
各盤面 (両プレイヤー、STABLE想定、窒息盤面除外) で既存指標
`current_max_chain` (takapt定石探索、既存関数、再実装しない) を評価する。

**注記 (正直な注記)**: これは「実際に発火した連鎖数」の履歴的再構成ではない
(score差分からの発火イベント抽出は既知の困難問題、
memory `project_fire_event_fragmentation_2026-08-02` 参照)。
`current_max_chain` は「この盤面から到達可能な最大連鎖」の指標であり、
既存パイプライン (`scripts/label_exchange_outcome.py:809`
`"approx_fire_chains": max(1.0, fire_feats["current_max_chain"])`) でも
同じ考え方が「発火連鎖数の近似」として採用されている (本測定はこの
既存の使い方を踏襲する代理指標)。

各盤面について:
    1. current_max_chain を評価
    2. `estimate_chain_anim_duration_sec(chain_count)` で時間予算に変換
    3. `_time_budget_to_beam_depth` で手数に変換し、
       `EXACT_SHALLOW_MAX_DEPTH` 以下 (exact_shallow行き) か超える
       (beam行き) かを分類

cProfile 禁止 (perf_counter使用、ただしこの測定自体は速度ではなく頻度を
測るもの)。
"""
from __future__ import annotations

import glob
import sys
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.board import COLOR_UNKNOWN, Board  # noqa: E402
from src.chain import ChainSimulator  # noqa: E402
from src.console_init import init_console  # noqa: E402

init_console()

import src.indicators_v2 as iv  # noqa: E402
from scripts.mc_counter_estimator import (  # noqa: E402
    EXACT_SHALLOW_MAX_DEPTH,
    _time_budget_to_beam_depth,
)

_DATA_DIR = _ROOT / "data" / "indicators_v2" / "boards_lean_model50v2_2026-08-20"
# 62本全部は重いため、動画あたりのサンプル数を抑える (代表性を保ちつつ
# 現実的な実行時間にする、シーンからの逆算ではなくランダムサンプル)。
_SAMPLES_PER_VIDEO: int = 60
_RNG_SEED: int = 20260821


def _iter_sampled_boards() -> "list[Board]":
    files = sorted(glob.glob(str(_DATA_DIR / "*.npz")))
    rng = np.random.RandomState(_RNG_SEED)
    boards: "list[Board]" = []
    for path in files:
        data = np.load(path, allow_pickle=True)
        grids = data["grids"]
        n = grids.shape[0]
        if n == 0:
            continue
        idxs = rng.choice(n, size=min(_SAMPLES_PER_VIDEO, n), replace=False)
        for i in idxs:
            grid = grids[i].astype(np.uint8)
            if np.any(grid == COLOR_UNKNOWN):
                continue
            b = Board()
            b._grid = grid
            if b.is_dead():
                continue
            if int((grid != 0).sum()) == 0:
                continue  # 空盤面 (試合開始直後等) は「連鎖の場面」ではない
            boards.append(b)
    return boards


def main() -> int:
    t0 = time.perf_counter()
    boards = _iter_sampled_boards()
    print(f"サンプル盤面数: {len(boards)} ({_DATA_DIR.name} 全{len(sorted(glob.glob(str(_DATA_DIR/'*.npz'))))}本から"
          f"動画あたり最大{_SAMPLES_PER_VIDEO}件抽出)")

    sim = ChainSimulator()
    chain_counts: "list[int]" = []
    for b in boards:
        chain_counts.append(int(iv.current_max_chain(b, simulator=sim).raw))
    elapsed = time.perf_counter() - t0
    print(f"current_max_chain 評価完了 ({elapsed:.1f}秒)")

    arr = np.array(chain_counts)
    print()
    print("current_max_chain の分布:")
    print(f"  中央値={np.median(arr):.1f} p25={np.percentile(arr,25):.1f} "
          f"p75={np.percentile(arr,75):.1f} 最大={arr.max()}  ゼロ率={np.mean(arr==0)*100:.1f}%")

    depths = np.array([
        _time_budget_to_beam_depth(float(iv.estimate_chain_anim_duration_sec(float(c))))
        for c in chain_counts
    ])
    short_ratio = float(np.mean(depths <= EXACT_SHALLOW_MAX_DEPTH))
    print()
    print(f"EXACT_SHALLOW_MAX_DEPTH={EXACT_SHALLOW_MAX_DEPTH} 手 での振り分け:")
    print(f"  exact_shallow行き (depth<={EXACT_SHALLOW_MAX_DEPTH}): {short_ratio*100:.1f}%")
    print(f"  beam行き (depth>{EXACT_SHALLOW_MAX_DEPTH}): {(1-short_ratio)*100:.1f}%")
    print()
    print("深さの分布 (depth = 手数換算):")
    for d in sorted(set(depths.tolist())):
        pct = float(np.mean(depths == d)) * 100
        print(f"  depth={d:2d}: {pct:5.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
