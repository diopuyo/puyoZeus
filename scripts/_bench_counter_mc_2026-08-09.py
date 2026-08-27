"""打ち合い応手 MC がリアルタイムに乗るかを実測する (2026-08-09).

user 確認: 「モンテカルロ入りだとリアルタイムの認識厳しそうですか」

## 測ること
`scripts/mc_counter_estimator.estimate_counter_distribution` の 1 回あたりの
所要時間を、 ロールアウト本数と時間予算 (= 連鎖数) を変えて実測する。

## 判定基準
配信オーバーレイの目標は 30fps = **1 フレーム 33.3ms**。
認識自体が既に 31fps 出ている (= ほぼ 32ms を使い切っている) ため、
MC に割ける余裕はごくわずか。 ただし **MC は毎フレーム回す必要がない**
(盤面が変わったときだけ) ので、 呼び出し頻度も併せて考える。

読み取り専用のベンチマーク。
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.board import BOARD_COLS, BOARD_ROWS, Board  # noqa: E402
from src.console_init import init_console  # noqa: E402

init_console()

import scripts.mc_counter_estimator as mc  # noqa: E402
import src.indicators_v2 as iv  # noqa: E402

# 実戦に近い盤面 (半分ほど積んだ状態) を作る
COLORS = (1, 2, 3, 4)
THRESHOLD = 12.0
# 30fps の 1 フレーム予算
FRAME_BUDGET_MS: float = 1000.0 / 30.0


def _make_board(fill_rows: int, seed: int = 0) -> Board:
    rng = np.random.RandomState(seed)
    g = [[0] * BOARD_COLS for _ in range(BOARD_ROWS)]
    for r in range(BOARD_ROWS - fill_rows, BOARD_ROWS):
        for c in range(BOARD_COLS):
            g[r][c] = int(rng.choice(COLORS))
    return Board.from_list(g)


def _bench(board: Board, budget_sec: float, n_rollouts: int, reps: int = 3) -> float:
    """1 回あたりの所要時間 [ms] を返す。"""
    ts = []
    for i in range(reps):
        t0 = time.perf_counter()
        mc.estimate_counter_distribution(
            board, budget_sec, thresholds_ojama=(THRESHOLD,),
            n_rollouts=n_rollouts,
        )
        ts.append((time.perf_counter() - t0) * 1000.0)
    return float(np.median(ts))


def main() -> int:
    print(f"30fps の 1 フレーム予算 = {FRAME_BUDGET_MS:.1f} ms")
    print("(認識自体が既に 31fps ≒ 32ms を使っているため、余裕はほぼ無い)")
    print()
    board = _make_board(6)
    print(f"{'連鎖数':>6s} {'時間予算':>8s} {'本数':>6s} {'1回あたり':>10s} {'フレーム比':>10s}")
    print("-" * 46)
    for chain in (2, 5, 9, 13):
        budget = float(iv.estimate_chain_anim_duration_sec(float(chain)))
        for n in (10, 30, 60, 200):
            ms = _bench(board, budget, n)
            ratio = ms / FRAME_BUDGET_MS
            mark = "  ✕ 超過" if ratio > 1.0 else ("  △" if ratio > 0.3 else "  ○")
            print(f"{chain:6d} {budget:7.1f}s {n:6d} {ms:9.1f}ms {ratio:9.1f}x{mark}")
    print()
    # 盤面の高さによる差 (段別テーブルで時間消費が変わるため)
    print("盤面の高さ別 (連鎖9・本数60):")
    budget = float(iv.estimate_chain_anim_duration_sec(9.0))
    for rows in (3, 6, 9, 12):
        b = _make_board(rows, seed=rows)
        ms = _bench(b, budget, 60)
        print(f"  {rows:2d}段積み: {ms:7.1f}ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
