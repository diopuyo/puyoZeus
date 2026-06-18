"""III-3 到達火力 (reach_fire_power) のプロファイル計測スクリプト。

STABLEスナップショット相当の盤面に対して:
  - early pruning 有 (best_k=5): 22 + 5×22 = 132 sim/snapshot
  - early pruning 無 (best_k=22): 22 + 22×22 = 506 sim/snapshot
の ms/snapshot を計測し、STABLE 0.5s 間隔での常時計算が許容かを判定する。

使い方:
    PYTHONPATH=. python scripts/_profile_reach_fire_power.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.board import (
    BOARD_COLS,
    BOARD_ROWS,
    COLOR_BLUE,
    COLOR_GREEN,
    COLOR_RED,
    COLOR_YELLOW,
    COLOR_PURPLE,
    Board,
)
from src.chain import ChainSimulator
import src.indicators_v2 as iv

# 盤面定義 (典型的な中盤のリアル盤面を模した合成盤面)
BOARDS: list[tuple[str, Board, tuple[int, int], tuple[int, int]]] = []


def _make_midgame_board() -> Board:
    """中盤相当の複雑な盤面 (高さ 8 程度、多色混合)。"""
    # 6列13行の盤面 (row 12 = 最下段)
    grid: list[list[int]] = [[0] * BOARD_COLS for _ in range(BOARD_ROWS)]
    # col 0: 赤4, 青2 (高さ 6)
    for r in [12, 11, 10, 9]:
        grid[r][0] = COLOR_RED
    grid[8][0] = COLOR_BLUE
    grid[7][0] = COLOR_BLUE
    # col 1: 緑3, 黄2, 赤1 (高さ 6)
    for r in [12, 11, 10]:
        grid[r][1] = COLOR_GREEN
    grid[9][1] = COLOR_YELLOW
    grid[8][1] = COLOR_YELLOW
    grid[7][1] = COLOR_RED
    # col 2: 青3, 紫3 (高さ 6)
    for r in [12, 11, 10]:
        grid[r][2] = COLOR_BLUE
    for r in [9, 8, 7]:
        grid[r][2] = COLOR_PURPLE
    # col 3: 赤2, 緑4 (高さ 6)
    grid[12][3] = COLOR_RED
    grid[11][3] = COLOR_RED
    for r in [10, 9, 8, 7]:
        grid[r][3] = COLOR_GREEN
    # col 4: 黄3, 紫2 (高さ 5)
    for r in [12, 11, 10]:
        grid[r][4] = COLOR_YELLOW
    grid[9][4] = COLOR_PURPLE
    grid[8][4] = COLOR_PURPLE
    # col 5: 青2, 赤3 (高さ 5)
    grid[12][5] = COLOR_BLUE
    grid[11][5] = COLOR_BLUE
    for r in [10, 9, 8]:
        grid[r][5] = COLOR_RED
    return Board.from_list(grid)


def _make_simple_two_chain_board() -> Board:
    """シンプルな 2 連鎖盤面。"""
    grid: list[list[int]] = [[0] * BOARD_COLS for _ in range(BOARD_ROWS)]
    grid[12][0] = COLOR_RED
    grid[12][1] = COLOR_RED
    grid[11][0] = COLOR_RED
    grid[10][0] = COLOR_RED
    grid[12][2] = COLOR_BLUE
    grid[11][1] = COLOR_BLUE
    grid[10][1] = COLOR_BLUE
    grid[9][0] = COLOR_BLUE
    return Board.from_list(grid)


TEST_CASES = [
    ("midgame", _make_midgame_board(), (COLOR_RED, COLOR_BLUE), (COLOR_GREEN, COLOR_YELLOW)),
    ("two_chain", _make_simple_two_chain_board(), (COLOR_RED, COLOR_BLUE), (COLOR_PURPLE, COLOR_GREEN)),
    ("empty", Board(), (COLOR_RED, COLOR_BLUE), (COLOR_GREEN, COLOR_YELLOW)),
]

N_WARMUP = 5
N_ITER = 50


def profile_case(
    label: str,
    board: Board,
    next_pair: tuple[int, int],
    dnext_pair: tuple[int, int],
    best_k: int,
    sim: ChainSimulator,
) -> float:
    """N_ITER 回実行して平均 ms を返す。"""
    # warmup (キャッシュ効果あり → 実運用に近い)
    for _ in range(N_WARMUP):
        iv.reach_fire_power(board, next_pair, dnext_pair, best_k=best_k, simulator=sim)
    t0 = time.perf_counter()
    for _ in range(N_ITER):
        iv.reach_fire_power(board, next_pair, dnext_pair, best_k=best_k, simulator=sim)
    elapsed = time.perf_counter() - t0
    return elapsed / N_ITER * 1000.0


def main() -> None:
    sim = ChainSimulator(cache_enabled=True)
    print(f"=== reach_fire_power プロファイル (N_ITER={N_ITER}) ===")
    print(f"{'case':<15} {'best_k':<8} {'ms/snap':>10}  {'判定'}")
    print("-" * 50)

    THRESHOLD_MS = 500.0  # STABLE 0.5s 間隔の上限 (ms)

    for label, board, next_pair, dnext_pair in TEST_CASES:
        for best_k in [5, 22]:  # 有 / 無 pruning 相当
            ms = profile_case(label, board, next_pair, dnext_pair, best_k, sim)
            ok = "OK" if ms < THRESHOLD_MS else "OVER"
            print(f"{label:<15} {best_k:<8} {ms:>10.2f}  {ok}")

    print()
    print(f"閾値: {THRESHOLD_MS:.0f} ms (STABLE 0.5s 間隔の全時間)")
    print("※ キャッシュ有効時。初回(キャッシュ冷え)は数倍の可能性あり。")


if __name__ == "__main__":
    main()
