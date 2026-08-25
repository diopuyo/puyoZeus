"""S1b (2閾値1回化) の速度実測ベンチ (2026-08-21、一時スクリプト)。

`counter_reach_probability` の `thresholds_ojama` 引数追加により、
vs_opp_fire / vs_forecast の2閾値を1回のシミュレーションでまとめて計算
できるようになった (シミュレーション自体は閾値に無依存、hits 比較のみ
閾値依存)。本スクリプトは 40 件の盤面で「2回個別呼び」と「1回まとめ呼び」
の壁時間を比較し、コスト半減を実測する (cProfile 禁止、perf_counter のみ
使用、CLAUDE.md 準拠)。

使い方: PYTHONPATH=. python scripts/_bench_counter_reach_multi_threshold_2026-08-21.py
"""
from __future__ import annotations

import time

import numpy as np

from src.board import Board
from src.indicators_v2 import counter_reach_probability

N_BOARDS: int = 40  # user 指定の実測件数
THRESHOLD_A: float = 6.0  # vs_opp_fire 相当のダミー閾値
THRESHOLD_B: float = 18.0  # vs_forecast 相当のダミー閾値
RNG_SEED_FOR_SIM: int = 1  # 乱数系列を固定して比較を公平にする


def _build_boards(n: int) -> list[Board]:
    """再現可能な乱数で n 件のランダム盤面を作る (窒息しない範囲)。"""
    rng = np.random.default_rng(0)
    boards: list[Board] = []
    for i in range(n):
        board = Board()
        n_puyo = 6 + int(rng.integers(0, 6))
        cols = rng.integers(0, 6, size=n_puyo)
        for j, col in enumerate(cols):
            color = 1 + int((i + j) % 4)
            row = 12 - int(j % 3)
            board.set(row, int(col), color)
        boards.append(board)
    return boards


def _measure_individual(boards: list[Board]) -> float:
    """2閾値を個別に2回呼ぶ場合の壁時間。"""
    t0 = time.perf_counter()
    for board in boards:
        counter_reach_probability(board, THRESHOLD_A, rng_seed=RNG_SEED_FOR_SIM)
        counter_reach_probability(board, THRESHOLD_B, rng_seed=RNG_SEED_FOR_SIM)
    return time.perf_counter() - t0


def _measure_combined(boards: list[Board]) -> float:
    """1回のまとめ呼びで2閾値を得る場合の壁時間。"""
    t0 = time.perf_counter()
    for board in boards:
        counter_reach_probability(
            board, THRESHOLD_A, rng_seed=RNG_SEED_FOR_SIM,
            thresholds_ojama=(THRESHOLD_B,),
        )
    return time.perf_counter() - t0


def main() -> None:
    boards = _build_boards(N_BOARDS)
    t_individual = _measure_individual(boards)
    t_combined = _measure_combined(boards)
    print(f"件数: {N_BOARDS}")
    print(f"個別2回呼び: {t_individual:.3f}s ({t_individual / N_BOARDS * 1000:.1f}ms/件)")
    print(f"まとめ1回呼び: {t_combined:.3f}s ({t_combined / N_BOARDS * 1000:.1f}ms/件)")
    print(f"比率 (まとめ/個別): {t_combined / t_individual:.3f}")


if __name__ == "__main__":
    main()
