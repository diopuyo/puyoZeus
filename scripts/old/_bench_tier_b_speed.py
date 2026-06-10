"""Tier B 指標高速化ベンチ. 100 frame で indicator 全計算の所要時間を計測."""
from __future__ import annotations

import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np

from src.board import BOARD_COLS, BOARD_ROWS, COLOR_EMPTY, COLOR_RED, Board
from src.chain import ChainSimulator
from src.old.indicators import (
    BaseFlatnessIndicator,
    IndicatorCalculator,
    PlanningEntropyIndicator,
    StructureSolidityIndicator,
    _CC_CACHE,
)


def make_random_boards(n: int, seed: int = 0) -> list[Board]:
    """ランダムだが現実的な高さ分布の盤面を作成."""
    rng = np.random.default_rng(seed)
    boards: list[Board] = []
    for _ in range(n):
        grid = [[COLOR_EMPTY] * BOARD_COLS for _ in range(BOARD_ROWS)]
        # 各列ランダムな高さ (0..10)
        for c in range(BOARD_COLS):
            h = int(rng.integers(0, 10))
            for r in range(BOARD_ROWS - h, BOARD_ROWS):
                grid[r][c] = int(rng.integers(1, 6))
        boards.append(Board.from_list(grid))
    return boards


def bench_indicator_calculator(boards: list[Board]) -> float:
    """IndicatorCalculator.compute_all (Tier B 含む全指標) のスループット計測."""
    calc = IndicatorCalculator()
    t0 = time.perf_counter()
    for b in boards:
        calc.compute_all(b)
    return time.perf_counter() - t0


def bench_tier_b_only(boards: list[Board]) -> dict[str, float]:
    """Tier B 3 指標のみのスループット."""
    sim = ChainSimulator()
    pe = PlanningEntropyIndicator()
    ss = StructureSolidityIndicator()
    bf = BaseFlatnessIndicator()
    out: dict[str, float] = {}
    for name, ind in [
        ("planning_entropy", pe),
        ("structure_solidity", ss),
        ("base_flatness", bf),
    ]:
        t0 = time.perf_counter()
        for b in boards:
            ind.compute(b, simulator=sim)
        out[name] = time.perf_counter() - t0
    return out


def main() -> int:
    n = 100
    boards = make_random_boards(n)
    print(f"[bench] {n} ランダム盤面で計測")

    # Tier B 単体
    _CC_CACHE.clear()
    t_tier_b = bench_tier_b_only(boards)
    print("Tier B 指標単体時間 (ms/call):")
    for name, t in t_tier_b.items():
        print(f"  {name}: {t * 1000 / n:.3f} ms/call")

    # Calculator (Tier B 含む全指標)
    _CC_CACHE.clear()
    t_calc = bench_indicator_calculator(boards)
    print(f"IndicatorCalculator.compute_all: {t_calc * 1000 / n:.3f} ms/call")
    print(f"  → 75 動画 × 試合数 × 5 phase × 2P 想定で:")
    print(f"     5,000 サンプルなら {t_calc * 50:.1f} 秒")
    print(f"     50,000 サンプルなら {t_calc * 500:.1f} 秒 (≒ {t_calc * 500 / 60:.1f} 分)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
