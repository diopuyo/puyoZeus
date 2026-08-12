"""puyo_core (Rust ネイティブ拡張) の性能ベンチ (2026-08-12 user確定指示)。

計測項目:
    (a) 1盤面シミュレーション (µs)
    (b) 深さ13/幅250、深さ16/幅250 の1探索実測ms (単スレッド / rayon並列)

実盤面 (data/indicators_v2/boards_lean_phase_l_2026-08-11) で計測する。
読み取り専用 (盤面データは変更しない)。

目標: 1探索100ms以内 (8スレッドで10〜20ms が理想)。
"""
from __future__ import annotations

import statistics
import sys
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.board import BOARD_COLS, BOARD_ROWS, COLOR_UNKNOWN, Board  # noqa: E402
from src.puyo_core_bridge import NATIVE_AVAILABLE  # noqa: E402

_DATA_DIR = _ROOT / "data" / "indicators_v2" / "boards_lean_phase_l_2026-08-11"
# ベンチ対象の探索深さ (task指示: 13〜16手)。
_BENCH_DEPTHS: tuple[int, ...] = (13, 16)
_BENCH_BEAM_WIDTH: int = 250
_N_SIM_TRIALS: int = 300
_N_SEARCH_TRIALS: int = 5
_RNG_SEED: int = 20260812
# ベンチ用ツモ色 (試合は4色のみ使用、reference_four_colors_per_match_2026-07-22 準拠)。
_BENCH_COLORS: tuple[int, ...] = (1, 2, 3, 4)


def _load_bench_board() -> "list[int]":
    """実盤面から中盤程度 (積みあり) の1盤面を選び flatten して返す。"""
    npz_files = sorted(_DATA_DIR.glob("*.npz"))
    rng = np.random.RandomState(_RNG_SEED)
    for path in rng.permutation(npz_files):
        data = np.load(path, allow_pickle=True)
        grids = data["grids"]
        for i in rng.permutation(len(grids)):
            grid = grids[i].astype(np.uint8)
            if np.any(grid == COLOR_UNKNOWN):
                continue
            n_puyo = int(np.sum((grid != 0) & (grid != COLOR_UNKNOWN)))
            # 中盤程度 (20〜50個) の盤面を選ぶ (空盤面や満杯盤面は代表性が低い)
            if 20 <= n_puyo <= 50:
                return grid.flatten().tolist()
    # 見つからなければ空盤面 (フォールバック)
    return [0] * (BOARD_ROWS * BOARD_COLS)


def _bench_single_sim(puyo_core, grid: "list[int]") -> float:
    """1盤面シミュレーション µs (median)。"""
    times = []
    for _ in range(_N_SIM_TRIALS):
        t0 = time.perf_counter()
        puyo_core.simulate_chain_py(grid, True)
        times.append((time.perf_counter() - t0) * 1e6)
    return statistics.median(times)


def _bench_search(
    puyo_core, grid: "list[int]", depth: int, num_threads: "int | None",
) -> float:
    """深さ depth / 幅 _BENCH_BEAM_WIDTH の1探索 ms (median)。"""
    rng = np.random.RandomState(_RNG_SEED + depth)
    times = []
    for _ in range(_N_SEARCH_TRIALS):
        pairs = [
            (int(rng.choice(_BENCH_COLORS)), int(rng.choice(_BENCH_COLORS)))
            for _ in range(depth)
        ]
        t0 = time.perf_counter()
        puyo_core.beam_search_py(grid, pairs, _BENCH_BEAM_WIDTH, True, num_threads)
        times.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(times)


def main() -> int:
    if not NATIVE_AVAILABLE:
        print("puyo_core ネイティブ拡張が未ビルドです (maturin develop を実行してください)")
        return 1
    import puyo_core

    grid = _load_bench_board()
    n_puyo = sum(1 for v in grid if v not in (0, COLOR_UNKNOWN))
    print(f"ベンチ対象盤面: puyo数={n_puyo}")
    try:
        import os
        load1, _load5, _load15 = os.getloadavg()
        print(f"⚠️ システム負荷 (loadavg 1分): {load1:.1f} "
              f"(他プロセスと競合すると並列計測値が悪化する、注意)")
    except (AttributeError, OSError):
        pass
    print()

    us = _bench_single_sim(puyo_core, grid)
    print(f"(a) 1盤面シミュレーション: {us:.2f} µs (median, n={_N_SIM_TRIALS})")
    print()

    print(f"(b) ビームサーチ (幅={_BENCH_BEAM_WIDTH}):")
    print(f"{'深さ':>6s} {'単スレッド(ms)':>16s} {'8スレッド並列(ms)':>18s}")
    for depth in _BENCH_DEPTHS:
        single_ms = _bench_search(puyo_core, grid, depth, num_threads=None)
        parallel_ms = _bench_search(puyo_core, grid, depth, num_threads=8)
        print(f"{depth:6d} {single_ms:16.2f} {parallel_ms:18.2f}")

    print()
    print("目標: 1探索100ms以内 (8スレッドで10〜20msが理想)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
