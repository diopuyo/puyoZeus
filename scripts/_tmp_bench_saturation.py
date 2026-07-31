"""saturation_chain の 1 盤面あたりコスト micro-bench (目標 300-500ms/盤面)。

data/indicators_v2/boards/v29.npz からランダムサンプルした盤面で計測する。
熱対策: 単プロセス・スレッド制限。

使い方:
    PYTHONPATH=. OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
        ./venv/bin/python -m scripts._tmp_bench_saturation
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.board import Board
from src.chain import ChainSimulator
import src.indicators_v2 as iv

BOARDS_NPZ = Path("data/indicators_v2/boards/v29.npz")
N_SAMPLE = 30
BEAM_WIDTHS = (3, 4, 6, 8)


def main() -> None:
    print("=== saturation_chain micro-bench ===")
    data = np.load(str(BOARDS_NPZ), allow_pickle=True)
    grids = data["grids"]
    rng = np.random.default_rng(1)
    n = min(N_SAMPLE, len(grids))
    idxs = rng.choice(len(grids), size=n, replace=False)
    boards = [Board.from_list(grids[i].tolist()) for i in idxs]
    boards = [b for b in boards if not b.is_dead()]
    print(f"サンプル盤面数: {len(boards)}")

    sim = ChainSimulator()

    for beam_width in BEAM_WIDTHS:
        times: list[float] = []
        raws: list[float] = []
        for board in boards:
            t0 = time.perf_counter()
            v = iv.saturation_chain(board, beam_width=beam_width, simulator=sim)
            times.append(time.perf_counter() - t0)
            raws.append(v.raw)
        times_arr = np.array(times)
        print(
            f"beam_width={beam_width}: "
            f"mean={times_arr.mean()*1000:.1f}ms "
            f"median={np.median(times_arr)*1000:.1f}ms "
            f"max={times_arr.max()*1000:.1f}ms "
            f"raw_mean={np.mean(raws):.2f} raw_max={np.max(raws):.0f}"
        )

    print("")
    print("=== 既定 (beam_width=6) vs build_ceiling_chain(depth=2) の raw 比較 ===")
    sat_raws = []
    ceil_raws = []
    t_sat = 0.0
    t_ceil = 0.0
    for board in boards:
        t0 = time.perf_counter()
        sat = iv.saturation_chain(board, simulator=sim)
        t_sat += time.perf_counter() - t0
        t0 = time.perf_counter()
        ceil = iv.build_ceiling_chain(board, simulator=sim)
        t_ceil += time.perf_counter() - t0
        sat_raws.append(sat.raw)
        ceil_raws.append(ceil.raw)
    sat_arr = np.array(sat_raws)
    ceil_arr = np.array(ceil_raws)
    print(f"saturation_chain: mean={sat_arr.mean():.2f} 平均時間={t_sat/len(boards)*1000:.1f}ms/盤面")
    print(f"build_ceiling_chain: mean={ceil_arr.mean():.2f} 平均時間={t_ceil/len(boards)*1000:.1f}ms/盤面")
    print(f"平均差 (saturation - ceiling): {(sat_arr - ceil_arr).mean():.2f}")
    print(f"差の分位点: {np.percentile(sat_arr - ceil_arr, [50, 75, 90, 95]).round(2)}")

    print("\n=== 完了 ===")


if __name__ == "__main__":
    main()
