"""chain_bitboard.py の速度ベンチ (単発呼び出し・バッチ呼び出し両方)。

`scripts/_tmp_bench_saturation.py` と同枠組みで実測する。

使い方:
    PYTHONPATH=. OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
        ./venv/bin/python -m scripts._tmp_bench_chain_bitboard
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

from src.board import Board  # noqa: E402
from src.chain import ChainSimulator  # noqa: E402
from src.chain_bitboard import simulate_single, batch_from_boards, simulate_batch  # noqa: E402

BOARDS_NPZ = Path("data/indicators_v2/boards/v29.npz")


def main() -> None:
    print("=== chain_bitboard 速度ベンチ ===")
    data = np.load(str(BOARDS_NPZ), allow_pickle=True)
    grids = data["grids"]
    rng = np.random.default_rng(1)
    n = min(30, len(grids))
    idxs = rng.choice(len(grids), size=n, replace=False)
    boards = [Board.from_list(grids[i].tolist()) for i in idxs]
    boards = [b for b in boards if not b.is_dead()]
    print(f"サンプル盤面数: {len(boards)}")

    sim = ChainSimulator(cache_enabled=False)

    # (a) 既存 ChainSimulator: 1盤面ずつ
    times_old = []
    for b in boards:
        t0 = time.perf_counter()
        sim.simulate(b)
        times_old.append(time.perf_counter() - t0)
    times_old = np.array(times_old)
    print(
        f"[既存ChainSimulator] 1盤面ずつ: mean={times_old.mean()*1000:.3f}ms "
        f"median={np.median(times_old)*1000:.3f}ms"
    )

    # (b) chain_bitboard: 1盤面ずつ (simulate_single、バッチ化なし)
    times_bb_single = []
    for b in boards:
        t0 = time.perf_counter()
        simulate_single(b)
        times_bb_single.append(time.perf_counter() - t0)
    times_bb_single = np.array(times_bb_single)
    print(
        f"[chain_bitboard] 1盤面ずつ(simulate_single): mean={times_bb_single.mean()*1000:.3f}ms "
        f"median={np.median(times_bb_single)*1000:.3f}ms"
    )

    # (c) chain_bitboard: バッチ化 (N=len(boards) を一括)
    t0 = time.perf_counter()
    batch = batch_from_boards(boards)
    results = simulate_batch(batch)
    t_batch = time.perf_counter() - t0
    print(
        f"[chain_bitboard] バッチ一括(N={len(boards)}): 合計={t_batch*1000:.2f}ms "
        f"1盤面あたり={t_batch/len(boards)*1000:.3f}ms"
    )

    # (d) バッチサイズを変えた場合のスケーリング (N=50,100,300)
    print("\n--- バッチサイズ別スケーリング ---")
    for target_n in (50, 100, 300, 550):
        # boards をリピートして目標件数に近づける (実データが足りない分は複製)
        rep = (target_n + len(boards) - 1) // len(boards)
        big_boards = (boards * rep)[:target_n]
        t0 = time.perf_counter()
        big_batch = batch_from_boards(big_boards)
        simulate_batch(big_batch)
        t_big = time.perf_counter() - t0
        print(
            f"N={target_n}: 合計={t_big*1000:.1f}ms 1盤面あたり={t_big/target_n*1000:.4f}ms"
        )

    print("\n=== 完了 ===")


if __name__ == "__main__":
    main()
