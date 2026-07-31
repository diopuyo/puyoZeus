"""chain_bitboard.py の正当性検証 (既存 ChainSimulator との完全一致確認)。

使い方:
    PYTHONPATH=. OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
        ./venv/bin/python -m scripts._tmp_verify_chain_bitboard
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.board import Board
from src.chain import ChainSimulator
from src.chain_bitboard import simulate_single, board_to_planes, planes_to_board


def _boards_equal(a: Board, b: Board) -> bool:
    return bool(np.array_equal(a._grid, b._grid))


def main() -> None:
    print("=== chain_bitboard 正当性検証 ===")

    # まず round-trip 変換の確認 (board -> planes -> board)
    data = np.load("data/indicators_v2/boards/v29.npz", allow_pickle=True)
    grids = data["grids"]
    rng = np.random.default_rng(3)
    idxs = rng.choice(len(grids), size=40, replace=False)
    boards = [Board.from_list(grids[i].tolist()) for i in idxs]

    roundtrip_fail = 0
    for b in boards:
        planes = board_to_planes(b)
        back = planes_to_board(planes)
        if not _boards_equal(b, back):
            roundtrip_fail += 1
    print(f"round-trip 変換: {len(boards)}件中 失敗={roundtrip_fail}")

    sim = ChainSimulator()
    n_match_chain = 0
    n_match_erased = 0
    n_match_ojama = 0
    n_total = 0
    mismatches = []

    for i, b in enumerate(boards):
        if b.is_dead():
            continue
        n_total += 1
        expected = sim.simulate(b)
        got = simulate_single(b)

        ok_chain = expected.chain_count == got.chain_count
        ok_erased = expected.total_erased == got.total_erased
        ok_ojama = expected.total_ojama == got.total_ojama
        n_match_chain += int(ok_chain)
        n_match_erased += int(ok_erased)
        n_match_ojama += int(ok_ojama)

        if not (ok_chain and ok_erased and ok_ojama):
            mismatches.append((i, expected.chain_count, got.chain_count,
                                expected.total_erased, got.total_erased,
                                expected.total_ojama, got.total_ojama))

    print(f"n={n_total}")
    print(f"chain_count 一致={n_match_chain}/{n_total}")
    print(f"total_erased 一致={n_match_erased}/{n_total}")
    print(f"total_ojama 一致={n_match_ojama}/{n_total}")

    if mismatches:
        print(f"\n不一致 {len(mismatches)} 件 (先頭5件):")
        for m in mismatches[:5]:
            print(
                f"  盤面{m[0]}: chain expected={m[1]} got={m[2]} | "
                f"erased expected={m[3]} got={m[4]} | ojama expected={m[5]} got={m[6]}"
            )
    else:
        print("\n完全一致 (全件)")

    print("\n=== 完了 ===")


if __name__ == "__main__":
    main()
