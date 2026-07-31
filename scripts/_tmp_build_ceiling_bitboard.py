"""build_ceiling_chain の bitboard バッチ深化版プロトタイプ。

コーディネータ方針 (2026-07-22): 埋め切りビルダーは断念確定。
build_ceiling_chain (骨格を壊さない running max 方式) を chain_bitboard で
深化 (depth 3-4+) し「より天井に近い値」を作る方向で検証する。

既存 `src/indicators_v2.py` の `build_ceiling_chain` / `_build_ceiling_expand`
(1手= 単ぷよ1個・5色×6列=30通り、beam_widthで頭打ちを剪定するビームサーチ、
running max で深さを跨いだ最大値を保持) の探索構造はそのまま流用し、
`ChainSimulator.simulate` の1候補ずつループを `chain_bitboard.simulate_batch`
のバッチ判定に置き換えるのみ。既存 build_ceiling_chain / ChainSimulator は
一切変更しない (本スクリプトは独立プロトタイプ)。

使い方:
    PYTHONPATH=. OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
        ./venv/bin/python -m scripts._tmp_build_ceiling_bitboard
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

from src.board import BOARD_COLS, Board  # noqa: E402
from src.chain import ChainSimulator  # noqa: E402
from src.chain_bitboard import batch_from_boards, simulate_batch  # noqa: E402
import src.indicators_v2 as iv  # noqa: E402

IGNITION_TRIAL_COLORS = iv.IGNITION_TRIAL_COLORS  # 5色 (RED,BLUE,GREEN,YELLOW,PURPLE)
_drop_one_color = iv._drop_one_color             # 既存ヘルパーをそのまま流用


def _expand_batched(
    frontier_boards: "list[Board]", beam_width: int,
) -> "list[tuple[int, Board]]":
    """既存 `_build_ceiling_expand` のバッチ判定版 (chain_bitboard使用)。

    frontier の各盤面から5色×6列=30通りの単ぷよ1個落としを生成し、
    まとめて1回 `simulate_batch` する。chain_count 降順で上位 beam_width 件を返す。
    """
    variant_boards: "list[Board]" = []
    for base_board in frontier_boards:
        for col in range(BOARD_COLS):
            for color in IGNITION_TRIAL_COLORS:
                dropped = _drop_one_color(base_board, col, color)
                if dropped is not None:
                    variant_boards.append(dropped)
    if not variant_boards:
        return []
    batch = batch_from_boards(variant_boards)
    results = simulate_batch(batch)
    scored = [(res.chain_count, b) for b, res in zip(variant_boards, results)]
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:beam_width]


def build_ceiling_chain_deep(
    board: Board,
    depth: int = 4,
    beam_width: int = 8,
) -> "tuple[float, float]":
    """build_ceiling_chain のバッチ深化版 (プロトタイプ、既存関数は変更しない)。

    Returns:
        (raw_best_chain, elapsed_sec)
    """
    if board.is_dead():
        return 0.0, 0.0
    t0 = time.perf_counter()
    best_chain = 0
    frontier: "list[Board]" = [board]
    for _ in range(depth):
        expanded = _expand_batched(frontier, beam_width)
        if not expanded:
            break
        best_chain = max(best_chain, expanded[0][0])
        frontier = [b for _, b in expanded]
    elapsed = time.perf_counter() - t0
    return float(best_chain), elapsed


# ============================
# ベンチ本体
# ============================

BOARDS_NPZ = Path("data/indicators_v2/boards/v29.npz")


def main() -> None:
    print("=== build_ceiling_chain bitboard深化版 検証 ===")
    data = np.load(str(BOARDS_NPZ), allow_pickle=True)
    grids = data["grids"]
    rng = np.random.default_rng(1)
    n = min(20, len(grids))
    idxs = rng.choice(len(grids), size=n, replace=False)
    boards = [Board.from_list(grids[i].tolist()) for i in idxs]
    boards = [b for b in boards if not b.is_dead()]
    sim = ChainSimulator()

    print("\n### (A) 深さ×ビーム幅 別の値カーブ・速度 (難所盤面 current_max_chain=11) ###")
    hard_board = boards[1]
    print(f"current_max_chain={iv.current_max_chain(hard_board, sim).raw:.0f}")
    for beam_width in (8, 20, 50):
        print(f"--- beam_width={beam_width} ---")
        for depth in (1, 2, 3, 4, 5, 6):
            raw, elapsed = build_ceiling_chain_deep(hard_board, depth=depth, beam_width=beam_width)
            print(f"  depth={depth}: raw={raw:.0f} time={elapsed*1000:.1f}ms")

    print("\n### (B) 複数盤面での値カーブ (beam_width=8既定値) ###")
    sample = boards[:8]
    currents = [iv.current_max_chain(b, sim).raw for b in sample]
    print("current_max_chain:", [f"{c:.0f}" for c in currents])
    for depth in (1, 2, 3, 4, 5, 6, 8):
        raws = []
        times = []
        for b in sample:
            raw, elapsed = build_ceiling_chain_deep(b, depth=depth, beam_width=8)
            raws.append(raw)
            times.append(elapsed)
        times_arr = np.array(times)
        print(
            f"depth={depth}: raw={[f'{r:.0f}' for r in raws]} "
            f"time_mean={times_arr.mean()*1000:.1f}ms time_max={times_arr.max()*1000:.1f}ms"
        )

    print("\n=== 完了 ===")


if __name__ == "__main__":
    main()
