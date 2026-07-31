"""saturation_chain の構築ビームが何ステップで停止しているか診断する。"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.board import Board
from src.chain import ChainSimulator
import src.indicators_v2 as iv

BOARDS_NPZ = Path("data/indicators_v2/boards/v29.npz")


def main() -> None:
    data = np.load(str(BOARDS_NPZ), allow_pickle=True)
    grids = data["grids"]
    rng = np.random.default_rng(1)
    idxs = rng.choice(len(grids), size=10, replace=False)
    sim = ChainSimulator()

    for i in idxs:
        board = Board.from_list(grids[i].tolist())
        if board.is_dead():
            continue
        start_count = board.count_puyos()
        target = round(iv.SATURATION_FILL_RATIO_DEFAULT * iv.FULL_BOARD_CAP)
        steps_requested = min(max(0, target - start_count), iv.SATURATION_MAX_BUILD_STEPS)

        frontier = [board]
        actual_steps = 0
        for _ in range(steps_requested):
            nf = iv._sat_expand_step(frontier, iv.SATURATION_BEAM_WIDTH_DEFAULT)
            if not nf:
                break
            frontier = nf
            actual_steps += 1

        final_count = frontier[0].count_puyos()
        best_chain = iv._sat_measure_terminal_chain(frontier, sim)
        print(
            f"board#{i}: start_count={start_count} target={target} "
            f"steps_requested={steps_requested} actual_steps={actual_steps} "
            f"final_count={final_count} best_chain={best_chain}"
        )


if __name__ == "__main__":
    main()
