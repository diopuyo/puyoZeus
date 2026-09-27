"""探索の手数補正がA/Bの応手評価に与える物理的影響を測る。"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.board import Board
from src.chain import ChainSimulator
from src.indicators_v2 import near_future_fire_power
from src.production_config import GHOST_CHAIN_RULE_ENABLED


def main() -> None:
    """保存した確定盤面で1手と従来K=1を比較する。"""
    sim = ChainSimulator(exclude_hidden_row_from_pop=GHOST_CHAIN_RULE_ENABLED)
    for line in Path("logs/e10c/physics.jsonl").read_text().splitlines():
        row = json.loads(line)
        i = 1 if 2690 < row["t"] < 2700 else 0
        if row["t"] < 2600:
            continue
        side = row["boards"][i]
        board = Board()
        board._grid = np.array(side["grid"], dtype=np.int8)
        resolved = sim.simulate(board).final_board
        first, second = side["queue"]
        result = near_future_fire_power(resolved, first, second, k_levels=(-1, 0, 1),
                                       resolve_before_death=True, beam_width=22)
        print(row["t"], {k+2: v.raw for k, v in result.values.items()}, flush=True)


if __name__ == "__main__":
    main()
