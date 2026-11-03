"""既存死亡証明用の個数による十分条件。将来の消去を無視した総量断定はしない。"""
from __future__ import annotations

from collections import Counter
import math
import numpy as np
from src.board import Board, COLOR_UNKNOWN, BOARD_ROWS, BOARD_COLS
from src.post_counter_geometry import usable_colors, all_landings_dead, COLORS
from src.scoring import OJAMA_MAX_DROP_PER_TURN, compute_effective_rate

PAIR_SIZE = 2
DROP_SCORE_UPPER = BOARD_ROWS * PAIR_SIZE


def capacity_proof(board: Board, queue: tuple, incoming: int, hands: int,
                   elapsed: float, credit: int = 0) -> dict | None:
    """次の一手でも消去不能なら、最大相殺後の最初の着弾だけで窒息を証明する。"""
    if hands != 1 or np.any(board._grid == COLOR_UNKNOWN) or board.is_dead():
        return None
    pair = queue[:PAIR_SIZE]
    supply = Counter(pair) if len(pair) == PAIR_SIZE and all(c in COLORS for c in pair) else Counter(
        {c: PAIR_SIZE for c in COLORS})
    if usable_colors(board, supply):
        return None
    cancel = math.ceil(DROP_SCORE_UPPER/compute_effective_rate(elapsed))+credit
    remaining = max(0, incoming-cancel)
    dropped = min(remaining, OJAMA_MAX_DROP_PER_TURN)
    # 総空きだけでは凹凸列の溢れを扱えないため、既存の全端数窒息条件も確認する。
    if not all_landings_dead(board, dropped):
        return None
    return dict(dead=True, reason='capacity_all_landings_dead', nodes=0,
        rounds=[], incoming=incoming, cancel_upper=cancel, drop=dropped,
        free_cells=int(np.count_nonzero(board._grid == 0)), remaining=remaining-dropped)
