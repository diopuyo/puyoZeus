"""攻撃側盤面に依存する着弾を保持した、5Cの厳密な応手遷移表。"""
from __future__ import annotations

from dataclasses import replace
from functools import lru_cache

import numpy as np

from src import prefire_v5_search as base
from src import prefire_v5b_search as exact
from src.board import BOARD_COLS, DEATH_COL, DEATH_ROW, COLOR_UNKNOWN
from src.scoring import OJAMA_MAX_DROP_PER_TURN

TABLE_SIZE = 32768
SIDE_TABLE_SIZE = 128


@lru_cache(maxsize=SIDE_TABLE_SIZE)
def side_options(state: base.Position, depth: int | None = None, side: int = 0) -> tuple:
    """片側の全候補を一度だけ重複統合する。相手側が変わっても再利用できる。"""
    return exact.unique(exact.candidates(state, depth, side=side))


def dead(raw: bytes) -> bool:
    """既存窒息セルとUNKNOWN除外を、配列の再生成なしに読む。"""
    return raw[DEATH_ROW * BOARD_COLS + DEATH_COL] not in (0, COLOR_UNKNOWN)


def landing(raw: bytes, opponent: bytes, dropped: int) -> bytes:
    """端数配置は必ず既存の両盤面シードを使う。"""
    if not dropped:
        return raw
    board, _, _ = base.land_pending_ojama_onto_board(
        base.sim._board(raw), base.sim._board(opponent or raw), dropped)
    return board._grid.astype(np.int8).tobytes()


class TransitionTable:
    """量ごとの相殺と片側着弾を遅延作成する。不要な全直積は確保しない。"""

    def __init__(self) -> None:
        self.land = lru_cache(maxsize=TABLE_SIZE)(landing)
        self.account = lru_cache(maxsize=TABLE_SIZE)(self._account)
        self.sends: dict[tuple, int] = {}

    def generated(self, score: int, elapsed: float) -> int:
        """マージン境界を含む換算率で得点変換を再利用する。"""
        key = (score, base.sim.effective_rate(elapsed))
        if key not in self.sends:
            if len(self.sends) >= TABLE_SIZE:
                self.sends.clear()
            self.sends[key] = int(base.sim.send_ojama(score, elapsed))
        return self.sends[key]

    @staticmethod
    def _account(pending: tuple, generated: tuple, attacker: int) -> tuple:
        """自己相殺→相手へ送付の順序を既存会計関数へ委譲する。"""
        remaining, cancelled = list(pending), [0, 0]
        for side in (attacker, 1-attacker):
            cancelled[side] = min(generated[side], remaining[side])
            remaining[side], remaining[1-side] = base.cancel_own_pending_then_send_surplus(
                generated[side], remaining[side], remaining[1-side])
        return tuple(remaining), tuple(cancelled)

    def resolve(self, first: base.Position, second: base.Position,
                attacker: int, elapsed: float) -> base.Exchange:
        """手順などの監査情報は現候補から戻し、数値と盤面だけ表引きする。"""
        sides = (first, second)
        generated = tuple(self.generated(p.score, elapsed) for p in sides)
        pending, cancelled = self.account(tuple(p.pending for p in sides), generated, attacker)
        dropped = tuple(min(p, OJAMA_MAX_DROP_PER_TURN) for p in pending)
        # 端数なしでは乱数が配置へ影響しない。端数ありだけ相手盤面をキーに含む。
        boards = tuple(self.land(p.board, sides[1-i].board if dropped[i] % BOARD_COLS else b'',
                                 dropped[i]) for i, p in enumerate(sides))
        result = tuple(replace(p, board=boards[i], pending=pending[i]-dropped[i])
                       for i, p in enumerate(sides))
        return base.Exchange(result, cancelled, dropped, tuple(dead(raw) for raw in boards))
