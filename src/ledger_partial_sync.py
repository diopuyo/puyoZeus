"""履歴不足時の受け予告の品質。アイコンは証拠に使用しない。"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.scoring import OJAMA_MAX_DROP_PER_TURN

QUIET_STATES = ('STABLE', 'TSUMO_FALL')
MAX_DROP_ROWS = OJAMA_MAX_DROP_PER_TURN // 6


def has_landing_space(board: list | None) -> bool:
    """各列に最大着弾分の空きがある時だけ、頭打ちなしと判断する。"""
    return bool(board and len(board) >= MAX_DROP_ROWS
                and all(cell == 0 for row in board[:MAX_DROP_ROWS] for cell in row))


@dataclass
class PartialSync:
    """30個未満の着弾が完了した両側だけ、受け予告ゼロを確認する。"""

    clear: list[bool] = field(default_factory=lambda: [False, False])
    falling: list[bool] = field(default_factory=lambda: [False, False])
    baseline: list[int | None] = field(default_factory=lambda: [None, None])

    def observe(self, obs: Any, states: list[Any], events: list[Any]) -> bool:
        """[推測] 上限未満の着弾と十分な空きから、その受け側の残りをゼロとする。"""
        busy = any(s.chain is not None or s.queue or o.formula_step is not None
                   or o.state == 'CHAIN' for s, o in zip(states, obs.sides))
        if busy:
            self.clear, self.falling, self.baseline = [False, False], [False, False], [None, None]
            return False
        for i, side in enumerate(obs.sides):
            if side.state == 'OJAMA_FALL' and not self.falling[i]:
                self.falling[i] = self.baseline[i] is not None
            for event in events:
                if event.kind != 'observed_landing' or event.side != i:
                    continue
                amount = event.get('board_increase', 0)
                if (self.falling[i] and 0 < amount < OJAMA_MAX_DROP_PER_TURN
                        and side.landing_space and not side.dead):
                    self.clear[i] = True
                self.falling[i] = False
            if side.state in QUIET_STATES and not self.falling[i]:
                self.baseline[i] = side.ojama_cells
        return all(self.clear) and all(s.state in QUIET_STATES for s in obs.sides)
