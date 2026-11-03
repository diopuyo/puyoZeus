"""表示加点と攻撃クレジットを分離する因果的会計。学習・prefire共通。"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any

BONUS_POINTS = 2100
BONUS_OJAMA = 30
DROP_SLACK = 26

@dataclass
class AllClearCredit:
    """物理証拠は到着済みの発火前盤だけ。終端通知は新しい連鎖にしない。"""
    current: dict | None = None
    pending: list[float] = field(default_factory=list)
    awarded: set[float] = field(default_factory=set)
    released: dict[float, int] = field(default_factory=dict)
    recognized_chain_ids: set[int] = field(default_factory=set)
    uncertain_points: int = 0

    def observe(self, info: dict | None, state: str | None, adopted: int | None,
                stamp: float, native: Any) -> None:
        if not info or state != 'CHAIN':
            return
        trigger = float(info['trigger_sec'])
        if trigger > stamp:
            raise ValueError('全消し証拠に未来の発火時刻')
        if self.current is None or trigger != self.current['trigger']:
            board = info.get('before_board')
            board = board.get('board', {}).get('array') if isinstance(board, dict) else board
            flat = [int(c) for row in board for c in row] if board else []
            valid = len(flat) == 78 and all(c in (0, 1, 2, 3, 4, 5, 9) for c in flat)
            trace = native(flat, True) if valid else []
            self.current = dict(trigger=trigger, baseline=adopted, scores=[s[4] for s in trace],
                                clear=bool(trace and not any(trace[-1][1])), reported=0, count=0)
        self.current.update(reported=int(info.get('total_score', 0)), count=int(info.get('chain_count', 0)))

    def split(self, adopted: int, delta: int) -> tuple[int, str | None]:
        """物理素点・観測段数・表示差を照合。2100という値だけでは認定しない。"""
        cur = self.current
        if cur is None or cur['baseline'] is None or cur['trigger'] in self.awarded:
            return delta, None
        scores = cur['scores']
        expected = sum(scores)
        residual = adopted-cur['baseline']-expected
        if not (cur['clear'] and cur['count'] == len(scores) and cur['reported'] == expected
                and BONUS_POINTS <= residual <= BONUS_POINTS+DROP_SLACK and delta >= BONUS_POINTS):
            return delta, None
        self.pending.append(cur['trigger'])
        self.awarded.add(cur['trigger'])
        return delta-BONUS_POINTS, 'physical_and_score'

    def release(self) -> int:
        """別の実CHAINの最初の攻撃段へ一度だけ固定30個を解放する。"""
        if self.current is None:
            return 0
        trigger = self.current['trigger']
        older = [key for key in self.pending if key < trigger]
        self.pending = [key for key in self.pending if key >= trigger]
        count = BONUS_OJAMA*len(older)
        if count:
            self.released[trigger] = self.released.get(trigger, 0)+count
        return count

    def forecast(self, trigger: float) -> int:
        return self.released.get(trigger, 0)+BONUS_OJAMA*sum(key < trigger for key in self.pending)
