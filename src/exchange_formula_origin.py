"""A2: 操作を挟む連鎖境界で、式セッションの段・得点原点を分離する。"""
from __future__ import annotations

from copy import copy
from dataclasses import is_dataclass, replace
from typing import Any

from src.board_state_machine import BoardState
from src.chain_detector import CHAIN_MECHANISM_FORMULA_READ
from src.ojama_accounting import CHAIN_TOTAL_MIN_SCORE


def replaced(value: Any, **changes: Any) -> Any:
    """保存入力と認識器のイベントを変更せず、浅い複製へ反映する。"""
    if is_dataclass(value):
        return replace(value, **changes)
    result = copy(value)
    for key, item in changes.items():
        setattr(result, key, item)
    return result


class FormulaOrigin:
    """式の一致を探索せず、STABLE中の操作加点から原点を確定する。"""

    def __init__(self) -> None:
        self.last: Any = None
        self.offset = (0, 0)
        self.stable_score: float | None = None
        self.boundary: dict | None = None
        self.audit: list[dict] = []

    def observe(self, side: Any, formula: float | None, display: float | None,
                visible: bool, stamp: float) -> tuple[Any, float | None]:
        """新しい実読段が来るまで、観測済みの操作境界を保持する。"""
        event = side.chain_event
        if (side.state == BoardState.STABLE and side.confirmed_board is not None
                and not visible and display is not None):
            self._stable(display, stamp)
        if event is None or event.mechanism != CHAIN_MECHANISM_FORMULA_READ:
            return side, self._total(formula)
        if self.last is not None and (event.chain_count < self.last.chain_count
                                     or event.total_score < self.last.total_score):
            self.offset = (0, 0)
            self.boundary = None
        if self.last is not None and self.boundary is not None:
            if event.trigger_sec > self.boundary['t_sec']:
                self.offset = (0, 0)
                if (event.chain_count == self.last.chain_count + 1
                        and event.total_score > self.last.total_score):
                    self.offset = (self.last.chain_count, self.last.total_score)
                    self.audit.append(dict(**self.boundary, trigger_sec=event.trigger_sec,
                        count_offset=self.offset[0], score_offset=self.offset[1]))
                # 新発火が既に段1へリセット済みなら、次段で再補正してはならない。
                self.boundary = None
        if visible:
            self.stable_score = None
        self.last = event
        count, score = self.offset
        corrected = replaced(event, chain_count=event.chain_count-count,
                             total_score=event.total_score-score)
        return replaced(side, chain_event=corrected), self._total(formula)

    def _stable(self, display: float, stamp: float) -> None:
        """連鎖最小得点未満の正の加点は操作の証拠。欠測や逆行は使わない。"""
        if self.stable_score is not None and self.last is not None:
            delta = display-self.stable_score
            if 0 < delta < CHAIN_TOTAL_MIN_SCORE:
                self.boundary = dict(t_sec=stamp, before=self.stable_score,
                                     after=display, reason='stable_placement')
        self.stable_score = display

    def _total(self, formula: float | None) -> float | None:
        """元セッションがリセット済みなら差し引かず、未知値は未知のまま返す。"""
        if formula is None or formula < self.offset[1]:
            return formula
        return formula-self.offset[1]
