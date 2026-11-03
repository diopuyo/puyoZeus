"""左右のMCを独立した世代で回収するライブ専用adapter。"""
from __future__ import annotations

import math
from typing import Any

from .live_counter import AsyncCounter, EMPTY_RESULT, LIVE_ROLLOUTS

SIDES = ('1P', '2P')


class SideCounter:
    """一つの探索processを共有し、片側結果を相手の盤面変更で無効にしない。"""

    def __init__(self, executor: Any = None, rollouts: int = LIVE_ROLLOUTS) -> None:
        if type(rollouts) is not int or rollouts < 1:
            raise ValueError('mc_rolloutsは正の整数が必要です')
        first = AsyncCounter(executor, rollouts)
        self.lanes = (first, AsyncCounter(first.executor, rollouts))
        self.executor, self.owns_executor = first.executor, first.owns_executor
        self.rollouts, self.required = rollouts, SIDES
        self._last_result = EMPTY_RESULT
        self.last_hands = self.last_budget_sec = 0.0
        self.contributes = False

    @property
    def pending(self) -> bool:
        return any(lane.pending for side, lane in zip(SIDES, self.lanes) if side in self.required)

    def invalidate(self, context: tuple) -> None:
        """試合と入力状態は共通、盤面は自側だけ。通知中のABAも保持する。"""
        for index, lane in enumerate(self.lanes):
            lane.invalidate((*context[:2], context[index+2]))

    def update(self, b1: Any, b2: Any, budget_sec: float = 0.0,
               next1: tuple | None = None, next2: tuple | None = None,
               t_sec: float | None = None, defender_side: str | None = None,
               threshold_ojama: float | None = None, reuse_if_board_unchanged: bool = False,
               quantize_budget_sec: bool = False) -> tuple:
        from scripts.visualize_advantage_overlay import COUNTER_SCALE
        self.required = (defender_side,) if defender_side in SIDES else SIDES
        probabilities, hands = [], []
        for side, lane, board, pair in zip(SIDES, self.lanes, (b1, b2), (next1, next2)):
            # 同じ盤面を両引数へ渡し、旧APIの片側探索だけを呼ぶ。
            result = lane.update(board, board, budget_sec if side in self.required else 0,
                next1=pair, t_sec=t_sec, defender_side='1P', threshold_ojama=threshold_ojama,
                reuse_if_board_unchanged=reuse_if_board_unchanged, quantize_budget_sec=quantize_budget_sec)
            probabilities.append(result[1])
            if side in self.required:
                hands.append(lane.last_hands)
        p1, p2 = probabilities
        self.contributes = all(math.isfinite(p) for side, p in zip(SIDES, probabilities) if side in self.required)
        advantage = (p1-p2)*COUNTER_SCALE if all(map(math.isfinite, probabilities)) else 0.0
        self.last_budget_sec = budget_sec
        self.last_hands = sum(hands)/len(hands)
        self._last_result = (advantage, p1, p2)
        return self._last_result

    def status(self) -> dict[str, Any]:
        """集約状態に加え、値の出所を左右別の世代・時刻で公開する。"""
        states = {side: dict(lane.status(), required=side in self.required,
                            probability=lane._last_result[1] if lane._last_result and
                            math.isfinite(lane._last_result[1]) else None)
                  for side, lane in zip(SIDES, self.lanes)}
        required = [states[side] for side in self.required]
        generation = sum(s['generation'] for s in states.values())
        complete = all(s['result_generation'] == s['generation'] for s in required)
        times = [s['result_time'] for s in required if s['result_time'] is not None]
        requests = [s['request_time'] for s in required if s['request_time'] is not None]
        return dict(pending=self.pending, generation=generation,
            result_generation=generation if complete else None,
            request_time=max(requests) if requests else None, result_time=min(times) if times else None,
            submitted=sum(s['submitted'] for s in states.values()),
            accepted=sum(s['accepted'] for s in states.values()),
            discarded=sum(s['discarded'] for s in states.values()),
            error=next((s['error'] for s in states.values() if s['error']), None),
            sides=states, rollouts=self.rollouts, contributes=self.contributes,
            source='mc_counter', state='計算中' if self.pending else '確定')

    def close(self) -> None:
        self.lanes[0].close()
