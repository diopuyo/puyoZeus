"""E22: 窒息セル占有側のbaseline新発火だけを式読取・実得点で確認する。"""
from __future__ import annotations
from collections import deque
from copy import deepcopy
from typing import Any
import math
from src.chain_detector import CHAIN_MECHANISM_BASELINE
from src.exchange_event_death_candidate import occupied

# 既存3動画の可測15件のP99（higher）=0.4666666667秒。欠測165件は除外して併記。
# logs/e22/FORMULA_DELAY.json。2026-09-28の事前測定を固定し、評価結果から調整しない。
FORMULA_HOLD_FRAMES, FORMULA_MEASUREMENT_FPS = 14, 30
FORMULA_HOLD_MAX_SEC = FORMULA_HOLD_FRAMES / FORMULA_MEASUREMENT_FPS
TIME_EPSILON = 1e-9
SCORE_HISTORY_SIZE = 256
SIDES = ('1P', '2P')


def valid_score(value: Any) -> bool:
    """欠測・非数・負値を得点増加の証拠にしない。"""
    return value is not None and math.isfinite(float(value)) and value >= 0


class DeathFormulaGuard:
    """認識通知を変えず、評価器へ渡すbaselineの確定時点だけを限定する。"""

    def __init__(self) -> None:
        self.audit: list[dict] = []
        self.reset()

    def reset(self) -> None:
        """試合境界で未確定通知を破棄し、別試合の式で復活させない。"""
        if hasattr(self, 'pending'):
            for idx in range(len(SIDES)):
                for trigger in list(self.pending[idx]):
                    self._finish(idx, trigger, 'discarded_boundary', self.stamp)
        self.pending: list[dict[float, tuple[Any, dict]]] = [{}, {}]
        self.resolved: list[dict[float, str]] = [{}, {}]
        self.scores = [deque(maxlen=SCORE_HISTORY_SIZE), deque(maxlen=SCORE_HISTORY_SIZE)]
        self.visible, self.current_scores = (False, False), (None, None)
        self.formula_sec = [float('-inf'), float('-inf')]
        self.dead: set[str] = set()
        self.stamp, self.game = 0., None

    def observe(self, result: Any, stamp: float, game: int, scores: tuple | None,
                visible: tuple) -> None:
        """式は当該フレームの有効読取、得点は会計補正前の表示値だけを使う。"""
        self.stamp, self.game, self.visible = stamp, game, visible
        self.current_scores = scores if scores is not None else (result.p1.score, result.p2.score)
        self.dead.update(getattr(result, 'confirmed_dead_sides', ()))
        for idx, score in enumerate(self.current_scores):
            if visible[idx]:
                self.formula_sec[idx] = stamp
            if valid_score(score):
                self.scores[idx].append((stamp, float(score)))
            if SIDES[idx] in self.dead:
                for trigger in list(self.pending[idx]):
                    self._finish(idx, trigger, 'discarded_death', stamp)

    def notifications(self, idx: int, event: Any, history: list, accepted_key: tuple | None) -> list:
        """対象外は原通知のまま通し、確認された保留分だけ元時刻で返す。"""
        ready = self._release(idx)
        if event is None:
            return ready
        trigger = event.trigger_sec
        if event.mechanism != CHAIN_MECHANISM_BASELINE:
            return ready+[event]
        if trigger in self.resolved[idx]:
            if self.resolved[idx][trigger].startswith('discarded') or any(e.trigger_sec == trigger for e in ready):
                return ready
            return ready+[event]
        if trigger in self.pending[idx]:
            return ready
        previous = next((s for s in reversed(history) if s.t_sec <= trigger), None)
        if (accepted_key is not None and accepted_key[0] == trigger) or previous is None or not occupied(previous.board):
            return ready+[event]
        baseline = next((value for stamp, value in reversed(self.scores[idx]) if stamp < trigger), None)
        if self.formula_sec[idx] >= trigger or self._score_increased(idx, baseline, trigger):
            return ready+[event]
        row = dict(game=self.game, side=SIDES[idx], trigger_sec=trigger, held_sec=self.stamp,
                   deadline_sec=self.stamp+FORMULA_HOLD_MAX_SEC, score_before=baseline,
                   board_sec=previous.t_sec, outcome='pending')
        self.audit.append(row)
        self.pending[idx][trigger] = (deepcopy(event), row)
        return ready

    def _release(self, idx: int) -> list:
        """期限切れを自動承認せず、後から別の連鎖の式で再送しない。"""
        ready = []
        for trigger, (event, row) in list(self.pending[idx].items()):
            if self.stamp > row['deadline_sec']+TIME_EPSILON:
                self._finish(idx, trigger, 'discarded_timeout', self.stamp)
            elif self.formula_sec[idx] >= trigger or self._score_increased(idx, row['score_before'], trigger):
                reason = 'confirmed_formula' if self.formula_sec[idx] >= trigger else 'confirmed_score'
                ready.append(event)
                self._finish(idx, trigger, reason, self.stamp)
        return ready

    def _score_increased(self, idx: int, before: float | None, trigger: float) -> bool:
        """予測得点・累積式の残響を使わず、観測した表示得点の増加を見る。"""
        return valid_score(before) and any(stamp >= trigger and score > before
                                          for stamp, score in self.scores[idx])

    def _finish(self, idx: int, trigger: float, outcome: str, stamp: float) -> None:
        """通知単位の最終状態を一度だけ保存する。"""
        _, row = self.pending[idx].pop(trigger)
        row.update(outcome=outcome, resolved_sec=stamp, wait_sec=stamp-row['held_sec'])
        self.resolved[idx][trigger] = outcome
