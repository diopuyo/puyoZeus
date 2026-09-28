"""全通知の物理状態更新と、公開時だけの数値評価を分離する。"""
from __future__ import annotations

from typing import Any

from src.board_state_machine import BoardState
from src.exchange_event_overlay import ExchangeEventOverlay
from src.exchange_event_tracker import ExchangeEventTracker


class DeferredTracker(ExchangeEventTracker):
    """段更新・終了時刻は毎通知で確定し、モデル入力だけ最新一件を保持する。"""

    def __init__(self, models: Any) -> None:
        super().__init__(models)
        self.pending: tuple | None = None
        self._deferred_records: dict[int, tuple] = {}

    def _evaluate(self, event: Any, source: str, t_sec: float) -> bool:
        # 未計算値を既知の勝率と偽らない。公開直前にこの行だけ数値を埋める。
        value = dict(source=source, t_sec=t_sec, p1=None, deferred=True)
        self.pending = (event, source, t_sec, value)
        if self.current is not None:
            self.current.values.append(value)
            self._deferred_records[self.current.exchange_id] = self.pending
        return True

    def calculate(self) -> None:
        if self.pending is None:
            return
        event, source, t_sec, value = self.pending
        self.pending = None
        if super()._evaluate(event, source, t_sec):
            value.update(p1=self.probability, deferred=False)
            if self.current is not None:
                self.current.values.pop()  # 入力記録と計算結果を同じ行へまとめる。

    def boundary(self, game_idx: int, t_sec: float) -> None:
        if game_idx != self._game_idx:
            self.pending = None
            self._deferred_records.clear()
        super().boundary(game_idx, t_sec)

    def _resume_existing(self, observations: tuple) -> bool:
        """物理区間の再開時、閉じる前に未計算だったS3の入力も復元する。"""
        resumed = super()._resume_existing(observations)
        if resumed and self.current is not None:
            pending = self._deferred_records.get(self.current.exchange_id)
            if pending is not None and pending[-1].get('deferred'):
                self.pending = pending
        return resumed

    def _close(self, t_sec: float, reason: str) -> None:
        self.pending = None
        super()._close(t_sec, reason)


class SplitExchangeOverlay(ExchangeEventOverlay):
    """特徴の更新要求とSTABLE入力を保持し、公開時の最新状態を評価する。"""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.tracker = DeferredTracker(self.tracker.models)
        self.latest: tuple | None = None
        self.static_input: tuple | None = None
        self.feature_dirty = False
        self.notifications = 0
        self.calculations = 0
        self._landed = False

    def update(self, result: Any, snapshot: Any, finalization: Any,
               t_sec: float, game_idx: int, formula_totals: tuple = (None, None),
               displayed_scores: tuple | None = None,
               formula_visible: tuple = (False, False)) -> None:
        if self._game != game_idx:
            self._reset(game_idx, t_sec)
            self.static_input = None
        self.tracker.begin_frame()
        sides = (result.p1, result.p2)
        self._observe_placements(sides, displayed_scores, t_sec)
        triggers = tuple(s.chain_event.trigger_sec if s.chain_event else None for s in sides)
        fresh = self._changed_chains(sides, triggers, t_sec)
        if fresh:
            self._fire(result, snapshot, t_sec, triggers, fresh)
            if self.tracker.current is not None:
                self.static_input = None
        for idx, visible in enumerate(formula_visible):
            if visible:
                self._last_formula[idx] = t_sec
                self.tracker.activity(('1P', '2P')[idx], t_sec)
        self._observe_signals(result, snapshot, finalization, t_sec, formula_visible)
        self._observe_scores(sides, t_sec, formula_totals, displayed_scores)
        self._remember(sides, snapshot, t_sec)
        self.feature_dirty = True
        self.tracker.confirm_frame_inputs(t_sec)
        self.tracker.finish_frame(t_sec)
        self._observe_landing(result, snapshot, t_sec)
        stable = [s.state == BoardState.STABLE for s in sides]
        settled = any(stable) if self._per_side_settled else all(stable)
        if settled and all(self._history):
            self._mark_static(snapshot, t_sec)
        self.latest = (result, snapshot, t_sec)
        self._previous = tuple(s.state for s in sides)
        self.notifications += 1

    def _observe_landing(self, result: Any, snapshot: Any, t_sec: float) -> None:
        projection = self._landing_projection
        projection._observe_frame(self, result, snapshot)
        self._landed = projection._refresh_death(self, t_sec)
        # 新発火・段数・着地・境界による保持解除は間引かない。
        if projection.death is not None and all(self._history):
            projection._select_boards(self, t_sec)

    def _mark_static(self, snapshot: Any, t_sec: float) -> None:
        if self._landing_projection.death is not None:
            return
        confirmed = tuple(h[-1].t_sec for h in self._history)
        if self.tracker.ready_for_static(t_sec, confirmed):
            self.tracker.close_confirmed(t_sec, confirmed)
            self.static_input = (snapshot, t_sec, tuple(h[-1] for h in self._history))

    def calculate(self) -> None:
        """同じ通知の二重計算を防ぎ、S3→着弾→静止の既存優先順を保つ。"""
        if self.latest is None or not self.feature_dirty:
            return
        result, snapshot, t_sec = self.latest
        super()._refresh_features(snapshot, t_sec)
        self.tracker.finish_frame(t_sec)
        self.tracker.calculate()
        if not self._landed:
            self._landing_projection.update(self, result, snapshot, t_sec)
        if self.static_input is not None and self.tracker.current is None:
            self._calculate_static()
        self.feature_dirty = False
        self.calculations += 1

    def _calculate_static(self) -> None:
        snapshot, t_sec, latest = self.static_input
        self.static_input = None
        if self._landing_projection.death is not None or self._m0 is None:
            return
        import numpy as np
        boards = tuple(s.board for s in latest)
        try:
            probability = self._m0(np.stack([b._grid for b in boards]),
                                   np.stack([s.queue for s in latest]))
            event = self._build_static(boards, snapshot, t_sec-self._start, probability)
            self.tracker._evaluate(event, 'G_fe', t_sec)
            self.tracker.calculate()
        except (ValueError, TypeError, FloatingPointError) as error:
            self.tracker.missing_input('static_input: '+str(error), t_sec, 'G_fe')


def sampled_ema(base: type, bridge: Any) -> type:
    """公開間の通知件数を記録し、最新入力の零次保持でEMAを再現可能にする。"""
    class SampledEMA(base):
        def apply(self, adv: float, probability: float,
                  t_sec: float | None = None) -> tuple[float, float]:
            from scripts.visualize_advantage_overlay import EMA_ALPHA
            if t_sec is None or t_sec != self.last_sec:
                count = bridge.notification_count-getattr(self, '_previous_count', 0)
                alpha = 1-(1-EMA_ALPHA)**max(1, count)
                self.adv += alpha*(adv-self.adv)
                self.probability += alpha*(probability-self.probability)
                self.last_sec, self._previous_count = t_sec, bridge.notification_count
            return self.adv, self.probability
    return SampledEMA
