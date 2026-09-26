"""既存overlayの観測とイベント評価器を結ぶ読み取り専用アダプター。"""
from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Callable
from typing import Any, Protocol

import numpy as np

from src.board import Board
from src.board_state_machine import BoardState
from src.chain_detector import CHAIN_MECHANISM_FORMULA, CHAIN_MECHANISM_FORMULA_READ
from src.chain_id_resolver import ChainObservation, ObservationKind
from src.exchange_event_evaluator import ExchangeModels, StaticInput
from src.exchange_event_features import prefire_side_features
from src.exchange_event_tracker import ExchangeEventTracker, SIDE_LABELS

STATIC_INTERVAL_SEC = .3
UNUSED_S1_M0 = .5  # S1/S3の特徴列にはM0がなく、G_feにはこの値を渡さない。


class EndSignals(Protocol):
    """既存の絶対終了判定器を再利用するための境界。"""

    def update(self, result: Any, snapshot: Any, t_sec: float) -> str | None: ...


@dataclass(frozen=True)
class ConfirmedSide:
    """STABLE時だけ複製した盤面・NEXT。発火検知遅延を遡って参照できる。"""

    t_sec: float
    board: Board
    queue: np.ndarray


StaticBuilder = Callable[[tuple[Board, Board], Any, float, float], StaticInput]
M0Predictor = Callable[[np.ndarray, np.ndarray], float]
SignalFactory = Callable[[int, Any, Any, float], EndSignals]


class ExchangeEventOverlay:
    """確定盤面以外を特徴化せず、毎認識フレームの通知を束ねる。"""

    def __init__(self, models: ExchangeModels, build_static: StaticBuilder,
                 signal_factory: SignalFactory, m0_predictor: M0Predictor | None = None) -> None:
        self.tracker = ExchangeEventTracker(models)
        self._build_static, self._signal_factory = build_static, signal_factory
        self._m0 = m0_predictor
        self._history: list[list[ConfirmedSide]] = [[], []]
        self._snapshots: list[tuple[float, Any]] = []
        self._signals: dict[int, EndSignals] = {}
        self._counts = (0, 0)
        self._previous = (None, None)
        self._game: int | None = None
        self._start: float | None = None
        self._last_static = -np.inf
        self._falling = [False, False]
        self._chain_keys: list[tuple | None] = [None, None]
        self._scores: list[list[tuple[float, float]]] = [[], []]

    def update(self, result: Any, snapshot: Any, finalization: Any,
               t_sec: float, game_idx: int,
               formula_totals: tuple[float | None, float | None] = (None, None),
               displayed_scores: tuple[float | None, float | None] | None = None) -> None:
        """発火→両側終了/確定→S3→着地後G_feの順で一括更新する。"""
        sides = (result.p1, result.p2)
        if self._game != game_idx:
            self._reset(game_idx, t_sec)
        triggers = tuple(s.chain_event.trigger_sec if s.chain_event else None for s in sides)
        fresh = self._changed_chains(sides, triggers)
        if fresh:
            self._fire(result, snapshot, t_sec, triggers, fresh)
        self._observe_signals(result, snapshot, finalization, t_sec)
        self._observe_scores(sides, t_sec, formula_totals, displayed_scores)
        self.tracker.finish_frame(t_sec)
        self._remember(sides, snapshot, t_sec)
        if all(s.state == BoardState.STABLE for s in sides) and all(self._history):
            self._static(snapshot, t_sec)
        self._previous = tuple(s.state for s in sides)

    def _reset(self, game_idx: int, t_sec: float) -> None:
        """試合内の参照履歴と信号基準をまとめて初期化する。"""
        self.tracker.boundary(game_idx, t_sec)
        self._game, self._start = game_idx, None
        self._history, self._snapshots, self._signals = [[], []], [], {}
        self._counts, self._previous = (0, 0), (None, None)
        self._last_static = -np.inf
        self._falling = [False, False]
        self._chain_keys = [None, None]
        self._scores = [[], []]

    def _observe_scores(self, sides: tuple, t_sec: float, formula_totals: tuple,
                        displayed_scores: tuple | None) -> None:
        """発火前の実表示得点を基準とし、会計値を早期確定には使わない。"""
        for idx, (label, side) in enumerate(zip(SIDE_LABELS, sides)):
            chain = self.tracker.latest_chain(label)
            before = None if chain is None else next(
                (s for t, s in reversed(self._scores[idx]) if t < chain.trigger_sec), None)
            score = side.score if displayed_scores is None else displayed_scores[idx]
            self.tracker.observe_score(label, t_sec, score, before, formula_totals[idx])
            if side.score is not None:
                self._scores[idx].append((t_sec, side.score))

    def _changed_chains(self, sides: tuple, triggers: tuple) -> list[tuple[int, float]]:
        """triggerが同じ段継続も観測し、Noneの点滅後の残響は重複送信しない。"""
        changed = []
        for idx, (side, trigger) in enumerate(zip(sides, triggers)):
            event = side.chain_event
            if event is None:
                continue
            key = (trigger, event.mechanism, event.chain_count, event.total_score)
            if key != self._chain_keys[idx]:
                changed.append((idx, trigger))
        return changed

    def _fire(self, result: Any, snapshot: Any, t_sec: float,
              triggers: tuple, fresh: list[tuple[int, float]]) -> None:
        """最初の発火より前の各側STABLE盤面を凍結する。"""
        first = min(ts for _, ts in fresh)
        selected = [next((s for s in reversed(h) if s.t_sec < first), None)
                    for h in self._history]
        if not all(selected):
            return  # 発火前盤面が揃う以前の区間は未来盤面で補わない。
        elapsed = max(0.0, first - self._start)
        boards = tuple(s.board for s in selected)
        before_snap = next(s for t, s in reversed(self._snapshots) if t < first)
        static = self._build_static(boards, before_snap, elapsed, UNUSED_S1_M0)
        if self.tracker.current is None:
            prefire = np.stack([prefire_side_features(s.board._grid, s.queue, elapsed)
                                for s in selected])
        else:
            prefire = self.tracker.firing.prefire_sides
        observations = self._observations(result, t_sec, fresh)
        self.tracker.fire(t_sec=t_sec, triggers=triggers, static=static,
                          prefire_sides=prefire, score_elapsed_sec=elapsed,
                          observations=observations)
        if self.tracker.current is None:
            return
        for chain in self.tracker.current.chains:
            if chain.chain_id not in self._signals:
                idx = SIDE_LABELS.index(chain.side)
                self._signals[chain.chain_id] = self._signal_factory(idx, result, snapshot, t_sec)

    def _observations(self, result: Any, t_sec: float,
                      changed: list[tuple[int, float]]) -> tuple[ChainObservation, ...]:
        """既存episodeアダプターと同じ機構→観測種別でresolverへ供給する。"""
        observations = []
        for idx, trigger in changed:
            event = (result.p1, result.p2)[idx].chain_event
            kind = (ObservationKind.FORMULA_STEP if event.mechanism in
                    (CHAIN_MECHANISM_FORMULA, CHAIN_MECHANISM_FORMULA_READ)
                    else ObservationKind.CHAIN_SETTLED)
            observations.append(ChainObservation(SIDE_LABELS[idx], t_sec, kind,
                                                  event.chain_count, event.total_score, event.mechanism))
            self._chain_keys[idx] = (trigger, event.mechanism, event.chain_count, event.total_score)
        return tuple(observations)

    def _observe_signals(self, result: Any, snapshot: Any, finalization: Any,
                         t_sec: float) -> None:
        """同じ得点の別連鎖も確定回数の増分で識別する。"""
        counts = (finalization.finalized_count_p1, finalization.finalized_count_p2)
        deltas = (finalization.chain_total_score_p1, finalization.chain_total_score_p2)
        for idx, (label, side) in enumerate(zip(SIDE_LABELS, (result.p1, result.p2))):
            chain = self.tracker.latest_chain(label)
            if chain is not None and chain.end_signal_sec is None:
                reason = self._signals[chain.chain_id].update(result, snapshot, t_sec)
                if reason:
                    self.tracker.end(label, t_sec, reason)
            if counts[idx] > self._counts[idx]:
                self.tracker.finalize(label, t_sec, deltas[idx])
            if side.state == BoardState.OJAMA_FALL and self._previous[idx] != side.state:
                self.tracker.fall_start(label, t_sec)
                self._falling[idx] = True
            # EventPhysicalRecorderと同じく、落下後の最初のSTABLEで着地完了。
            if self._falling[idx] and side.state == BoardState.STABLE and side.confirmed_board is not None:
                self.tracker.landing(label, t_sec)
                self._falling[idx] = False
        self._counts = counts

    def _remember(self, sides: tuple, snapshot: Any, t_sec: float) -> None:
        """両側履歴を独立に保存し、NON-STABLEの生盤面を排除する。"""
        saved = False
        for idx, side in enumerate(sides):
            if side.state != BoardState.STABLE or side.confirmed_board is None:
                continue
            queue = np.array([*(side.next_pair or (0, 0)), *(side.dnext_pair or (0, 0))])
            self._history[idx].append(ConfirmedSide(t_sec, side.confirmed_board.copy(), queue))
            saved = True
        if saved:
            self._snapshots.append((t_sec, snapshot))
            if self._start is None:
                self._start = t_sec

    def _static(self, snapshot: Any, t_sec: float) -> None:
        """M0未指定ならG_feを偽の確率で動かさず、配線待ちを明示する。"""
        latest = tuple(h[-1] for h in self._history)
        confirmed = tuple(s.t_sec for s in latest)
        if not self.tracker.ready_for_static(t_sec, confirmed):
            return
        if self._m0 is None:
            self.tracker.close_confirmed(t_sec, confirmed)
            if self.tracker.probability is None:
                self.tracker.source = "waiting_m0"
            return
        if t_sec - self._last_static < STATIC_INTERVAL_SEC:
            return
        boards = tuple(s.board for s in latest)
        probability = self._m0(np.stack([b._grid for b in boards]),
                               np.stack([s.queue for s in latest]))
        event = self._build_static(boards, snapshot, t_sec - self._start, probability)
        self.tracker.static(event, t_sec, confirmed)
        self._last_static = t_sec


def displayed_scores_from_pipeline(pipeline: object, frame: np.ndarray) -> tuple:
    """保持済みscoreではなく当該フレームの表示値を読む。会計状態は更新しない。"""
    ocr = getattr(pipeline, "_score_ocr", None)
    if ocr is None:
        return (None, None)
    return tuple(ocr.read_side(frame, side)[0] for side in SIDE_LABELS)
