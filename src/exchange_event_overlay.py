"""既存overlayの観測とイベント評価器を結ぶ読み取り専用アダプター。"""
from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Callable
from typing import Any, Protocol

import numpy as np

from src.board import Board, COLOR_UNKNOWN
from src.board_state_machine import BoardState
from src.chain_detector import CHAIN_MECHANISM_FORMULA, CHAIN_MECHANISM_FORMULA_READ
from src.chain_id_resolver import ChainObservation, ObservationKind
from src.exchange_event_evaluator import CountObservation, ExchangeModels, StaticInput
from src.exchange_event_features import prefire_side_features
from src.exchange_event_tracker import ExchangeEventTracker, SIDE_LABELS, valid_nonnegative
from src.score_ocr import FORMULA_SESSION_RESET_SEC
from src.ojama_accounting import CHAIN_TOTAL_MIN_SCORE
from src.scoring import calculate_chain_score, compute_effective_rate

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
                 signal_factory: SignalFactory, m0_predictor: M0Predictor | None = None,
                 per_side_settled: bool = False, live_count: bool = False) -> None:
        self.tracker = ExchangeEventTracker(models, live_count=live_count)
        self._live_count_key: tuple | None = None
        self._build_static, self._signal_factory = build_static, signal_factory
        self._m0 = m0_predictor
        self._per_side_settled = per_side_settled
        self._history: list[list[ConfirmedSide]] = [[], []]
        self._snapshots: list[tuple[float, Any]] = []
        self._signals: dict[int, EndSignals] = {}
        self._counts = (0, 0)
        self._previous = (None, None)
        self._game: int | None = None
        self._start: float | None = None
        self._falling = [False, False]
        self._chain_keys: list[tuple | None] = [None, None]
        self._scores: list[list[tuple[float, float]]] = [[], []]
        self._last_formula: list[float | None] = [None, None]
        self._last_displayed: list[float | None] = [None, None]
        self._feature_cache: dict[tuple, np.ndarray] = {}
        from src.exchange_event_landing import ExchangeLandingProjection
        self._landing_projection = ExchangeLandingProjection()

    def update(self, result: Any, snapshot: Any, finalization: Any,
               t_sec: float, game_idx: int,
               formula_totals: tuple[float | None, float | None] = (None, None),
               displayed_scores: tuple[float | None, float | None] | None = None,
               formula_visible: tuple[bool, bool] = (False, False)) -> None:
        """発火→両側終了/確定→S3→着地後G_feの順で一括更新する。"""
        sides = (result.p1, result.p2)
        if self._game != game_idx:
            self._reset(game_idx, t_sec)
        self.tracker.begin_frame()
        self._observe_placements(sides, displayed_scores, t_sec)
        triggers = tuple(s.chain_event.trigger_sec if s.chain_event else None for s in sides)
        fresh = self._changed_chains(sides, triggers, t_sec)
        if fresh:
            self._fire(result, snapshot, t_sec, triggers, fresh)
        for idx, (label, visible) in enumerate(zip(SIDE_LABELS, formula_visible)):
            if visible:
                self._last_formula[idx] = t_sec
                self.tracker.activity(label, t_sec)
        self._observe_signals(result, snapshot, finalization, t_sec, formula_visible)
        self._observe_scores(sides, t_sec, formula_totals, displayed_scores)
        self._remember(sides, snapshot, t_sec)
        self._refresh_features(snapshot, t_sec)
        self.tracker.confirm_frame_inputs(t_sec)
        self.tracker.finish_frame(t_sec)
        self._landing_projection.update(self, result, snapshot, t_sec)
        stable = [s.state == BoardState.STABLE for s in sides]
        settled = any(stable) if self._per_side_settled else all(stable)
        if settled and all(self._history):
            self._static(snapshot, t_sec)
        self._previous = tuple(s.state for s in sides)

    def _reset(self, game_idx: int, t_sec: float) -> None:
        """試合内の参照履歴と信号基準をまとめて初期化する。"""
        self.tracker.boundary(game_idx, t_sec)
        self._game, self._start = game_idx, None
        self._history, self._snapshots, self._signals = [[], []], [], {}
        self._counts, self._previous = (0, 0), (None, None)
        self._falling = [False, False]
        self._chain_keys = [None, None]
        self._scores = [[], []]
        self._last_formula = [None, None]
        self._last_displayed = [None, None]
        self._feature_cache.clear()
        self._live_count_key = None

    def _observe_placements(self, sides: tuple, scores: tuple | None, t_sec: float) -> None:
        """終了済み区間についても実表示の操作加点を観測し、次の発火と区別する。"""
        for idx, label in enumerate(SIDE_LABELS):
            score = sides[idx].score if scores is None else scores[idx]
            previous = self._last_displayed[idx]
            if not valid_nonnegative(score):
                continue
            if previous is not None and 0 < score - previous < CHAIN_TOTAL_MIN_SCORE:
                self.tracker.note_placement(label, t_sec)
            self._last_displayed[idx] = score

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

    def _changed_chains(self, sides: tuple, triggers: tuple,
                        t_sec: float = 0.0) -> list[tuple[int, float]]:
        """triggerが同じ段継続も観測し、Noneの点滅後の残響は重複送信しない。"""
        changed = []
        for idx, (side, trigger) in enumerate(zip(sides, triggers)):
            event = side.chain_event
            if event is None:
                continue
            key = (trigger, event.mechanism, event.chain_count, event.total_score)
            if not valid_nonnegative(trigger):
                self.tracker.missing_input("unknown_firing_side", t_sec, "S1", (idx, key))
                continue
            if key != self._chain_keys[idx]:
                changed.append((idx, trigger))
        return changed

    def _fire(self, result: Any, snapshot: Any, t_sec: float,
              triggers: tuple, fresh: list[tuple[int, float]]) -> None:
        """今回の発火前に得られた各側STABLE盤面で評価入力を作る。"""
        first = min(ts for _, ts in fresh)
        selected = [next((s for s in reversed(h) if s.t_sec < first), None)
                    for h in self._history]
        if not all(selected) or self._start is None:
            self.tracker.missing_input("missing_prefire_board", t_sec, "S1", triggers)
            return  # 発火前盤面が揃う以前の区間は未来盤面で補わない。
        elapsed = max(0.0, first - self._start)
        boards = tuple(s.board for s in selected)
        before_snap = next((s for t, s in reversed(self._snapshots) if t < first), None)
        if before_snap is None:
            self.tracker.missing_input("missing_prefire_snapshot", t_sec, "S1", triggers)
            return
        try:
            static = self._build_static(boards, before_snap, elapsed, UNUSED_S1_M0)
            prefire = np.stack([prefire_side_features(s.board._grid, s.queue, elapsed)
                                for s in selected])
        except (ValueError, TypeError, FloatingPointError) as error:
            self.tracker.missing_input("prefire_input: " + str(error), t_sec, "S1", triggers)
            return
        observations = self._observations(result, t_sec, fresh)
        self.tracker.fire(t_sec=t_sec, triggers=triggers, static=static,
                          prefire_sides=prefire, score_elapsed_sec=elapsed,
                          observations=observations,
                          count_observation=(CountObservation(np.stack([s.board._grid for s in selected]),
                              np.stack([s.queue for s in selected]), elapsed)
                              if getattr(self.tracker.models, "count_features", False) else None))
        if self.tracker.current is None:
            return
        for chain in self.tracker.current.chains:
            if chain.chain_id not in self._signals:
                idx = SIDE_LABELS.index(chain.side)
                self._signals[chain.chain_id] = self._signal_factory(idx, result, snapshot, t_sec)
                self._predict_completion(chain, (result.p1, result.p2)[idx].chain_event, idx)

    def _predict_completion(self, chain: Any, event: Any, idx: int) -> None:
        """決着先読みと同じ完走シミュレーションを発火時に一度だけ行う。"""
        board = getattr(event, "before_board", None)
        if board is None:
            saved = next((s for s in reversed(self._history[idx])
                          if s.t_sec < chain.trigger_sec), None)
            board = saved.board if saved is not None else None
        if board is None:
            return
        try:
            result = self._landing_projection.simulator.simulate(board)
        except (ValueError, TypeError, FloatingPointError):
            return
        chain.predicted_final_score = float(calculate_chain_score(result).total_score)
        chain.predicted_chain_count = result.chain_count
        if result.chain_count > 0 and not np.any(board._grid == COLOR_UNKNOWN):
            chain.predicted_final_board = result.final_board._grid.tolist()
        # 旧記録には起点盤面がない。発火通知に保存された既存シミュ結果も再用する。
        if (event is not None and getattr(event, "before_board", None) is None
                and event.mechanism != CHAIN_MECHANISM_FORMULA_READ):
            chain.predicted_final_score = max(chain.predicted_final_score, event.total_score)
            chain.predicted_chain_count = max(chain.predicted_chain_count, event.chain_count)
            if (chain.predicted_final_score != calculate_chain_score(result).total_score
                    or chain.predicted_chain_count != result.chain_count):
                chain.predicted_final_board = None

    def _refresh_features(self, snapshot: Any, t_sec: float) -> None:
        """両側の最新確定盤面でDと近未来火力を更新し、同一盤面の探索を再用する。"""
        if getattr(self.tracker.models, "count_features", False):
            if self.tracker.live_count:
                self._refresh_live_count(t_sec)
            return
        if self.tracker.current is None or self._start is None or not all(self._history):
            return
        elapsed = t_sec - self._start
        latest = tuple(h[-1] for h in self._history)
        try:
            static = self._build_static(tuple(s.board for s in latest), snapshot, elapsed, UNUSED_S1_M0)
            features = []
            for side in latest:
                grid = side.board._grid
                key = (grid.tobytes(), grid.dtype.str, side.queue.tobytes(), compute_effective_rate(elapsed))
                if key not in self._feature_cache:
                    self._feature_cache[key] = prefire_side_features(grid, side.queue, elapsed)
                features.append(self._feature_cache[key])
            self.tracker.refresh_features(static, np.stack(features))
        except (ValueError, TypeError, FloatingPointError) as error:
            self.tracker.missing_input("current_input: " + str(error), t_sec, "S3")

    def _refresh_live_count(self, t_sec: float) -> None:
        """STABLE更新と連鎖遷移だけでcountを交換し、暫定S3を失効させる。"""
        tracker = self.tracker
        if tracker.current is None or self._start is None or not all(self._history):
            return
        grids, queues = [], []
        for idx, history in enumerate(self._history):
            latest = history[-1]
            chain = tracker.latest_chain(SIDE_LABELS[idx])
            grid = latest.board._grid
            if chain is not None and (chain.end_signal_sec is None
                                      or latest.t_sec < chain.end_signal_sec):
                if chain.predicted_final_board is None:
                    from src.exchange_event_count_features import completion
                    saved = next((s for s in reversed(history) if s.t_sec < chain.trigger_sec), None)
                    if saved is None:
                        tracker.missing_input("missing_live_completion", t_sec, "S3", chain.chain_id)
                        return
                    _, _, final = completion(saved.board._grid.astype(np.int8).tobytes(), True,
                                              chain.trigger_sec - self._start)
                    grid = np.frombuffer(final, np.int8).reshape(saved.board._grid.shape)
                else:
                    grid = np.asarray(chain.predicted_final_board, dtype=np.int8)
            grids.append(np.asarray(grid, dtype=np.int8))
            queues.append(latest.queue)
        observation = CountObservation(np.stack(grids), np.stack(queues),
            t_sec - self._start, live=True, score_elapsed_sec=tracker._score_elapsed)
        key = (tracker.current.exchange_id, observation.grids.tobytes(),
               observation.queues.tobytes(), compute_effective_rate(observation.elapsed_sec),
               tuple(h[-1].board._grid.tobytes() for h in self._history))
        if key == self._live_count_key:
            return
        # E15はcountだけを更新し、既存のD・到着特徴の定義を保つ。
        tracker.refresh_features(tracker.firing.static, tracker.firing.prefire_sides, observation)
        self._live_count_key = key

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
                         t_sec: float, formula_visible: tuple = (False, False)) -> None:
        """同じ得点の別連鎖も確定回数の増分で識別する。"""
        counts = (finalization.finalized_count_p1, finalization.finalized_count_p2)
        deltas = (finalization.chain_total_score_p1, finalization.chain_total_score_p2)
        for idx, (label, side) in enumerate(zip(SIDE_LABELS, (result.p1, result.p2))):
            chain = self.tracker.latest_chain(label)
            if chain is not None:
                signal = self._signals.get(chain.chain_id)
                if signal is None:
                    self.tracker.missing_input("missing_end_signal", t_sec, "S3", chain.chain_id)
                reason = signal.update(result, snapshot, t_sec) if signal is not None else None
                recent_formula = (self._last_formula[idx] is not None and
                                  t_sec - self._last_formula[idx] < FORMULA_SESSION_RESET_SEC)
                blocked = formula_visible[idx] or (reason == "slide" and recent_formula)
                if blocked and signal is not None:
                    invalidate = getattr(signal, "invalidate", None)
                    if invalidate is not None:
                        invalidate()
                if reason and not blocked:
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
        if self._landing_projection.death is not None:
            return
        latest = tuple(h[-1] for h in self._history)
        confirmed = tuple(s.t_sec for s in latest)
        if not self.tracker.ready_for_static(t_sec, confirmed):
            return
        if self._m0 is None:
            self.tracker.close_confirmed(t_sec, confirmed)
            if self.tracker.probability is None:
                self.tracker.source = "waiting_m0"
            return
        boards = tuple(s.board for s in latest)
        try:
            probability = self._m0(np.stack([b._grid for b in boards]),
                                   np.stack([s.queue for s in latest]))
            event = self._build_static(boards, snapshot, t_sec - self._start, probability)
        except (ValueError, TypeError, FloatingPointError) as error:
            self.tracker.missing_input("static_input: " + str(error), t_sec, "G_fe")
            return
        self.tracker.static(event, t_sec, confirmed)


def displayed_scores_from_pipeline(pipeline: object, frame: np.ndarray) -> tuple:
    """保持済みscoreではなく当該フレームの表示値を読む。会計状態は更新しない。"""
    ocr = getattr(pipeline, "_score_ocr", None)
    if ocr is None:
        return (None, None)
    return tuple(ocr.read_side(frame, side)[0] for side in SIDE_LABELS)


def formula_visible_from_pipeline(pipeline: object) -> tuple[bool, bool]:
    """毎フレーム初期化される掛け算式の実読結果だけを使う。"""
    return tuple(bool(getattr(getattr(pipeline, "_formula_last_read_" + label, None),
                              "valid", False)) for label in ("1p", "2p"))
