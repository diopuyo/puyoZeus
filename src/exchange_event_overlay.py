"""既存overlayの観測とイベント評価器を結ぶ読み取り専用アダプター。"""
from __future__ import annotations

from dataclasses import dataclass, is_dataclass, replace
from copy import copy
from itertools import zip_longest
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


def _replaced(value: Any, **changes: Any) -> Any:
    """入力を変えずに一部の属性だけ差し替えた複製を返す。

    リアルタイムの側データは frozen dataclass で代入できないため replace を使う。
    それ以外 (記録再生の可変オブジェクト) は従来どおり浅い複製へ代入する。
    """
    if is_dataclass(value) and not isinstance(value, type) and value.__dataclass_params__.frozen:
        return replace(value, **changes)
    saved = copy(value)
    for name, item in changes.items():
        setattr(saved, name, item)
    return saved

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
    """現在層は確定盤面だけを使い、途中盤面は明示ONの予測層へ限定する。"""

    def __init__(self, models: ExchangeModels, build_static: StaticBuilder,
                 signal_factory: SignalFactory, m0_predictor: M0Predictor | None = None,
                 per_side_settled: bool = False, live_count: bool = False,
                 e16: bool = False, count_sync: bool = False,
                 death_guard: bool = False, evaluation_layers: bool = False,
                 completion_check: bool = False, landing_counter_response: bool = False,
                 confirmed_death_hold: bool = False, landing_counter_prob: bool = False,
                 counter_probability_model: Any = None, landing_hands_spec: bool = False,
                 death_candidate_guard: bool = False, death_formula_guard: bool = False,
                 multi_landing_death: bool = False, landing_state_safety: bool = False,
                 pending_ledger: bool = False, color_score_safety: bool = False,
                 completion_recovery: bool = False, midchain_completion: bool = False,
                 death_pending_ledger: bool = False, hidden_row_death: bool = False,
                 midchain_single_observation: bool = False,
                 prefire_candidates: bool = False, prefire_snapshot: bool = False,
                 hidden_row_belief: bool = False, prefire_stage_timeout: bool = False,
                 prefire_stage_timeout_only: bool = False,
                 prefire_origin_guard: bool = False, prefire_match_gate: Any = None,
                 post_counter_death_bound: bool = False,
                 single_death_proof_guard: bool = False,
                 single_death_proof_negative_only: bool = False,
                 post_counter_early_exit: bool = False,
                 multilanding_node_limit: int | None = None,
                 hidden_scenario_cap: int | None = None) -> None:
        from src.exchange_hidden_row_probability import validate_scenario_cap
        validate_scenario_cap(hidden_scenario_cap)
        self._initialize_layers(models, live_count, e16, count_sync, death_guard,
                                evaluation_layers, completion_check, confirmed_death_hold)
        self._build_static, self._signal_factory = build_static, signal_factory
        self._m0, self._per_side_settled = m0_predictor, per_side_settled
        self._initialize_state()
        from src.exchange_event_landing import ExchangeLandingProjection
        if landing_counter_prob and counter_probability_model is None:
            from src.landing_counter_probability import LogisticResponseProbability
            counter_probability_model = LogisticResponseProbability.load()
        self._landing_projection = ExchangeLandingProjection(counter_response=landing_counter_response,
            counter_probability_model=counter_probability_model if landing_counter_prob else None,
            hands_spec=landing_hands_spec, multi_landing_death=multi_landing_death,
            landing_state_safety=landing_state_safety, pending_ledger=pending_ledger,
            color_score_safety=color_score_safety, completion_recovery=completion_recovery,
            death_pending_ledger=death_pending_ledger, hidden_row_death=hidden_row_death,
            single_death_proof_guard=single_death_proof_guard,
            single_death_proof_negative_only=single_death_proof_negative_only)
        self._initialize_post_counter(post_counter_death_bound,
            multi_landing_death and death_pending_ledger and confirmed_death_hold, post_counter_early_exit)
        self._initialize_prediction_guards(death_candidate_guard, death_formula_guard,
            midchain_completion, hidden_row_death, midchain_single_observation)
        self._initialize_prefire(prefire_candidates, prefire_snapshot, hidden_row_belief,
                                 prefire_stage_timeout, prefire_stage_timeout_only)
        self._initialize_origin_guard(prefire_origin_guard, prefire_snapshot, prefire_match_gate)
        self._initialize_latency_bounds(multilanding_node_limit, hidden_scenario_cap)

    def _initialize_latency_bounds(self, node_limit: int | None, scenario_cap: int | None) -> None:
        """遅延対策の決定的な予算 (既定 None = 従来どおり無制限)。壁時計ではなく件数で打ち切る。"""
        self._landing_projection.multilanding_node_limit = node_limit
        if scenario_cap is not None:
            belief = getattr(self.tracker, 'hidden_row_belief', None)
            if belief is None:
                raise ValueError('隠し段候補の上限には--hidden-row-beliefが必要')
            belief.scenario_cap = scenario_cap

    def _initialize_layers(self, models: ExchangeModels, live_count: bool, e16: bool,
                           count_sync: bool, death_guard: bool, evaluation_layers: bool,
                           completion_check: bool, confirmed_death_hold: bool) -> None:
        """追跡器とE16評価層を、いずれかの層フラグがONのときだけ組み立てる。"""
        enabled = e16 or count_sync or death_guard or evaluation_layers or completion_check or confirmed_death_hold
        self._confirmed_death_hold = confirmed_death_hold
        self.tracker = ExchangeEventTracker(models, live_count=live_count or enabled)
        from src.exchange_event_layers import ExchangeEvaluationLayers
        self._e16 = ExchangeEvaluationLayers(e16 or count_sync, e16 or death_guard or confirmed_death_hold,
            e16 or evaluation_layers or completion_check, completion_check) if enabled else None
        if enabled and self._e16.layer_enabled:
            self.tracker.layer_rows = []

    def _initialize_state(self) -> None:
        """試合内で持つ履歴・信号・得点の初期値を用意する。"""
        self._history: list[list[ConfirmedSide]] = [[], []]
        self._snapshots: list[tuple[float, Any]] = []
        self._signals: dict[int, EndSignals] = {}
        self._counts, self._previous = (0, 0), (None, None)
        self._game: int | None = None
        self._start: float | None = None
        self._falling = [False, False]
        self._chain_keys: list[tuple | None] = [None, None]
        self._scores: list[list[tuple[float, float]]] = [[], []]
        self._last_formula: list[float | None] = [None, None]
        self._last_displayed: list[float | None] = [None, None]

    def _initialize_post_counter(self, enabled: bool, production_ready: bool,
                                 early_exit: bool = False) -> None:
        """E35の打ち返し後死亡上限を、本番構成の前提が揃うときだけ有効にする。"""
        self._landing_projection.post_counter_bound = None
        if enabled:
            if not production_ready:
                raise ValueError('E35には複数着弾・死亡台帳・死亡保持（本番構成）が必要')
            from src.exchange_post_counter_bound import PostCounterDeathBound
            self._landing_projection.post_counter_bound = PostCounterDeathBound(early_exit=early_exit)

    def _initialize_origin_guard(self, enabled: bool, prefire_snapshot: bool,
                                 match_gate: Any) -> None:
        """E34の起点ガードを、発火前観測と試合範囲があるときだけ有効にする。"""
        self._origin_guard = None
        if enabled:
            from src.exchange_prefire_origin import PrefireOriginGuard
            if not prefire_snapshot or match_gate is None:
                raise ValueError('E34には--prefire-snapshotと既存試合範囲が必要')
            self._origin_guard = PrefireOriginGuard(match_gate)

    def _initialize_prefire(self, prefire_candidates: bool, prefire_snapshot: bool = False,
                            hidden_row_belief: bool = False, prefire_stage_timeout: bool = False,
                            prefire_stage_timeout_only: bool = False) -> None:
        """新候補の入力と死亡専用台帳を、明示ONのときだけ追加する。"""
        self._live_count_key: tuple | None = None
        self._feature_cache: dict[tuple, np.ndarray] = {}
        if prefire_stage_timeout and prefire_stage_timeout_only:
            raise ValueError('段タイムアウトの二つのフラグは同時指定できない')
        if (prefire_stage_timeout or prefire_stage_timeout_only) and not hidden_row_belief:
            raise ValueError('段タイムアウトには--hidden-row-beliefが必要')
        if hidden_row_belief and not prefire_snapshot:
            raise ValueError('--hidden-row-beliefには--prefire-snapshotが必要')
        from src.exchange_prefire_candidates import PrefireCandidates
        self._prefire = PrefireCandidates(self._landing_projection.simulator) if prefire_candidates else None
        if prefire_snapshot:
            if prefire_candidates:
                raise ValueError('発火前観測と配置列挙は同時指定できない')
            from src.exchange_prefire_snapshot import PrefireSnapshot
            self._prefire = PrefireSnapshot(self._landing_projection.simulator)
            if hidden_row_belief:
                from src.exchange_hidden_row_belief import HiddenRowPrefire
                self._prefire = HiddenRowPrefire(self._landing_projection.simulator)
                if prefire_stage_timeout or prefire_stage_timeout_only:
                    from src.exchange_prefire_stage_timeout import StageTimeoutPrefire
                    self._prefire = StageTimeoutPrefire(self._landing_projection.simulator,
                                                       timeout_only=prefire_stage_timeout_only)
                self.tracker.hidden_row_belief = self._prefire
        self._landing_projection.prefire = self._prefire
        if prefire_candidates:
            from src.exchange_landing_safety import LandingStateSafety
            self._landing_projection.death_only_inputs = True
            if self._landing_projection.safety is None:
                self._landing_projection.safety = LandingStateSafety(False, False, False)

    def _initialize_prediction_guards(self, death_candidate_guard: bool,
                                      death_formula_guard: bool, midchain_completion: bool,
                                      hidden_row_death: bool = False,
                                      midchain_single_observation: bool = False) -> None:
        """既定OFFの追加検証器をまとめて初期化する。"""
        from src.exchange_event_death_candidate import DeathCandidateGate
        self._candidate_gate = DeathCandidateGate(self._landing_projection.simulator) if death_candidate_guard else None
        from src.exchange_event_death_formula import DeathFormulaGuard
        self._formula_guard = DeathFormulaGuard() if death_formula_guard else None
        from src.exchange_midchain_completion import MidchainCompletion
        self._midchain = MidchainCompletion(self._landing_projection.simulator,
            single_observation=midchain_single_observation) if midchain_completion else None
        if self._midchain is not None:
            self._landing_projection.midchain = self._midchain
        from src.exchange_hidden_row_death import HiddenRowDeathCompletion
        self._hidden_death = HiddenRowDeathCompletion(self._landing_projection.simulator,
            single_observation=midchain_single_observation) if hidden_row_death else None
        self._landing_projection.hidden_death = self._hidden_death

    def update(self, result: Any, snapshot: Any, finalization: Any,
               t_sec: float, game_idx: int,
               formula_totals: tuple[float | None, float | None] = (None, None),
               displayed_scores: tuple[float | None, float | None] | None = None,
               formula_visible: tuple[bool, bool] = (False, False)) -> None:
        """発火→両側終了/確定→S3→着地後G_feの順で一括更新する。"""
        sides = (result.p1, result.p2)
        if self._game != game_idx:
            self._reset(game_idx, t_sec)
        if self._landing_projection.post_counter_bound is not None:
            self._landing_projection.post_counter_bound.observe(result, game_idx)
        if self._origin_guard is not None:
            self._origin_guard.observe(sides, t_sec)
        self._observe_prefire(sides, t_sec, game_idx)
        self._observe_guards(result, t_sec, game_idx, displayed_scores, formula_visible)
        if self._e16 is not None and self._e16.before(self, result, t_sec):
            self._e16.apply(self, result, snapshot, t_sec)
            self._hold_confirmed_death(t_sec)
            return
        self.tracker.begin_frame()
        self._observe_placements(sides, displayed_scores, t_sec)
        triggers = tuple(s.chain_event.trigger_sec if s.chain_event else None for s in sides)
        fresh = self._changed_chains(sides, triggers, t_sec)
        self._deliver_fires(result, snapshot, t_sec, triggers, fresh)
        for idx, (label, visible) in enumerate(zip(SIDE_LABELS, formula_visible)):
            if visible:
                self._last_formula[idx] = t_sec
                self.tracker.activity(label, t_sec)
        self._observe_signals(result, snapshot, finalization, t_sec, formula_visible)
        self._observe_scores(sides, t_sec, formula_totals, displayed_scores)
        self._observe_predictions(result, t_sec)
        self._remember(sides, snapshot, t_sec)
        if self._e16 is not None:
            self._e16.observe(sides, t_sec)
        self._refresh_features(snapshot, t_sec)
        self.tracker.confirm_frame_inputs(t_sec)
        self.tracker.finish_frame(t_sec)
        if self._midchain is not None:
            self._mark_midchain_prediction(t_sec)
        self._landing_projection.update(self, result, snapshot, t_sec)
        if self._prefire is not None:
            self._mark_prefire_prediction(t_sec)
        stable = [s.state == BoardState.STABLE for s in sides]
        settled = any(stable) if self._per_side_settled else all(stable)
        if settled and all(self._history):
            self._static(snapshot, t_sec)
        self._previous = tuple(s.state for s in sides)
        if self._e16 is not None:
            self._e16.apply(self, result, snapshot, t_sec)

    def _observe_prefire(self, sides: tuple, stamp: float, game: int) -> None:
        """予測入力を復元してから、明示ONの因果履歴を更新する。"""
        if self._prefire is not None:
            self._prefire.observe_colors(sides)
            if hasattr(self._prefire, 'observe_history'):
                self._prefire.observe_history(sides, stamp, game)

    def _observe_predictions(self, result: Any, stamp: float) -> None:
        """既存途中予測を更新してから、発火候補の平均を予測層へ反映する。"""
        for engine in (self._midchain, self._hidden_death, self._prefire):
            if engine is not None:
                engine.observe(self, result, stamp)

    def _mark_midchain_prediction(self, stamp: float) -> None:
        """途中完走盤面を使うS3出力だけに、予測層由来を記録する。"""
        record = self.tracker.current
        if record is None:
            return
        provenance = self._midchain.provenance(self._game, record.chains)
        if not provenance:
            return
        for value in reversed(record.values):
            if value['t_sec'] != stamp:
                break
            if value['source'].startswith('S3'):
                value['midchain_prediction'] = provenance

    def _mark_prefire_prediction(self, stamp: float) -> None:
        """S3・仮想着弾・死亡に候補由来を付け、現在層へ混入させない。"""
        record = self.tracker.current or self._landing_projection.death_record
        if record is None:
            return
        provenance = self._prefire.provenance(record.chains)
        for value in reversed(record.values):
            if value['t_sec'] != stamp:
                break
            if provenance and (value['source'].startswith('S3') or value['source'] == 'unavoidable_death'):
                value['prefire_prediction'] = provenance

    def _reset(self, game_idx: int, t_sec: float) -> None:
        """試合内の参照履歴と信号基準をまとめて初期化する。"""
        if self._origin_guard is not None:
            self._origin_guard.reset()
        if self._landing_projection.safety is not None:
            self._landing_projection.safety.reset()
        if self._midchain is not None:
            self._midchain.reset()
        if self._hidden_death is not None:
            self._hidden_death.reset()
        if self._prefire is not None:
            self._prefire.reset()
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
        if self._candidate_gate is not None:
            self._candidate_gate.reset()
        if self._formula_guard is not None:
            self._formula_guard.reset()
        if self._landing_projection.hands_observation is not None:
            self._landing_projection.hands_observation.reset()
        if self._e16 is not None:
            self._e16.reset()

    def _observe_guards(self, result: Any, stamp: float, game: int,
                        scores: tuple | None, visible: tuple) -> None:
        """物理観測を発火処理より先に渡し、死亡確定時には保留を破棄する。"""
        sides = (result.p1, result.p2)
        if self._candidate_gate is not None:
            self._candidate_gate.observe(sides, getattr(result, 'confirmed_dead_sides', ()), stamp, game)
        if self._formula_guard is not None:
            self._formula_guard.observe(result, stamp, game, scores, visible)
        if self._landing_projection.hands_observation is not None:
            self._landing_projection.hands_observation.observe(sides, stamp)

    def _deliver_fires(self, result: Any, snapshot: Any, stamp: float,
                       triggers: tuple, fresh: list[tuple[int, float]]) -> None:
        """保留中だけ通知を検査し、解除時には元の通知順で既存発火処理へ戻す。"""
        if self._formula_guard is not None:
            self._deliver_formula_fires(result, snapshot, stamp, triggers, fresh)
            return
        gate = self._candidate_gate
        if gate is None or not (any(gate.candidates) or any(gate.pending)):
            if fresh:
                self._fire(result, snapshot, stamp, triggers, fresh)
            return
        indices = {idx for idx, _ in fresh}
        sides = (result.p1, result.p2)
        for idx, side in enumerate(sides):
            key = self._chain_keys[idx]
            if key is not None and side.chain_event is not None and key[0] == side.chain_event.trigger_sec:
                gate.accepted[idx].add(key[0])  # 既に受理済みの連鎖の段更新は新発火ではない。
        ready = [gate.notifications(i, s.chain_event if i in indices else None, stamp)
                 for i, s in enumerate(sides)]
        self._fire_notifications(result, snapshot, stamp, ready)

    def _deliver_formula_fires(self, result: Any, snapshot: Any, stamp: float,
                               triggers: tuple, fresh: list[tuple[int, float]]) -> None:
        """E22の対象外はtriggersも含め元の呼出しをそのまま維持する。"""
        indices, sides = {i for i, _ in fresh}, (result.p1, result.p2)
        expected = [[s.chain_event] if i in indices else [] for i, s in enumerate(sides)]
        ready = [self._formula_guard.notifications(i, s.chain_event if i in indices else None,
                 self._history[i], self._chain_keys[i]) for i, s in enumerate(sides)]
        unchanged = all(len(a) == len(b) and all(x is y for x, y in zip(a, b))
                        for a, b in zip(ready, expected))
        if unchanged:
            if fresh:
                self._fire(result, snapshot, stamp, triggers, fresh)
        else:
            self._fire_notifications(result, snapshot, stamp, ready)

    def _fire_notifications(self, result: Any, snapshot: Any, stamp: float, ready: list) -> None:
        """元の入力を変更せず、保留から復帰した通知を順序どおり処理する。"""
        for pair in zip_longest(*ready):
            saved = _replaced(result, p1=_replaced(result.p1, chain_event=pair[0]),
                              p2=_replaced(result.p2, chain_event=pair[1]))
            selected = [(i, event.trigger_sec) for i, event in enumerate(pair) if event is not None]
            stamps = tuple(e.trigger_sec if e is not None else None for e in pair)
            self._fire(saved, snapshot, stamp, stamps, selected)

    def _hold_confirmed_death(self, t_sec: float) -> None:
        """死亡後の更新拒否に入る前の予測値を凍結せず、境界まで確定表示する。"""
        if not self._confirmed_death_hold:
            return
        from src.exchange_event_terminal import confirmed_winner_probability
        probability = confirmed_winner_probability(self._e16.dead_sides)
        if probability is None:
            return
        if self.tracker.source != "confirmed_death":
            record = next((r for r in reversed(self.tracker.records) if r.game_idx == self._game), None)
            if record is not None:
                record.values.append(dict(source="confirmed_death", t_sec=t_sec, p1=probability,
                                          dead_sides=sorted(self._e16.dead_sides)))
        self.tracker.source, self.tracker.probability = "confirmed_death", probability

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
        if (self._e16 is not None and self._e16.sync_enabled
                and any(s.accepted is None for s in self._e16.sync)):
            self.tracker.missing_input("E16_waiting_initial_count_pair", t_sec, "S1", triggers)
            return
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
        """既存起点予測を維持し、明示ONだけ欠落起点の候補と整合証拠を保持する。"""
        history = self._history[idx]
        if self._origin_guard is not None:
            event = self._origin_guard.select(chain, event, idx, self._game)
            history = self._origin_guard.history[idx]
        self._predict_completion_base(chain, event, idx, history)
        if self._landing_projection.safety is not None:
            self._landing_projection.safety.recovery.seed(
                chain, event, history, self._landing_projection.simulator,
                allow_recovery=self._landing_projection.safety.recovery_enabled)
        if self._prefire is not None:
            self._prefire.seed(chain, event, history, self._game, self.tracker._score_elapsed)

    def _predict_completion_base(self, chain: Any, event: Any, idx: int,
                                 history: list | None = None) -> None:
        """決着先読みと同じ完走シミュレーションを発火時に一度だけ行う。"""
        board = getattr(event, "before_board", None)
        if board is None:
            saved = next((s for s in reversed(self._history[idx] if history is None else history)
                          if s.t_sec < chain.trigger_sec), None)
            board = saved.board if saved is not None else None
        if board is None:
            return
        if self._e16 is not None and np.any(board._grid == COLOR_UNKNOWN):
            self._e16.unknown.add(chain.chain_id)
        try:
            result = self._landing_projection.simulator.simulate(board)
        except (ValueError, TypeError, FloatingPointError):
            return
        chain.predicted_final_score = float(calculate_chain_score(result).total_score)
        chain.predicted_chain_count = result.chain_count
        if result.chain_count > 0 and not np.any(board._grid == COLOR_UNKNOWN):
            chain.predicted_final_board = result.final_board._grid.tolist()
        # 起点盤面を持たない通知では、保存された既存シミュ結果も再用する。
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
        latest_sides = self._count_inputs()
        if latest_sides is None:
            return
        for idx, latest in enumerate(latest_sides):
            history = self._history[idx]
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

    def _count_inputs(self) -> list[ConfirmedSide] | None:
        """新フラグONだけ、手番が揃った確定盤面/NEXTをcountへ渡す。"""
        if self._e16 is None or not self._e16.sync_enabled:
            return [h[-1] for h in self._history]
        synced = [s.accepted for s in self._e16.sync]
        if any(s is None for s in synced):
            return None
        return [ConfirmedSide(s.t_sec, Board.from_list(s.grid.tolist()), s.queue) for s in synced]

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
