"""E16の現在層・予測層を分離し、観測と矛盾した予測は採用しない。"""
from __future__ import annotations

from typing import Any
import numpy as np
from src.board import COLOR_UNKNOWN
from src.board_state_machine import BoardState
from src.exchange_event_evaluator import evaluate_exchange_event
from src.exchange_event_landing import logit_mean
from src.exchange_event_sync import CountTurnSynchronizer

SIDES = ("1P", "2P")
LAYER_COLUMNS = ("p1_current", "p1_prediction", "p1_layer_combined", "prediction_reasons")


class ExchangeEvaluationLayers:
    """合成を重ねず、モデル予測と現在の確定盤面評価を別々に保持する。"""
    def __init__(self, sync_enabled: bool = True, death_enabled: bool = True,
                 layer_enabled: bool = True, completion_fix: bool = False) -> None:
        self.sync_enabled, self.death_enabled = sync_enabled, death_enabled
        self.layer_enabled, self.completion_fix = layer_enabled, completion_fix
        self.reset()

    def reset(self) -> None:
        """試合を越えて死亡・同期・信頼性の判定を持ち越さない。"""
        self.sync = [CountTurnSynchronizer(), CountTurnSynchronizer()]
        self.prediction: float | None = None
        self.prediction_source = "waiting_confirmed"
        self.current: float | None = None
        self.current_key: tuple | None = None
        self.dead_sides: set[str] = set()
        self.checked: set[int] = set()
        self.mismatches: set[int] = set()
        self.unknown: set[int] = set()
        from src.exchange_event_completion_check import CompletionVerifier
        self.completion = CompletionVerifier()

    def before(self, overlay: Any, result: Any, stamp: float) -> bool:
        """前回合成値を除き、死亡確認済みなら以後の認識演出を評価へ流さない。"""
        tracker = overlay.tracker
        if self.layer_enabled and self.prediction is not None:
            tracker.probability, tracker.source = self.prediction, self.prediction_source
        self.dead_sides.update(getattr(result, "confirmed_dead_sides", ()))
        if not self.death_enabled:
            return False
        if not self.dead_sides:
            return False
        for idx, side in enumerate((result.p1, result.p2)):
            if side.chain_event is not None:
                tracker.missing_input("E16_fire_after_observed_death", stamp, "S1",
                                      (SIDES[idx], side.chain_event.trigger_sec))
        return True

    def observe(self, sides: tuple, stamp: float) -> None:
        """確定盤面とNEXTを同期器へ渡し、推論・学習で同じ規則を使う。"""
        if not self.sync_enabled:
            return
        for sync, side in zip(self.sync, sides):
            grid = side.confirmed_board._grid if side.confirmed_board is not None else None
            queue = np.array([*(side.next_pair or (0, 0)), *(side.dnext_pair or (0, 0))])
            sync.observe(grid, queue, stamp, side.state == BoardState.STABLE,
                         bool(getattr(side, "next_slide_motion", False)),
                         side.state in (BoardState.CHAIN, BoardState.GRAVITY_SETTLE))

    def _current(self, overlay: Any, snapshot: Any, stamp: float) -> float | None:
        """仮想着弾も連鎖完走予測も使わず、各側の最新確定盤面だけを評価する。"""
        if not all(overlay._history) or overlay._m0 is None:
            return None
        latest = [h[-1] for h in overlay._history]
        elapsed = stamp-overlay._start
        phase = int(np.searchsorted(overlay.tracker.models.elapsed_thresholds, elapsed, side="left"))
        key = (tuple((s.board._grid.tobytes(), s.queue.tobytes()) for s in latest),
               snapshot.net_balance_capped, snapshot.forecast_p1, phase)
        if key != self.current_key:
            m0 = overlay._m0(np.stack([s.board._grid for s in latest]), np.stack([s.queue for s in latest]))
            event = overlay._build_static(tuple(s.board for s in latest), snapshot, elapsed, m0)
            self.current = evaluate_exchange_event(event, overlay.tracker.models)
            self.current_key = key
        return self.current

    def _reasons(self, overlay: Any, result: Any) -> list[str]:
        """超過・UNKNOWN・事後盤面不一致を、検出されたフレームから反映する。"""
        tracker, reasons = overlay.tracker, set()
        if any(h and np.any(h[-1].board._grid == COLOR_UNKNOWN) for h in overlay._history):
            reasons.add("unknown_board")
        record = tracker.current or overlay._landing_projection.death_record
        for chain in record.chains if record else ():
            if chain.chain_id in self.unknown:
                reasons.add("unknown_prediction_origin")
            observed = max(chain.formula_total or 0, (chain.score_delta or 0)-chain.drop_bonus_score)
            if observed > (chain.predicted_final_score or 0):
                reasons.add("observed_score_exceeds_prediction")
            if self._observed_count(tracker, chain.chain_id) > (chain.predicted_chain_count or 0):
                reasons.add("observed_chain_exceeds_prediction")
            self._check_completion(overlay, result, chain)
            if chain.chain_id in self.mismatches:
                reasons.add("prediction_board_mismatch")
        if self.dead_sides:
            reasons.add("observed_death")
        return sorted(reasons)

    def _observed_count(self, tracker: Any, identity: int) -> int:
        """resolverの別名・終了済み連鎖でも、式で実測した段数を失わない。"""
        chains = (*tracker.resolver.active(), *tracker.resolver.resolved())
        return max((c.step_count for c in chains if c.growth_observed
                    and tracker._chain_aliases.get(c.chain_id, c.chain_id) == identity), default=0)

    def _check_completion(self, overlay: Any, result: Any, chain: Any) -> None:
        """E16旧照合を保存し、独立フラグで終了・着手境界を限定した照合へ切り替える。"""
        if self.completion_fix:
            self.mismatches.discard(chain.chain_id)
            if self.completion.check(overlay, result, chain):
                self.mismatches.add(chain.chain_id)
            return
        if chain.chain_id in self.checked or chain.end_signal_sec is None or chain.predicted_final_board is None:
            return
        idx = SIDES.index(chain.side)
        side = (result.p1, result.p2)[idx]
        history = overlay._history[idx]
        if side.state != BoardState.STABLE or not history or history[-1].t_sec < chain.end_signal_sec:
            return
        self.checked.add(chain.chain_id)
        if not np.array_equal(history[-1].board._grid, chain.predicted_final_board):
            self.mismatches.add(chain.chain_id)

    def apply(self, overlay: Any, result: Any, snapshot: Any, stamp: float) -> None:
        """現在層と予測層を等重みlogit平均し、不信頼時は現在層へ戻す。"""
        if not self.layer_enabled:
            return
        overlay._completion_stamp = stamp
        tracker = overlay.tracker
        self.prediction, self.prediction_source = tracker.probability, tracker.source
        current = self._current(overlay, snapshot, stamp)
        reasons = self._reasons(overlay, result)
        predicted = self.prediction if self.prediction_source != "G_fe" else None
        if current is None:
            combined = None
            reasons.append("missing_current_board")
        elif predicted is None or reasons:
            combined = current
        else:
            combined = logit_mean(current, predicted)
        record = tracker.current or overlay._landing_projection.death_record
        row = dict(t_sec=stamp, game_idx=overlay._game, exchange_id=record.exchange_id if record else None,
            p1_current=current, p1_prediction=predicted, p1_layer_combined=combined,
            prediction_reasons="|".join(reasons), prediction_source=self.prediction_source,
            confirmed_dead_sides=sorted(self.dead_sides),
            sync_reasons=[s.reason for s in self.sync],
            sync_sec=[s.accepted.t_sec if s.accepted else None for s in self.sync])
        tracker.layer_eval = row
        tracker.layer_rows.append(row)
        tracker.probability = combined
        if reasons:
            tracker.source = "E16_current"
