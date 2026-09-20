"""原Jの消費・原origin観測・次STABLEを、確率状態の一回更新へ接続する候補。"""
from __future__ import annotations
from dataclasses import asdict
import math
from types import FunctionType, SimpleNamespace
from typing import Any
import tracking_mode as M
import transition_candidates as T
import serialization as S

B = T.B
FPS = 60


class Mode(M.Mode):
    def __init__(self, connection: Any, state: dict[str, Any], stream: Any) -> None:
        super().__init__(connection, state, stream)
        self.origins: dict[str, dict[str, Any]] = {}
        self.origin_ids: dict[tuple[int, float], str] = {}
        self.applied: list[dict[str, Any]] = []
        self.prior = T.H.uncalibrated_uniform('原列挙配置をworld内で未較正一様と仮定。着手頻度ではない。')

    def origin_token(self, item: Any) -> str:
        B.require(self.native is not None and len(self.native.pending) <= 1, 'physical_pending_ambiguous')
        if self.native.pending:
            return self.native.pending[0].occurrence_token
        r = self.connection.recovery
        side = self.connection.binding.scope[-1]
        owner = r.provider.owner(r.pipe, side, item['epoch'])
        B.require(owner['tokens'] and type(owner['tokens'][0]) is str, 'physical_origin_without_owned_NEXT')
        return owner['tokens'][0]

    def capture_origin(self, item: Any) -> None:
        found = [event['active_origin'] for event in item['events'] if event.get('active_origin') is not None]
        for origin in found:
            trigger = origin['trigger_sec']
            B.require(type(trigger) in (float, int) and math.isfinite(trigger), 'physical_origin_clock')
            identity = (origin['object_id'], float(trigger))
            if identity in self.origin_ids:
                saved = self.origins[self.origin_ids[identity]]
                B.require(origin['before_board'] is not None and tuple(map(tuple, origin['before_board']['grid']))
                          == saved['grid'], 'physical_origin_mutated')
                continue
            token = self.origin_token(item)
            B.require(token not in self.origins, 'physical_duplicate_origin_for_hand')
            basis = self.connection.observer.gate.candidate
            B.require(type(trigger) in (float, int) and math.isfinite(trigger)
                      and basis.frame / FPS <= trigger <= item['scope']['time_sec'], 'physical_origin_clock')
            B.require(origin['before_board'] is not None, 'physical_origin_board_missing')
            board = B.Board.from_dict({'grid': origin['before_board']['grid']})
            self.origins[token] = dict(grid=B.grid(board), source_call_token=item['token'],
                first_observed_frame=item['scope']['frame_idx'], object_id=origin['object_id'],
                estimated_chain_count=origin['chain_count'], creation_call_witnessed=False)
            self.origin_ids[identity] = token

    def stable(self, item: Any, result: Any) -> tuple[Any, str | None]:
        r, side = self.connection.recovery, self.connection.binding.scope[-1]
        local = item['frame'].f_locals
        signals = local['signals']
        view = SimpleNamespace(frame=item['scope']['frame_idx'], clock=item['scope']['time_sec'])
        if result is None or result.state.value != 'stable': return None, 'not_STABLE'
        if signals.is_match_active is not True or signals.effect_gate_window_active is not False:
            return None, 'inactive_or_effect_window'
        if not r.provider.no_origin(r.pipe, side, view): return None, 'active_origin'
        grace = getattr(r.pipe, '_landing_grace_' + side.lower())
        if grace is not None and view.clock < grace[2]: return None, 'landing_grace'
        raw, _ = r.provider.raw(r.pipe, side, view)
        board = B.Board.from_dict({'grid': raw})
        observed = B.grid(board)[B.HIDDEN_ROWS:]
        others = (signals.cnn_board, local['sm'].context.confirmed_board, result.confirmed_board)
        if any(value is None or B.grid(value)[B.HIDDEN_ROWS:] != observed for value in others):
            return None, 'visible_channel_mismatch'
        return board, None

    def apply(self, item: Any, observed: Any) -> dict[str, Any]:
        c = self.connection
        B.require(self.native is not None and len(self.native.pending) == 1, 'physical_pending_ambiguous')
        event = self.native.pending[0]
        origin = self.origins.get(event.occurrence_token)
        value = c.registry.current(c.binding)
        frame = item['scope']['frame_idx']
        origin_board = None if origin is None else B.Board.from_dict({'grid': origin['grid']})
        # 原originの推定連鎖数は真値にしない。起源と実STABLE可視から事後分布を残す。
        def operation(current: B.Belief) -> tuple[Any, Any]:
            return T.run(current, value.scope, frame, event.occurrence_token, event.pair, observed,
                         0 if origin is None else None, self.prior, origin_observed=origin_board)
        following, report = c.registry.transition(c.recovery.factory, c.binding, value, item['token'], operation)
        self.native.pending.pop(0)  # 原FIFOではなく、検証済み消費の私有待機票だけを閉じる。
        receipt = dict(source_call_token=item['token'], occurrence_token=event.occurrence_token,
            consumed_frame=event.consumed_frame, applied_frame=frame, origin=origin,
            distribution_report=asdict(report), state=S.encode(following), physical_certified=False)
        self.applied.append(receipt)
        return receipt

    def observe(self, item: Any, result: Any, error: Any) -> dict[str, Any]:
        row = super().observe(item, result, error)
        self.capture_origin(item)
        if not self.native.pending:
            return row
        B.require(len(self.native.pending) == 1, 'physical_pending_ambiguous')
        observed, reason = self.stable(item, result)
        if reason is not None:
            return row | dict(reason=reason)
        try:
            receipt = self.apply(item, observed)
        except ValueError as failure:
            if str(failure) != 'probabilistic_scope:transition_zero_support': raise
            return row | dict(reason='stable_observation_zero_support', provisional_update=False)
        return row | dict(reason='probabilistic_transition_applied', physical_transition_applied=True,
                          pending_occurrences=[], transition=receipt, provisional_update=True)


def install(stack: Any, connection: Any, state: dict[str, Any]) -> Mode:
    return FunctionType(M.install.__code__, dict(vars(M), Mode=Mode))(stack, connection, state)
