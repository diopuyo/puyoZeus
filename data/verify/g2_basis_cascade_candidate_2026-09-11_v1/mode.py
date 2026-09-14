"""基準自身の原連鎖と、新しいNEXT手の原連鎖を分離する接続候補。"""
from __future__ import annotations
from dataclasses import asdict
import math
from types import FunctionType
from typing import Any
import physical_tracking as P
import cascade as C

B, M, S, FPS = P.B, P.M, P.S, P.FPS


class Mode(P.Mode):
    def __init__(self, connection: Any, state: dict[str, Any], stream: Any) -> None:
        super().__init__(connection, state, stream)
        self.basis_origin: dict[str, Any] | None = None
        self.basis_cascade_closed = False

    def adopt_basis_origin(self, item: Any, origin: Any) -> None:
        c = self.connection
        value = c.registry.current(c.binding)
        trigger = origin['trigger_sec']
        B.require(type(trigger) in (float, int) and math.isfinite(trigger)
                  and value.frame / FPS <= trigger <= item['scope']['time_sec'], 'basis_origin_clock')
        B.require(origin['before_board'] is not None, 'basis_origin_missing_board')
        grid = B.grid(B.Board.from_dict({'grid': origin['before_board']['grid']}))
        B.require(all(w.grid[B.HIDDEN_ROWS:] == grid[B.HIDDEN_ROWS:] for w in value.worlds),
                  'basis_origin_visible_mismatch')
        probe = B.ChainSimulator(exclude_hidden_row_from_pop=True).simulate(
            B.Board.from_dict({'grid': value.worlds[0].grid}))
        B.require(probe.chain_count > 0, 'basis_origin_without_existing_chain')
        token = 'basis-cascade:' + c.binding.initial_call_token + ':' + item['token']
        saved = dict(grid=grid, source_call_token=item['token'], operation_token=token,
            first_observed_frame=item['scope']['frame_idx'], object_id=origin['object_id'],
            estimated_chain_count=origin['chain_count'], creation_call_witnessed=False)
        self.origins[token] = saved
        self.origin_ids[(origin['object_id'], float(trigger))] = token
        self.basis_origin = saved

    def capture_origin(self, item: Any) -> None:
        found = [e['active_origin'] for e in item['events'] if e.get('active_origin') is not None]
        for origin in found:
            identity = (origin['object_id'], origin['trigger_sec'])
            if identity in self.origin_ids:
                continue
            if not self.native.seen_occurrences and not self.basis_cascade_closed:
                B.require(self.basis_origin is None, 'basis_multiple_origins')
                self.adopt_basis_origin(item, origin)
            elif self.native.pending:
                consumed = self.native.pending[0].consumed_frame
                trigger = origin['trigger_sec']
                B.require(type(consumed) is int and type(trigger) in (float, int)
                          and math.isfinite(trigger) and trigger >= consumed / FPS,
                          'origin_precedes_native_consumption')
        super().capture_origin(item)

    def settle_basis(self, item: Any, observed: Any) -> dict[str, Any]:
        c, origin = self.connection, self.basis_origin
        value = c.registry.current(c.binding)
        def operation(current: Any) -> Any:
            return C.settle(current, value.scope, item['scope']['frame_idx'], origin['operation_token'],
                            B.Board.from_dict({'grid': origin['grid']}), observed)
        following, report = c.registry.transition(c.recovery.factory, c.binding, value, item['token'], operation)
        receipt = dict(kind='basis_cascade', source_call_token=item['token'], origin=origin,
            applied_frame=item['scope']['frame_idx'], state=S.encode(following),
            distribution_report=asdict(report), physical_certified=False, next_consumed=False)
        self.applied.append(receipt)
        self.basis_cascade_closed = True
        return receipt

    def observe(self, item: Any, result: Any, error: Any) -> dict[str, Any]:
        # native原票は一度だけ取得。基準連鎖では原FIFOも私有消費票もpopしない。
        row = M.Mode.observe(self, item, result, error)
        self.capture_origin(item)
        if self.basis_origin is None or self.basis_cascade_closed:
            return self.follow_hand(item, result, row)
        B.require(not self.native.pending, 'native_hand_before_basis_cascade_closed')
        observed, reason = self.stable(item, result)
        if reason is not None:
            return row | dict(reason='basis_cascade:' + reason)
        try:
            receipt = self.settle_basis(item, observed)
        except ValueError as failure:
            if str(failure) != 'probabilistic_scope:observation_has_zero_support': raise
            return row | dict(reason='basis_cascade_zero_support', provisional_update=False)
        return row | dict(reason='basis_cascade_applied', transition=receipt, provisional_update=True)

    def follow_hand(self, item: Any, result: Any, row: dict[str, Any]) -> dict[str, Any]:
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
        return row | dict(reason='probabilistic_transition_applied', transition=receipt,
                          pending_occurrences=[], physical_transition_applied=True, provisional_update=True)


def install(stack: Any, connection: Any, state: dict[str, Any]) -> Mode:
    return FunctionType(M.install.__code__, dict(vars(M), Mode=Mode))(stack, connection, state)
