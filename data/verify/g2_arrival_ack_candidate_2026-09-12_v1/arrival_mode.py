"""原Native検査/原STABLE資格を保持し、到来反映と原popのackを別管理する。"""
from __future__ import annotations

from dataclasses import asdict
import json
import math
from types import FunctionType
from typing import Any

import base_mode as BASE
import cascade_arrival as C
import enqueue_capture as E
import ledger as L
import stable_capture as Q

B, CORE = BASE.B, BASE.M
FPS = 60


class Mode(BASE.Mode):
    def __init__(self, connection: Any, state: dict, stream: Any) -> None:
        super().__init__(connection, state, stream)
        self.arrival_ledger: L.Ledger | None = None
        self.arrival_capture: E.Capture | None = None
        self.arrival_step: dict | None = None
        self.stable_qualification: dict | None = None

    def stable(self, item: Any, result: Any) -> tuple[Any, str | None]:
        observed, reason = super().stable(item, result)
        self.stable_qualification = None if reason is not None else Q.capture(self, item, result, observed)
        return observed, reason

    def activate(self, item: dict) -> None:
        super().activate(item)
        if self.native is not None and self.arrival_ledger is None:
            value = self.connection.registry.current(self.connection.binding)
            self.arrival_ledger = L.Ledger(value.scope, value.frame, value.deadline, value.frame)

    def acknowledge(self, row: dict) -> None:
        event = row['native_consumption']
        if event is None:
            B.require(not self.native.pending, 'arrival_unprocessed_native_ack')
            return
        B.require(len(self.native.pending) == 1 and asdict(self.native.pending[0]) == event,
                  'arrival_native_ack_identity')
        self.arrival_ledger = L.acknowledge(self.arrival_ledger, event['occurrence_token'],
            tuple(event['pair']), event['consumed_frame'], event['source_call_token'])
        self.native.pending.pop(0)  # 原FIFOではない。原popを突合済みの私有受領票だけ解放。

    def capture_origin(self, item: dict) -> None:
        if not self.basis_cascade_closed:
            # 未閉鎖基準の原origin契約は元のまま。到来をnativeへ偽装しない。
            return super().capture_origin(item)
        for event in item['events']:
            origin = event.get('active_origin')
            if origin is None:
                continue
            identity = (origin['object_id'], origin['trigger_sec'])
            if identity in self.origin_ids:
                saved = self.origins[self.origin_ids[identity]]
                B.require(origin['before_board'] is not None
                    and tuple(map(tuple, origin['before_board']['grid'])) == saved['grid'], 'arrival_origin_mutated')
                continue
            self.new_origin(item, origin, identity)

    def new_origin(self, item: dict, origin: dict, identity: tuple) -> None:
        ledger = self.arrival_ledger
        trigger = origin['trigger_sec']
        B.require(type(trigger) in (int, float) and math.isfinite(trigger)
                  and ledger.start / FPS <= trigger <= item['scope']['time_sec'], 'arrival_origin_clock')
        eligible = [a for a in ledger.arrivals[len(ledger.applied):] if a.frame / FPS <= trigger]
        B.require(len(eligible) == 1, 'arrival_origin_assignment_ambiguous')
        arrival = eligible[0]
        B.require(arrival.token not in self.origins and origin['before_board'] is not None,
                  'arrival_origin_duplicate_or_missing')
        board = B.Board.from_dict({'grid': origin['before_board']['grid']})
        self.origins[arrival.token] = dict(grid=B.grid(board), source_call_token=item['token'],
            first_observed_frame=item['scope']['frame_idx'], object_id=origin['object_id'],
            estimated_chain_count=origin['chain_count'], creation_call_witnessed=False)
        self.origin_ids[identity] = arrival.token

    def apply_arrival(self, item: dict, observed: Any) -> dict:
        c, ledger = self.connection, self.arrival_ledger
        value = c.registry.current(c.binding)
        arrival = ledger.arrivals[len(ledger.applied)]
        origin = self.origins.get(arrival.token)
        composite = self.basis_origin is not None and not self.basis_cascade_closed
        frame = item['scope']['frame_idx']
        def operation(current: Any) -> Any:
            if composite:
                return C.run(current, ledger, frame, arrival,
                    B.Board.from_dict({'grid': self.basis_origin['grid']}), observed, self.prior)
            origin_board = None if origin is None else B.Board.from_dict({'grid': origin['grid']})
            return C.T.run(current, value.scope, frame, arrival.token, arrival.pair, observed,
                           0 if origin is None else None, self.prior, origin_observed=origin_board)
        following, report = c.registry.transition(c.recovery.factory, c.binding, value, item['token'], operation)
        self.arrival_ledger = L.applied(ledger, (arrival.token,), frame)
        receipt = dict(kind='basis_cascade_with_arrival' if composite else 'arrived_hand',
            source_call_token=item['token'], applied_frame=frame, arrival=asdict(arrival), origin=origin,
            basis_origin=self.basis_origin if composite else None, state=BASE.V1.S.encode(following),
            distribution_report=asdict(report), physical_certified=False, original_fifo_changed=False)
        self.applied.append(receipt)
        if composite:
            self.basis_cascade_closed = True
        return receipt

    def progress(self, item: dict, result: Any, row: dict) -> dict:
        ledger = self.arrival_ledger
        waiting = ledger.arrivals[len(ledger.applied):]
        basis = self.basis_origin is not None and not self.basis_cascade_closed
        if not waiting and not basis:
            return row | dict(reason='no_unapplied_arrival')
        observed, reason = self.stable(item, result)  # 元raw/CNN/SM/returned/effect/grace検査。
        if reason is not None:
            return row | dict(reason='arrival:' + reason)
        try:
            receipt = self.apply_arrival(item, observed) if waiting else self.settle_basis(item, observed)
        except ValueError as failure:
            if str(failure) not in ('probabilistic_scope:observation_has_zero_support',
                                    'probabilistic_scope:transition_zero_support'):
                raise
            return row | dict(reason='arrival_observation_zero_support', provisional_update=False)
        return row | dict(reason=receipt['kind'] + '_applied', transition=receipt, provisional_update=True,
                          stable_qualification=self.stable_qualification)

    def observe(self, item: dict, result: Any, error: Any) -> dict:
        B.require(self.arrival_ledger is not None and self.arrival_capture is not None,
                  'arrival_mode_not_initialized')
        row = CORE.Mode.observe(self, item, result, error)  # 原Nativeの同call資格と消費extractを一回実行。
        ledger = self.arrival_ledger
        if item['scope']['frame_idx'] > ledger.start:
            self.arrival_step = item
            try:
                self.arrival_capture.completed(item)
            finally:
                self.arrival_step = None
        self.acknowledge(row)
        self.capture_origin(item)
        row = self.progress(item, result, row)
        self.arrival_ledger = L.advance_clock(self.arrival_ledger, item['scope']['frame_idx'])
        ledger = self.arrival_ledger
        return row | dict(pending_occurrences=[], arrival_ledger=asdict(ledger),
            unapplied_arrivals=[a.token for a in ledger.arrivals[len(ledger.applied):]],
            unacknowledged_arrivals=[a.token for a in ledger.arrivals[len(ledger.acknowledgements):]])

    def close(self) -> None:
        super().close()
        ledger = self.arrival_ledger
        with (self.state['output'] / 'ARRIVAL_LEDGER_STATUS.json').open('x') as stream:
            json.dump(dict(state=None if ledger is None else asdict(ledger),
                drained=False if ledger is None else L.drained(ledger), error=self.error,
                quality_gate_clear=False), stream, indent=2)


def install(stack: Any, connection: Any, state: dict) -> Mode:
    mode = FunctionType(CORE.install.__code__, dict(vars(CORE), Mode=Mode))(stack, connection, state)
    mode.arrival_capture = E.install(stack, mode, state['output'])
    return mode
