"""2P prefixに到来/ACKの私有状態を接続する。元数学・FIFOは変更しない。"""
from __future__ import annotations
from dataclasses import asdict
import json
from pathlib import Path
import sys
from typing import Any
import second_arrival_state as A


def original_application(module: Any, ledger: Any, mode: Any, row: dict) -> Any:
    """元prefixの実commit直後だけ、対応する物理反映を私有台帳へ写す。"""
    receipt = row.get('transition')
    if not isinstance(receipt, dict) or receipt.get('kind') != 'second_consumed_prefix/v1':
        return ledger
    check, lane = module.require, mode.prefix.lane
    check(row.get('provisional_update') is True and mode.applied[-1] is receipt,
          'second_applied_receipt_owner')
    current = mode.connection.registry.current(mode.connection.binding)
    check(current is lane.current and current.scope == ledger.scope,
          'second_applied_registry_owner')
    before, after = receipt['before_prefix'], receipt['after_prefix']
    check(before == len(ledger.applied) and before < after == lane.committed,
          'second_applied_prefix')
    tokens = tuple(a.token for a in ledger.arrivals[before:after])
    check(tokens == tuple(receipt['applied_tokens'])
          and tuple(a.token for a in lane.arrivals[:after]) == ledger.applied + tokens,
          'second_applied_arrival_identity')
    check(after <= len(ledger.acknowledgements)
          and all((a.token, a.pair) == (b.token, b.pair)
                  for a, b in zip(lane.arrivals[:after], ledger.arrivals[:after])),
          'second_applied_without_ack')
    check(current.tokens == lane.initial.tokens + ledger.applied + tokens
          and current.frame == receipt['applied_frame'] == ledger.clock,
          'second_applied_clock_tokens')
    check(receipt['state'] == mode.prefix.serializer.encode(current), 'second_applied_state')
    check(not set(tokens) & {event.occurrence_token for event in mode.native.pending},
          'second_applied_native_not_removed')
    return module.applied(ledger, tokens, current.frame)


class Binding:
    def __init__(self, mode: Any, tap: Any, ledger_module: Any,
                 native_module: Any, path: Path) -> None:
        self.mode, self.tap, self.L, self.N = mode, tap, ledger_module, native_module
        initial = mode.prefix.lane.initial
        self.L.require(initial.scope[-1] == '2P', 'second_binding_side')
        self.ledger = self.L.Ledger(initial.scope, initial.frame, initial.deadline, initial.frame)
        self.closed = False
        self.failure: BaseException | None = None
        self.stream = path.open('x', encoding='utf-8')
        self.last_counts = None

    def observe(self, item: dict) -> None:
        """呼び手は元Native.observe完了後。pendingは検証済み受領票の参照だけ。"""
        try:
            mode, frame = self.mode, item['scope']['frame_idx']
            self.L.require(not self.closed and self.failure is None, 'second_binding_unusable')
            self.L.require(type(mode.native) is self.N.Recorder
                and mode.native.connection is mode.connection and mode.native.last_frame == frame
                and item['token'] in mode.native.seen_calls,
                'second_binding_native_unverified')
            self.L.require(mode.connection.binding.scope == self.ledger.scope,
                           'second_binding_scope_changed')
            self.L.require(mode.connection.recovery.journal is self.tap.journal,
                           'second_binding_journal_changed')
            known = {ack.token for ack in self.ledger.acknowledgements}
            events = tuple(event for event in mode.native.pending if event.occurrence_token not in known)
            rows = self.tap.between(self.ledger.clock, frame, self.ledger.scope)
            following = A.advance(self.L, self.ledger, rows, events, frame)
            counts = (len(following.arrivals), len(following.acknowledgements))
            if counts != self.last_counts:
                self.save(dict(kind='arrival_ack', frame=frame, state=asdict(following),
                               native_step_verified=True))
            self.ledger, self.last_counts = following, counts
        except BaseException as error:
            self.failure = error
            raise

    def original_applied(self, row: dict) -> None:
        """保存成功後に私有状態を進める。元Registry/FIFOを再更新しない。"""
        try:
            self.L.require(not self.closed and self.failure is None, 'second_binding_unusable')
            following = original_application(self.L, self.ledger, self.mode, row)
            if following is self.ledger:
                return
            self.save(dict(kind='original_application', frame=following.clock,
                state=asdict(following), original_receipt=row['transition'],
                original_physical_commit_verified=True))
            self.ledger = following
        except BaseException as error:
            self.failure = error
            raise

    def save(self, row: dict) -> None:
        self.stream.write(json.dumps(row | dict(original_fifo_changed=False,
            physical_applied=False, quality_gate_clear=False), allow_nan=False) + '\n')
        self.stream.flush()

    def close(self, kind: Any = None, body: Any = None, trace: Any = None) -> bool:
        if self.closed:
            return False
        failure = self.failure
        try:
            self.save(dict(kind='finish', state=asdict(self.ledger),
                error=None if self.failure is None else repr(self.failure)))
        except BaseException as error:
            failure = failure or error
        try:
            self.stream.close()
        except BaseException as error:
            failure = failure or error
        self.closed = True
        if failure is not None:
            if body is None:
                raise failure
            print('SECOND_ARRIVAL_CLOSE_ERROR=' + repr(failure), file=sys.stderr)
        return False
