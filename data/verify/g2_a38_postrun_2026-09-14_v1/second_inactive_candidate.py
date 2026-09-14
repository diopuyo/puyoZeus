"""2P同世代inactiveを原Witness tapから終了票へ分離する。実loader未採用。"""
from __future__ import annotations
from dataclasses import asdict
import json
import sys
from typing import Any
import terminal_boundary as C


def mode_factory(original: Any) -> Any:
    """既存session_connection.mode_classの生成時だけ適用する候補。"""
    def create(base: type, session: Any) -> type:
        selected = original(base, session)
        class Mode(selected):
            def observe(self, item: dict, result: Any, error: Any) -> dict:
                pair = session._arrival_bindings.get(id(self))
                if pair is None:
                    return super().observe(item, result, error)
                C.require(pair[0] is self, 'second_inactive_binding_owner')
                value = pair[1]
                ended = getattr(value, '_inactive_end', None)
                frame = item.get('frame')
                inactive = frame is not None and frame.f_locals.get('is_active') is False
                if ended is None and not inactive:
                    return super().observe(item, result, error)
                if ended is None:
                    ended = End(value, session)
                    value._inactive_end = ended
                    session.stack.push(ended.close)
                return ended.observe(item, error)
        return Mode
    return create


class End:
    def __init__(self, value: Any, session: Any) -> None:
        self.value, self.session = value, session
        self.ledger, self.current = value.ledger, None
        self.last, self.rows = value.ledger.clock, 0
        self.receipt, self.stream, self.failure = None, None, None
        self.closed = False

    def source(self, item: dict, error: Any) -> dict:
        if error is not None:
            raise error
        value, mode = self.value, self.value.mode
        journal, pipe = value.tap.journal, mode.connection.recovery.pipe
        C.require(not self.closed and self.failure is None and value.failure is None
            and not value.closed and mode.error is None and not journal.errors, 'second_inactive_lifetime')
        C.require(mode.connection.recovery.journal is journal and item['pipe'] is pipe
            and self.session._arrival_bindings[id(mode)] == (mode, value), 'second_inactive_owner')
        frame, scope = item['frame'], self.ledger.scope
        C.require(frame is not None and frame.f_code in journal.codes
            and frame.f_locals.get('is_active') is False, 'second_inactive_original_step')
        tick = item['scope']['frame_idx']
        C.require(item['scope']['side'] == '2P' and tick == self.last+C.STRIDE
            and tick <= self.ledger.deadline and frame.f_locals['frame_idx'] == tick
            and frame.f_locals['time_sec'] == tick/C.FPS, 'second_inactive_step_clock')
        actual = mode.connection.recovery.evidence.scope(mode.connection.recovery.factory, pipe)
        C.require(actual == scope and mode.connection.binding.scope == scope, 'second_inactive_same_scope')
        C.require(value.ledger is self.ledger and mode.native.last_frame == self.ledger.clock
            and not mode.native.pending and value.N.extract(item) is None, 'second_inactive_frozen_native')
        rows = value.tap.between(self.last, tick, scope)
        C.require(len(rows) == 1, 'second_inactive_source_count')
        row = rows[0]
        C.require(row['frame_idx'] == tick and row['time_sec'] == tick/C.FPS
            and row['active'] is False and row['status'] == 'returned'
            and row['returned_none'] is True, 'second_inactive_source_status')
        return row

    def prepare(self, row: dict) -> None:
        value, mode = self.value, self.value.mode
        context = self.session.state['provisional_context_observer'].rows[-1]
        end = C.end_from_context(self.ledger, context)
        receipt = C.prepare(value.L, self.ledger, row, end)
        terminal = value.terminal
        C.require(terminal is not None and terminal.committed and not terminal.closed,
            'second_inactive_terminal_uncommitted')
        current = mode.connection.registry.current(mode.connection.binding)
        C.require(current.scope == self.ledger.scope, 'second_inactive_current_scope')
        self.stream = (self.session._arrival_output/'SECOND_INACTIVE_END.jsonl').open('x', encoding='utf-8')
        self.receipt, self.current = receipt, current

    def observe(self, item: dict, error: Any) -> dict:
        try:
            row = self.source(item, error)
            if self.receipt is None:
                self.prepare(row)
            else:
                C.require(not row['added_occurrence_tokens'] and not row['fifo_occurrence_tokens']
                    and row['discarded_tokens'] == self.receipt['unacknowledged_terminal_tokens']
                    and not row['before']['pending_tsumo']
                    and not row['after']['pending_tsumo'] and not row['after']['tsumo_count']
                    and not row['update_begin_accounting']['pending_tsumo'], 'second_inactive_tail_fifo')
            connection = self.value.mode.connection
            C.require(connection.registry.current(connection.binding) is self.current
                and self.receipt['last_live_ledger'] == asdict(self.value.ledger), 'second_inactive_immutable')
            packet = dict(source_fields=row, completed_call_token=item['token'],
                terminal_receipt=self.receipt if self.rows == 0 else None,
                generation_after=self.value.tap.journal.scope(item['pipe'], '2P',
                    row['frame_idx'], row['time_sec'])['generation'], quality_gate_clear=False)
            self.stream.write(json.dumps(packet, allow_nan=False)+'\n')
            self.stream.flush()
            self.last, self.rows = row['frame_idx'], self.rows+1
            return dict(kind='probabilistic_tracking_observation', terminal_scope_observation=True,
                scope=item['scope'], journal_token=item['token'], provisional_update=False,
                reason='match_ended_scope_frozen', quality_gate_clear=False)
        except BaseException as failure:
            self.failure = failure
            raise

    def close(self, kind: Any, body: Any, trace: Any) -> bool:
        failure = self.failure
        try:
            if self.stream is not None:
                self.stream.close()
        except BaseException as caught:
            failure = failure or caught
        finally:
            self.closed = self.stream is None or self.stream.closed
        path = self.session._arrival_output/'SECOND_INACTIVE_END_STATUS.json'
        try:
            with path.open('x', encoding='utf-8') as stream:
                json.dump(dict(rows=self.rows, last_frame=self.last, closed=self.closed,
                    error=None if failure is None else repr(failure),
                    original_body=None if body is None else repr(body),
                    receipt=self.receipt, quality_gate_clear=False), stream)
        except BaseException as caught:
            failure = failure or caught
        if failure is not None:
            if body is None:
                raise failure
            print('SECOND_INACTIVE_CLOSE_ERROR=' + repr(failure), file=sys.stderr)
        return False
