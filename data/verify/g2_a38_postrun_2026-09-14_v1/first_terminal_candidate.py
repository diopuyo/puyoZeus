"""1Pの原enqueue成功→原step完了を終了票へ分離する候補。旧確率台帳は凍結する。"""
from __future__ import annotations
from dataclasses import asdict
import json
import sys
from typing import Any
import terminal_boundary as C

MAX_CALL_DEPTH = 16


def derive(arrival: Any) -> tuple[type, type]:
    """実loaderと同じ型のMode/Captureを受け取り、2P用BASE型には触れない。"""
    old_capture, old_mode = arrival.E.Capture, arrival.Mode
    capture = capture_type(old_capture, arrival)
    class Mode(old_mode):
        def observe(self, item: dict, result: Any, error: Any) -> dict:
            owner = self.arrival_capture
            if isinstance(owner, capture) and owner.pending is not None and owner.pending['active'] is False:
                return owner.complete_terminal(item, result, error)
            return super().observe(item, result, error)

        def save(self, row: dict) -> None:
            if row.get('terminal_scope_observation') is True:
                return  # 終了専用票はCaptureが保存済み。旧Native/tracking分母を水増ししない。
            super().save(row)
    return Mode, capture


def terminal_packet(owner: Any, old: dict, token: str, added: list, local: dict) -> dict:
    mode, ledger = owner.mode, owner.mode.arrival_ledger
    C.require(local['pipe'] is owner.recovery.pipe and owner.journal.active is None, 'terminal_actual_pipe')
    C.require(local['active'] is False and local['side'] == ledger.scope[-1], 'terminal_actual_inactive')
    original_returned_none(owner, token)
    row = dict(local['scope'], kind='enqueue', token=token, software_reset=old['epoch'],
        active=False, before=local['before'], after=owner.account(local['pipe'], local['side']),
        added_occurrence_tokens=list(added), fifo_occurrence_tokens=list(old['tokens']),
        discarded_tokens=list(old['discarded_tokens']), returned_none=True, status='returned',
        update_begin_accounting=owner.journal.update_before[local['side']]['accounting'])
    owner.verify(row)
    if owner.terminal_receipt is None:
        C.require(not mode.native.pending and (mode.basis_origin is None or mode.basis_cascade_closed),
            'terminal_unfinished_basis')
        context = mode.state['provisional_context_observer'].rows[-1]
        end = C.end_from_context(ledger, context)
        owner.terminal_prepared = C.prepare(owner.L, ledger, row, end)
    else:
        C.require(row['source_id'] == ledger.scope[0] and row['run_id'] == ledger.scope[1]
            and row['pipe_object_id'] == ledger.scope[3] and row['frame_idx'] == owner.terminal_last + C.STRIDE
            and row['frame_idx'] <= ledger.deadline and row['time_sec'] == row['frame_idx']/C.FPS, 'terminal_tail_scope_clock')
        C.require(not added and not row['fifo_occurrence_tokens'] and not row['before']['pending_tsumo']
            and not row['after']['pending_tsumo'] and not row['after']['tsumo_count'], 'terminal_tail_fifo')
    return row


def original_returned_none(owner: Any, token: str) -> None:
    frame = sys._getframe(1)
    try:
        for _ in range(MAX_CALL_DEPTH):
            if frame is None: break
            if frame.f_code == owner.code:
                local = frame.f_locals
                C.require(local['self'] is owner.journal and local['token'] == token
                    and 'result' in local and local['result'] is None, 'terminal_original_return')
                return
            frame = frame.f_back
    finally:
        del frame
    C.require(False, 'terminal_original_caller')


def capture_type(base: type, arrival: Any) -> type:
    class Capture(base):
        def __init__(self, mode: Any, output: Any) -> None:
            super().__init__(mode, output)
            self.L, self.terminal_prepared, self.terminal_receipt = arrival.L, None, None
            self.terminal_last, self.terminal_rows, self.terminal_stream = None, 0, None
            self.frozen_current, self.frozen_ledger = None, None

        def packet(self, old: dict, token: str, added: list, local: dict, ledger: Any) -> dict:
            if self.terminal_receipt is not None:
                return terminal_packet(self, old, token, added, local)
            try:
                return super().packet(old, token, added, local, ledger)
            except ValueError as error:
                if str(error) != 'arrival_ack:enqueue_clock_active' or local['active'] is not False:
                    raise
                return terminal_packet(self, old, token, added, local)

        def complete_terminal(self, item: dict, result: Any, error: Any) -> dict:
            return complete_terminal(self, arrival, item, result, error)

        def close(self, kind: Any, body: Any, trace: Any) -> bool:
            failure = None
            try:
                super().close(kind, body, trace)
            except BaseException as caught:
                failure = caught
            try:
                if self.terminal_stream is not None:
                    self.terminal_stream.close()
                with (self.output/'FIRST_TERMINAL_STATUS.json').open('x') as stream:
                    json.dump(dict(receipt=self.terminal_receipt, rows=self.terminal_rows,
                        last_frame=self.terminal_last, pending=self.pending is not None,
                        closed=self.terminal_stream is None or self.terminal_stream.closed,
                        error=self.error, original_body=None if body is None else repr(body),
                        quality_gate_clear=False), stream)
            except BaseException as caught:
                failure = failure or caught
            if failure is not None:
                if body is None: raise failure
                print('FIRST_TERMINAL_CLOSE_ERROR=' + repr(failure), file=sys.stderr)
            return False
    return Capture


def complete_terminal(owner: Any, arrival: Any, item: dict, result: Any, error: Any) -> dict:
    """原Jの同call完了通知からだけ呼ぶ。後続の元step保存失敗は外側へ残す。"""
    row, journal = owner.pending, owner.journal
    try:
        if error is not None:
            raise error  # 元step失敗の型・内容を終了資格エラーで置き換えない。
        C.require(error is None and owner.error is None and owner.recovery.error is None
            and not journal.errors and journal.active is None, 'terminal_original_failure')
        frame = item['frame']
        C.require(frame is not None and frame.f_code in journal.codes and item['pipe'] is owner.recovery.pipe,
            'terminal_original_step')
        C.require(frame.f_locals.get('is_active') is False and frame.f_locals['frame_idx'] == row['frame_idx']
            and frame.f_locals['time_sec'] == row['time_sec'], 'terminal_step_inactive_clock')
        native = arrival.CORE.Mode.observe.__globals__['N']
        C.require(item['scope']['frame_idx'] == row['frame_idx'] and item['scope']['side'] == row['side']
            and native.extract(item) is None, 'terminal_step_no_consumption')
        if owner.terminal_receipt is None:
            connection = owner.mode.connection
            owner.frozen_current = connection.registry.current(connection.binding)
            owner.frozen_ledger = owner.mode.arrival_ledger
            C.require(owner.frozen_current.scope == owner.frozen_ledger.scope, 'terminal_current_scope')
            owner.terminal_receipt = owner.terminal_prepared
            owner.terminal_stream = (owner.output/'FIRST_TERMINAL.jsonl').open('x', encoding='utf-8')
        C.require(owner.terminal_receipt['last_live_ledger'] == asdict(owner.mode.arrival_ledger), 'terminal_frozen_ledger')
        packet = dict(source_fields=row, completed_call_token=item['token'],
            terminal_receipt=owner.terminal_receipt if owner.terminal_rows == 0 else None,
            generation_after=journal.scope(owner.recovery.pipe, row['side'], row['frame_idx'], row['time_sec'])['generation'],
            quality_gate_clear=False)
        owner.terminal_stream.write(json.dumps(packet, allow_nan=False)+'\n')
        owner.terminal_stream.flush()
        owner.terminal_last, owner.pending = row['frame_idx'], None
        owner.terminal_rows += 1
        return dict(kind='probabilistic_tracking_observation', terminal_scope_observation=True,
            scope=item['scope'], journal_token=item['token'], reason='match_ended_scope_frozen',
            provisional_update=False, quality_gate_clear=False)
    except BaseException as failure:
        owner.error = repr(failure)
        owner.mode.fail(failure)
        raise
