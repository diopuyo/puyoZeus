"""原JのFIFO到来番号発行を観測し、原step完了後にのみ台帳へ渡す。"""
from __future__ import annotations

import json
from pathlib import Path
import sys
from types import CodeType
from typing import Any

import ledger as L

SOURCE = Path(__file__).resolve().parent.parent / 'g2_atomic_journal_capture_2026-09-09_v1/observer.py'
MAX_CALL_DEPTH, FPS = 16, 60


def named(code: CodeType, name: str) -> CodeType:
    values = [v for v in code.co_consts if isinstance(v, CodeType) and v.co_name == name]
    L.require(len(values) == 1, 'enqueue_source_code')
    return values[0]


def originals(journal: Any) -> tuple[CodeType, Any, Any]:
    factory = type(journal).wrap_enqueue
    compiled = compile(SOURCE.read_bytes(), str(SOURCE), 'exec', dont_inherit=True)
    expected = named(named(compiled, 'Recorder'), 'wrap_enqueue')
    L.require(factory.__code__ == expected, 'enqueue_factory_replaced')
    checker = factory.__globals__['verify_enqueue']
    L.require(checker.__code__ == named(compiled, 'verify_enqueue'), 'enqueue_checker_replaced')
    account = factory.__globals__['account']
    L.require(account.__code__ == named(compiled, 'account'), 'enqueue_account_replaced')
    L.require(journal.fifo.appended.__func__.__code__ == named(named(compiled, 'Fifo'), 'appended'),
              'enqueue_appended_replaced')
    return named(expected, 'enqueue'), checker, account


def caller(journal: Any, code: CodeType, old: dict, token: str) -> dict:
    current = sys._getframe(1)
    try:
        for _ in range(MAX_CALL_DEPTH):
            if current is None:
                break
            if current.f_code == code:
                local = current.f_locals
                L.require(local['self'] is journal and local['token'] == token
                          and local['saved'] is old, 'enqueue_call_owner')
                return {k: local[k] for k in ('scope', 'pipe', 'side', 'frame', 'clock', 'active', 'before')}
            current = current.f_back
    finally:
        del current
    raise ValueError('arrival_ack:outside_original_enqueue')


class Capture:
    def __init__(self, mode: Any, output: Path) -> None:
        self.mode, self.recovery = mode, mode.connection.recovery
        self.journal = self.recovery.journal
        self.code, self.verify, self.account = originals(self.journal)
        self.fifo, self.original = self.journal.fifo, self.journal.fifo.appended
        L.require('appended' not in vars(self.fifo), 'enqueue_foreign_hook')
        self.hook, self.output = self.appended, output
        self.rows, self.error, self.pending = 0, None, None
        self.stream = (output / 'ARRIVAL_SOURCE.jsonl').open('x', encoding='utf-8')
        self.fifo.appended = self.hook

    def packet(self, old: dict, token: str, added: list[str], local: dict, ledger: L.Ledger) -> dict:
        r, scope = self.recovery, local['scope']
        L.require(local['pipe'] is r.pipe and self.journal.active is None, 'enqueue_pipeline_or_active_step')
        L.require(r.evidence.scope(r.factory, r.pipe) == ledger.scope, 'enqueue_binding_scope')
        L.require(scope['source_id'] == ledger.scope[0] and scope['run_id'] == ledger.scope[1]
                  and old['epoch'] == ledger.scope[2] and scope['side'] == ledger.scope[-1], 'enqueue_row_scope')
        L.require(scope['pipe_object_id'] == ledger.scope[3] == id(r.pipe), 'enqueue_pipe_id')
        L.require(scope['generation']['reset_epoch'] == ledger.scope[5]
                  and scope['generation']['side'] == ledger.scope[-1], 'enqueue_generation')
        L.require(type(local['frame']) is int and ledger.start < local['frame']
                  and ledger.clock <= local['frame'] <= ledger.deadline
                  and scope['frame_idx'] == local['frame'] and scope['time_sec'] == local['clock']
                  and local['clock'] == local['frame'] / FPS and local['active'] is True, 'enqueue_clock_active')
        row = dict(scope, kind='enqueue', token=token, software_reset=old['epoch'], active=local['active'],
                   before=local['before'], after=self.account(r.pipe, local['side']),
                   added_occurrence_tokens=added, fifo_occurrence_tokens=list(old['tokens']))
        self.verify(row)
        prefix = row['fifo_occurrence_tokens'][:-1] if added else row['fifo_occurrence_tokens']
        L.require(tuple(prefix) == tuple(a.token for a in ledger.arrivals[len(ledger.acknowledgements):]),
                  'enqueue_unexplained_prefix')
        return row

    def appended(self, old: dict, token: str) -> list[str]:
        added = self.original(old, token)  # 原番号発行を一回だけ実行し、元例外はそのまま渡す。
        ledger = self.mode.arrival_ledger
        if ledger is None:
            return added
        try:
            local = caller(self.journal, self.code, old, token)
            if local['side'] != ledger.scope[-1]:
                return added
            L.require(self.fifo.appended is self.hook and self.pending is None, 'enqueue_owner_or_unfinished')
            packet = self.packet(old, token, added, local, ledger)
            self.pending = json.loads(json.dumps(packet, allow_nan=False))
            return added
        except BaseException as error:
            self.error = repr(error)
            self.mode.fail(error)
            raise

    def completed(self, item: dict) -> None:
        """呼び手は原Native.observeの同call検査後。元emit失敗/未完stepでは反映しない。"""
        try:
            row, ledger = self.pending, self.mode.arrival_ledger
            L.require(self.mode.arrival_step is item, 'enqueue_unverified_step')
            L.require(row is not None and row['frame_idx'] == item['scope']['frame_idx'], 'enqueue_missing_completed')
            L.require(not self.journal.errors and self.journal.active is None
                      and self.recovery.error is None and self.error is None, 'enqueue_original_failure')
            L.require(row['source_id'] == item['scope']['source_id'] and row['run_id'] == item['scope']['run_id']
                      and row['side'] == item['scope']['side'] and row['pipe_object_id'] == item['scope']['pipe_object_id']
                      and row['time_sec'] == item['scope']['time_sec'], 'enqueue_completed_scope')
            without_action = lambda g: {k: v for k, v in g.items() if k != 'action_revision'}
            L.require(without_action(row['generation']) == without_action(item['scope']['generation']), 'enqueue_completed_generation')
            added = row['added_occurrence_tokens']
            following = ledger
            if added:
                arrival = L.Arrival(ledger.scope, added[0], tuple(row['after']['pending_tsumo'][-1]),
                                    row['frame_idx'], row['token'])
                following = L.arrive(ledger, arrival)
            packet = dict(source_fields=row, completed_call_token=item['token'], quality_gate_clear=False)
            self.stream.write(json.dumps(packet, ensure_ascii=False, allow_nan=False) + '\n')
            self.stream.flush()
            self.mode.arrival_ledger, self.pending = following, None
            self.rows += 1
        except BaseException as error:
            self.error = repr(error)
            self.mode.fail(error)
            raise

    def close(self, kind: Any, body: Any, trace: Any) -> bool:
        owned = vars(self.fifo).get('appended') is self.hook
        if owned:
            del self.fifo.appended
        restored = owned and self.fifo.appended == self.original
        try:
            self.stream.close()
            with (self.output / 'ARRIVAL_SOURCE_STATUS.json').open('x') as stream:
                json.dump(dict(rows=self.rows, closed=self.stream.closed, restored=restored,
                    error=self.error, pending=self.pending is not None, quality_gate_clear=False), stream, indent=2)
            L.require(restored and (body is not None or self.pending is None), 'enqueue_restore_or_pending')
        except BaseException as error:
            if body is None:
                raise
            print('ARRIVAL_SOURCE_CLOSE_ERROR=' + repr(error), file=sys.stderr, flush=True)
        return False


def install(stack: Any, mode: Any, output: Path) -> Capture:
    value = Capture(mode, output)
    stack.push(value.close)
    return value
