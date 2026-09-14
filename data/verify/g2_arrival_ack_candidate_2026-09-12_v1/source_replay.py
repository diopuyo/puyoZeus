"""候補の台帳表示を信用せず、元enqueueと元popから別台帳を再構成する。"""
from __future__ import annotations

from dataclasses import asdict
import json
from typing import Any, Callable

import ledger as L
import native_consumption as N

FPS = 60


def normalized(value: Any) -> Any:
    return json.loads(json.dumps(value, allow_nan=False, sort_keys=True))


def scope(row: dict, ledger: L.Ledger) -> None:
    expected = ledger.scope
    L.require(row['source_id'] == expected[0] and row['run_id'] == expected[1]
        and row['software_reset'] == expected[2] and row['pipe_object_id'] == expected[3]
        and row['generation']['reset_epoch'] == expected[5] and row['generation']['side'] == expected[-1]
        and row['side'] == expected[-1], 'replay_original_scope')
    L.require(type(row['frame_idx']) is int and ledger.clock <= row['frame_idx'] <= ledger.deadline
              and row['time_sec'] == row['frame_idx'] / FPS, 'replay_original_clock')


def arrival(ledger: L.Ledger, packet: dict, original: dict, step: dict,
            verify_enqueue: Callable[[dict], None]) -> L.Ledger:
    scope(original, ledger)
    scope(step, ledger)
    L.require(original['kind'] == 'enqueue' and original['status'] == 'returned'
              and original['active'] is True, 'replay_original_enqueue')
    L.require(step['kind'] == 'step' and step['status'] == 'returned' and step['exception'] is None,
              'replay_original_step')
    L.require(packet['quality_gate_clear'] is False and packet['completed_call_token'] == step['token']
              and original['frame_idx'] == step['frame_idx'], 'replay_same_call')
    fields = packet['source_fields']
    required = {'source_id', 'run_id', 'frame_idx', 'time_sec', 'side', 'generation',
                'pipe_object_id', 'kind', 'token', 'software_reset', 'active', 'before',
                'after', 'added_occurrence_tokens', 'fifo_occurrence_tokens'}
    L.require(required <= fields.keys(), 'replay_source_fields_missing')
    L.require(all(k in original and original[k] == v for k, v in fields.items()), 'replay_source_fields')
    verify_enqueue(original)
    verify_enqueue(fields)
    added = original['added_occurrence_tokens']
    pending = original['fifo_occurrence_tokens'][:-1] if added else original['fifo_occurrence_tokens']
    L.require(tuple(pending) == tuple(a.token for a in ledger.arrivals[len(ledger.acknowledgements):]),
              'replay_enqueue_prefix')
    if not added:
        return ledger
    value = L.Arrival(ledger.scope, added[0], tuple(original['after']['pending_tsumo'][-1]),
                      original['frame_idx'], original['token'])
    return L.arrive(ledger, value)


def acknowledge(ledger: L.Ledger, row: dict, step: dict) -> L.Ledger:
    scope(step, ledger)
    event = N.extract(dict(events=step['events'], token=step['token'], scope=dict(frame_idx=step['frame_idx'])))
    L.require(row['native_consumption'] == (None if event is None else normalized(asdict(event))),
              'replay_native_consumption')
    if event is None:
        return ledger
    return L.acknowledge(ledger, event.occurrence_token, event.pair, event.consumed_frame, event.source_call_token)


def finished_row(ledger: L.Ledger, row: dict, step: dict) -> L.Ledger:
    """物理遷移を別途再計算してから呼ぶ。保存されたapplied列だけで追加しない。"""
    ledger = L.advance_clock(ledger, step['frame_idx'])
    L.require(row['journal_token'] == step['token'] and row['scope']['frame_idx'] == step['frame_idx'], 'replay_tracking_call')
    L.require(row['pending_occurrences'] == [] and row['arrival_ledger'] == normalized(asdict(ledger)), 'replay_ledger')
    L.require(row['unapplied_arrivals'] == [a.token for a in ledger.arrivals[len(ledger.applied):]]
        and row['unacknowledged_arrivals'] == [a.token for a in ledger.arrivals[len(ledger.acknowledgements):]],
        'replay_waiting_sets')
    return ledger


def finished_run(ledger: L.Ledger, source_status: dict, ledger_status: dict,
                 replayed_source_rows: int, expected_end: int) -> None:
    """原票全行を突合した呼び手からのみ使用。未ack/未反映/未復元を終了成功にしない。"""
    L.check(ledger)
    L.require(ledger.clock == expected_end and L.drained(ledger), 'replay_incomplete_tail')
    L.require(type(replayed_source_rows) is int and replayed_source_rows > 0
        and source_status['rows'] == replayed_source_rows
        and source_status['closed'] is True and source_status['restored'] is True
        and source_status['error'] is None and source_status['pending'] is False
        and source_status['quality_gate_clear'] is False, 'replay_source_close')
    L.require(ledger_status['state'] == normalized(asdict(ledger))
        and ledger_status['drained'] is True and ledger_status['error'] is None
        and ledger_status['quality_gate_clear'] is False, 'replay_ledger_close')
