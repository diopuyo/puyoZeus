"""元の確率prefix再計算と、終了後の原J全被覆を分けて検査する。"""
from __future__ import annotations
import json
from types import FunctionType, SimpleNamespace as N
from typing import Any
import terminal_boundary as C


def normalized(value: Any) -> Any:
    return json.loads(json.dumps(value, allow_nan=False))


def verify_records(original: Any, ledger: Any, raw: list, packets: list, contexts: dict) -> dict:
    side, last = ledger.scope[-1], ledger.clock
    selected = [r for r in raw if r['side'] == side and r['frame_idx'] > last]
    enqueues = [r for r in selected if r['kind'] == 'enqueue']
    steps = [r for r in selected if r['kind'] == 'step']
    frames = list(range(last+C.STRIDE, ledger.deadline+C.STRIDE, C.STRIDE))
    C.require(frames and [r['frame_idx'] for r in enqueues] == [r['frame_idx'] for r in steps] == frames,
        'terminal_saved_coverage')
    C.require(len(selected) == 2*len(frames) and len(packets) == len(frames)
        and len({r['token'] for r in selected}) == len(selected),
        'terminal_saved_tokens')
    prior = [r for r in raw if r['kind'] == 'step' and r['side'] == side and r['frame_idx'] == last]
    C.require(len(prior) == 1 and type(prior[0]['code_sha256']) is str, 'terminal_prior_code')
    end = C.end_from_context(ledger, contexts[last])
    receipt = C.prepare(original.L, ledger, enqueues[0], end)
    for index, (enqueue, step, packet) in enumerate(zip(enqueues, steps, packets, strict=True)):
        verify_pair(original, ledger, enqueue, step, packet, prior[0]['code_sha256'])
        if index:
            C.require(enqueue['discarded_tokens'] == receipt['unacknowledged_terminal_tokens']
                and not enqueue['update_begin_accounting']['pending_tsumo'], 'terminal_saved_tail_discard')
        C.require(packet['terminal_receipt'] == (normalized(receipt) if index == 0 else None),
            'terminal_saved_receipt')
    return dict(last_live_frame=last, terminal_start=frames[0], terminal_end=frames[-1],
        original_terminal_updates=len(frames), frozen_ledger=True, additional_acknowledgements=0,
        additional_physical_applications=0, quality_gate_clear=False)


def verify_pair(original: Any, ledger: Any, enqueue: dict, step: dict, packet: dict, code: str) -> None:
    C.require(enqueue['status'] == step['status'] == 'returned' and step['exception'] is None
        and enqueue['active'] is False and enqueue['returned_none'] is True, 'terminal_saved_original_success')
    C.require(enqueue['row_index'] < step['row_index'] and step['code_sha256'] == code,
        'terminal_saved_original_order_code')
    for row in (enqueue, step):
        C.require(row['source_id'] == ledger.scope[0] and row['run_id'] == ledger.scope[1]
            and row['pipe_object_id'] == ledger.scope[3] and row['time_sec'] == row['frame_idx']/C.FPS,
            'terminal_saved_original_identity')
    fields = packet['source_fields']
    C.require(fields and all(key in enqueue and enqueue[key] == value for key, value in fields.items()),
        'terminal_saved_source_fields')
    C.require(packet['completed_call_token'] == step['token'] and packet['quality_gate_clear'] is False
        and packet['generation_after'] == step['generation_after'], 'terminal_saved_completed_step')
    C.require(not enqueue['added_occurrence_tokens'] and not enqueue['fifo_occurrence_tokens']
        and not enqueue['before']['pending_tsumo'] and not enqueue['after']['pending_tsumo']
        and not enqueue['after']['tsumo_count'], 'terminal_saved_no_arrival')
    event = original.R.N.extract(dict(events=step['events'], token=step['token'], scope=dict(frame_idx=step['frame_idx'])))
    C.require(event is None, 'terminal_saved_no_consumption')


def terminal_files(original: Any, state: dict) -> tuple:
    mode = state['probabilistic_tracking_mode']
    capture = mode.arrival_capture
    C.require(mode.native.last_frame == mode.arrival_ledger.clock, 'terminal_saved_live_clock')
    read = lambda name: original.OLD.read(state['output'], name)
    raw, packets = read('atomic_journal.jsonl'), read('FIRST_TERMINAL.jsonl')
    _, checker, _ = original.E.originals(mode.connection.recovery.journal)
    for packet in packets:
        checker(packet['source_fields'])
    contexts = {row['frame_idx']: row for row in read('provisional_context.jsonl')}
    tail = verify_records(original, mode.arrival_ledger, raw, packets, contexts)
    status = read('FIRST_TERMINAL_STATUS.json')
    C.require(status['closed'] is True and status['pending'] is False and status['error'] is None
        and status['original_body'] is None and status['rows'] == capture.terminal_rows == len(packets)
        and status['last_frame'] == capture.terminal_last == tail['terminal_end']
        and status['receipt'] == normalized(capture.terminal_receipt), 'terminal_saved_close')
    return raw, tail


def prefix_original(original: Any, state: dict, raw: list) -> Any:
    mode = state['probabilistic_tracking_mode']
    prefix = [row for row in raw if row['side'] != '1P' or row['frame_idx'] <= mode.native.last_frame]
    def prefix_read(output: Any, name: str) -> Any:
        return prefix if name == 'atomic_journal.jsonl' else original.OLD.read(output, name)
    old = N(**(vars(original.OLD) | {'read': prefix_read}))
    return N(**(vars(original) | {'OLD':old}))


def verify(original: Any, state: dict, serializer: Any, native: Any) -> dict:
    mode = state['probabilistic_tracking_mode']
    if getattr(mode.arrival_capture, 'terminal_receipt', None) is None:
        return original.verify(state, serializer, native)
    raw, tail = terminal_files(original, state)
    old = prefix_original(original, state, raw).OLD
    function = FunctionType(original.verify.__code__, dict(vars(original), OLD=old))
    return function(state, serializer, native) | dict(terminal=tail)


def verify_prefix(prefix: Any, original: Any, module: Any, state: dict, serializer: Any, native: Any) -> dict:
    """実採用prefix consumer自体を残す。旧arrival-only replayへ戻さない。"""
    mode = state['probabilistic_tracking_mode']
    if getattr(mode.arrival_capture, 'terminal_receipt', None) is None:
        return prefix.verify(original, module, state, serializer, native)
    raw, tail = terminal_files(original, state)
    selected = prefix_original(original, state, raw)
    return prefix.verify(selected, module, state, serializer, native) | dict(terminal=tail)
