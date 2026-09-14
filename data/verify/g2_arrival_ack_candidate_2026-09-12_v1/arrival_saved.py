"""元J全列と到来票を一対一で突合する。物理資格の検査器なしでは実行しない。"""
from __future__ import annotations

from typing import Any, Callable

import source_replay as R
import receipt_replay as P

L = R.L


def selected(journal: list[dict], initial: Any, end: int) -> tuple[list[dict], list[dict]]:
    scope = initial.scope
    ends = [r['frame_idx'] for r in journal if r.get('kind') in ('step', 'enqueue') and r['side'] == scope[-1]]
    L.require(bool(ends) and end == max(ends), 'saved_original_end_mismatch')
    rows = [r for r in journal if r.get('kind') in ('step', 'enqueue')
            and r['side'] == scope[-1] and initial.frame < r['frame_idx'] <= end]
    L.require(bool(rows), 'saved_original_rows_missing')
    L.require(len({r['token'] for r in rows}) == len(rows), 'saved_duplicate_original_token')
    indices = [r['row_index'] for r in rows]
    L.require(all(type(i) is int for i in indices) and indices == sorted(set(indices)), 'saved_original_order')
    ledger = L.Ledger(scope, initial.frame, initial.deadline, initial.frame)
    for row in rows:
        R.scope(row, ledger)
    steps = [r for r in rows if r['kind'] == 'step']
    enqueues = [r for r in rows if r['kind'] == 'enqueue']
    frames = [s['frame_idx'] for s in steps]
    L.require(bool(frames) and frames == sorted(set(frames)) and frames[-1] == end, 'saved_step_order_or_end')
    L.require([r['frame_idx'] for r in enqueues] == frames, 'saved_missing_enqueue_or_step')
    L.require(all(e['row_index'] < s['row_index'] for e, s in zip(enqueues, steps)), 'saved_enqueue_after_step')
    return enqueues, steps


def replay(rows: list[dict], packets: list[dict], journal: list[dict], initial: Any,
           prior: Any, initial_call: str, end: int, verify_enqueue: Callable,
           qualify: Callable) -> tuple[Any, L.Ledger, list[dict], bool]:
    """qualifyは元raw/CNN/SM/effect/grace資格の独立検査。省略や既定PASSを許さない。"""
    L.require(callable(qualify) and callable(verify_enqueue), 'saved_missing_verifier')
    enqueues, steps = selected(journal, initial, end)
    L.require(len(rows) == len(packets) == len(steps), 'saved_row_count')
    L.require([r['journal_token'] for r in rows] == [s['token'] for s in steps], 'saved_tracking_order')
    L.require([p['completed_call_token'] for p in packets] == [s['token'] for s in steps], 'saved_source_order')
    originals = {r['token']: r for r in journal if r['kind'] == 'step'}
    L.require(len(originals) == sum(r['kind'] == 'step' for r in journal), 'saved_duplicate_step')
    ledger = L.Ledger(initial.scope, initial.frame, initial.deadline, initial.frame)
    current, receipts, closed = initial, [], False
    for row, packet, enqueue, step in zip(rows, packets, enqueues, steps):
        ledger = R.arrival(ledger, packet, enqueue, step, verify_enqueue)
        ledger = R.acknowledge(ledger, row, step)
        if row.get('transition') is not None:
            L.require(qualify(row, step) is True, 'saved_physical_qualification')
            current, ledger, closed = P.transition(current, ledger, row, step, originals,
                                                   prior, closed, initial_call)
            receipts.append(row['transition'])
        ledger = R.finished_row(ledger, row, step)
    return current, ledger, receipts, closed
