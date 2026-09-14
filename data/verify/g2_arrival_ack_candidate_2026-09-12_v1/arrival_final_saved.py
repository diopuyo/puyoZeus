"""原基準/復元検査を残し、新しい到来分離replayを実終了検査へ供給する。"""
from __future__ import annotations

from dataclasses import asdict
from typing import Any

import old_saved as OLD
import arrival_saved as V
import stable_capture as Q
import enqueue_capture as E

R, L = V.R, V.L


def verify(state: dict, serializer: Any, native: Any) -> dict:
    c, mode = state['probabilistic_basis_connection'], state['probabilistic_tracking_mode']
    initial = OLD.basis(state, serializer)
    output = state['output']
    read = lambda name: OLD.read(output, name)
    rows, packets = read('PROBABILISTIC_TRACKING.jsonl'), read('ARRIVAL_SOURCE.jsonl')
    status = read('PROBABILISTIC_TRACKING_STATUS.json')
    L.require(status['activation'] == R.normalized(mode.activation)
        and status['error'] is mode.error is None and len(rows) == status['rows'] == mode.rows,
        'final_tracking_status')
    context_rows = read('provisional_context.jsonl')
    contexts = {r['frame_idx']: r for r in context_rows}
    L.require(len(contexts) == len(context_rows), 'final_duplicate_context')
    _, checker, _ = E.originals(c.recovery.journal)
    L.require(native.extract is R.N.extract, 'final_original_extract_identity')
    current, ledger, receipts, closed = V.replay(rows, packets, read('atomic_journal.jsonl'), initial,
        mode.prior, c.binding.initial_call_token, mode.native.last_frame, checker,
        lambda row, step: Q.verify(row, step, contexts[step['frame_idx']]))
    return finish(state, current, ledger, receipts, closed, packets)


def finish(state: dict, current: Any, ledger: Any, receipts: list,
           closed: bool, packets: list) -> dict:
    c, mode = state['probabilistic_basis_connection'], state['probabilistic_tracking_mode']
    read = lambda name: OLD.read(state['output'], name)
    R.finished_run(ledger, read('ARRIVAL_SOURCE_STATUS.json'), read('ARRIVAL_LEDGER_STATUS.json'),
                   len(packets), mode.native.last_frame)
    L.require(receipts == R.normalized(mode.applied), 'final_live_receipts')
    L.require(current == c.registry.current(c.binding), 'final_live_current')
    L.require(ledger == mode.arrival_ledger, 'final_live_ledger')
    L.require({a.token for a in ledger.acknowledgements} == mode.native.seen_occurrences,
              'final_native_occurrences')
    rows = read('PROBABILISTIC_TRACKING.jsonl')
    L.require({r['journal_token'] for r in rows} == mode.native.seen_calls
        and not mode.native.pending and read('PROBABILISTIC_TRACKING_STATUS.json')['pending_native_consumptions'] == 0,
        'final_native_calls_or_pending')
    L.require(mode.basis_origin is None or closed is mode.basis_cascade_closed is True, 'final_unclosed_basis')
    L.require(read('INFLIGHT_QUARANTINE.json')['references_restored'] is True, 'final_original_restore')
    return dict(saved_distribution_and_consumptions_verified=True, transitions=len(receipts),
        consumptions=len(ledger.acknowledgements), arrivals=len(ledger.arrivals),
        basis_frame=ledger.start, current_frame=current.frame,
        quality_gate_clear=False, physical_certified=False, production_permission=False)
