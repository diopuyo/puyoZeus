"""旧基準/原Native/全終端検査を保持し、新prefix consumerを終了入口へ接続する。"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any
import prefix_replay as P

FILES = dict(initial='PREFIX_LANE_INITIAL.jsonl', origins='PREFIX_ORIGIN_ASSIGNMENTS.jsonl',
    decisions='PREFIX_STABLE_DECISIONS.jsonl', qualifications='PREFIX_STABLE_QUALIFICATIONS.jsonl',
    observations='PREFIX_LANE_OBSERVATIONS.jsonl')


def verify(original: Any, mode_module: Any, state: dict, serializer: Any, native: Any) -> dict:
    c, mode = state['probabilistic_basis_connection'], state['probabilistic_tracking_mode']
    initial = original.OLD.basis(state, serializer)
    read = lambda name: original.OLD.read(state['output'], name)
    rows, packets = read('PROBABILISTIC_TRACKING.jsonl'), read('ARRIVAL_SOURCE.jsonl')
    status = read('PROBABILISTIC_TRACKING_STATUS.json')
    L, R = original.L, original.R
    L.require(status['activation'] == R.normalized(mode.activation)
        and status['error'] is mode.error is None and len(rows) == status['rows'] == mode.rows,
        'final_tracking_status')
    context_rows = read('provisional_context.jsonl')
    contexts = {row['frame_idx']: row for row in context_rows}
    L.require(len(contexts) == len(context_rows), 'final_duplicate_context')
    _, checker, _ = original.E.originals(c.recovery.journal)
    L.require(native.extract is R.N.extract, 'final_original_extract_identity')
    prefix = read('PREFIX_LANE_STATUS.json')
    L.require(prefix['closed'] is True and prefix['error'] is None and prefix['close_error'] is None
        and prefix['quality_gate_clear'] is False, 'prefix_final_close')
    saved = {key: read(name) if name in prefix['files'] else [] for key, name in FILES.items()}
    L.require(set(prefix['files']) <= set(FILES.values()), 'prefix_final_unknown_file')
    parts = SimpleNamespace(mode=mode_module, arrival_saved=original)
    current, ledger, receipts, closed = P.replay(parts, saved, contexts, rows, packets,
        read('atomic_journal.jsonl'), initial, mode.prior, c.binding.initial_call_token,
        mode.native.last_frame, checker)
    L.require(prefix['observations'] == len(saved['observations'])
        and prefix['commits'] == sum(row['kind'] == P.C.VERSION for row in receipts)
        and prefix['initialized'] == bool(saved['initial']), 'prefix_final_counts')
    return original.finish(state, current, ledger, receipts, closed, packets)
