"""確率の原取得票・保存遷移・原FIFO消費を、保持された実modeと突合する。"""
from __future__ import annotations
from dataclasses import asdict
import json
from pathlib import Path
from typing import Any


def read(output: Path, name: str) -> Any:
    text = (output / name).read_text()
    return [json.loads(line) for line in text.splitlines()] if name.endswith('.jsonl') else json.loads(text)


def normalized(value: Any) -> Any:
    return json.loads(json.dumps(value, allow_nan=False, sort_keys=True))


def basis(state: Any, serializer: Any) -> Any:
    c = state['probabilistic_basis_connection']
    mode, output = state['probabilistic_tracking_mode'], state['output']
    packets = read(output, 'PROBABILISTIC_BASIS.jsonl')
    assert len(packets) == c.rows == 1 and c.stream.closed and mode.stream.closed
    packet = packets[0]
    assert packet['kind'] == 'actual_settled_probabilistic_basis'
    initial = serializer.decode(packet['state'])
    assert initial.scope == c.binding.scope and initial.frame == mode.activation['frame']
    assert packet['source_call_token'] == c.binding.initial_call_token == mode.activation['source_call_token']
    assert packet['initial_candidate'] == normalized(asdict(c.observer.gate.candidate))
    assert packet['basis_deadline'] == c.observer.gate.deadline == mode.activation['acquisition_deadline']
    assert packet['tracking_deadline'] == initial.deadline == c.deadline
    assert all(packet[key] is False for key in ('integer_current_published', 'legacy_collector_append', 'quality_gate_clear'))
    observer = read(output, 'SETTLED_BASIS_OBSERVER.json')
    assert observer['candidate'] == packet['initial_candidate'] and observer['error'] is None
    assert observer['restored'] is True and observer['restore_binding_owned'] is True
    assert c.observer.gate.state == 'ISSUED'
    return initial


def transitions(rows: Any, initial: Any, serializer: Any, native: Any, steps: Any) -> tuple[Any, Any, Any]:
    """原extractを保存Jに適用。検証用FIFOを再現し原FIFOには触れない。"""
    current, pending, used, receipts = initial, {}, set(), []
    for row in rows:
        step = steps[row['journal_token']]
        assert step['frame_idx'] == row['scope']['frame_idx'] and step['side'] == initial.scope[-1]
        event = native.extract(dict(events=step['events'], token=step['token'],
                                    scope=dict(frame_idx=step['frame_idx'])))
        assert row['native_consumption'] == (None if event is None else normalized(asdict(event)))
        if event is not None:
            assert event.occurrence_token not in used
            used.add(event.occurrence_token)
            pending[event.occurrence_token] = event
        receipt = row.get('transition')
        if receipt is not None:
            assert receipt['source_call_token'] == row['journal_token']
            following = serializer.decode(receipt['state'])
            assert current.frame < following.frame == receipt['applied_frame'] == row['scope']['frame_idx']
            assert following.scope == current.scope and following.deadline == current.deadline
            if receipt.get('kind') == 'basis_cascade':
                assert not pending and receipt['next_consumed'] is False
                operation = receipt['origin']['operation_token']
            else:
                operation = receipt['occurrence_token']
                assert operation in pending and receipt['consumed_frame'] == pending[operation].consumed_frame
                del pending[operation]
            assert following.tokens == current.tokens + (operation,)
            current = following
            receipts.append(receipt)
        assert row['pending_occurrences'] == list(pending)
    return current, dict(pending=pending, used=used), receipts


def verify(state: Any, serializer: Any, native: Any) -> dict[str, Any]:
    c, mode = state['probabilistic_basis_connection'], state['probabilistic_tracking_mode']
    initial = basis(state, serializer)
    rows = read(state['output'], 'PROBABILISTIC_TRACKING.jsonl')
    status = read(state['output'], 'PROBABILISTIC_TRACKING_STATUS.json')
    assert status['activation'] == normalized(mode.activation) and status['error'] is mode.error is None
    assert len(rows) == status['rows'] == mode.rows
    journal = [row for row in read(state['output'], 'atomic_journal.jsonl') if row['kind'] == 'step']
    steps = {row['token']: row for row in journal}
    assert len(steps) == len(journal), 'duplicate_J_token'
    current, events, receipts = transitions(rows, initial, serializer, native, steps)
    assert receipts == normalized(mode.applied)
    assert current == c.registry.current(c.binding), 'saved_current_mismatch'
    assert events['used'] == mode.native.seen_occurrences
    assert set(row['journal_token'] for row in rows) == mode.native.seen_calls
    assert not events['pending'] and not mode.native.pending and status['pending_native_consumptions'] == 0
    assert mode.basis_origin is None or mode.basis_cascade_closed, 'basis_cascade_unclosed'
    assert read(state['output'], 'INFLIGHT_QUARANTINE.json')['references_restored'] is True
    return dict(saved_distribution_and_consumptions_verified=True, transitions=len(receipts),
                consumptions=len(events['used']), basis_frame=initial.frame, current_frame=current.frame,
                quality_gate_clear=False, physical_certified=False, production_permission=False)
