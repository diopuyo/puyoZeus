"""元Recorder/ledger/epoch observerを使う読取・保存・終了の限定CPU検査。"""
from dataclasses import asdict
import json
from pathlib import Path
import pytest
from scripts.chain_end_epoch_shadow_v1 import ChainEndEpochObserver, EndEvidence
from test_diagnose_video38_prediction_ledger_shadow_v1 import (
    _recorder, _open, _begin, _event, _board, _result, FRAME, TIME,
)
from origin_reference_reader import read_origin


def test_current_clock_read_save_and_cleanup_are_readonly(tmp_path: Path) -> None:
    rec, generation, stream = _recorder()
    handle = _open(rec)
    observer = ChainEndEpochObserver(rec)
    before, lines = rec.ledger.snapshot(handle), stream.getvalue()
    value = read_origin(rec, observer, '2P', FRAME, TIME)
    assert value.status == 'READY' and not value.accounting_permission
    packet = json.loads(value.origin_json)
    assert packet['origin']['prediction_revision'] == 1
    assert packet['meaning'] == 'total_from_fixed_origin_not_remaining_attack'
    assert not packet['later_prediction_substitution_allowed']
    path = tmp_path / 'origin.json'
    path.write_text(json.dumps(asdict(value)))
    assert json.loads(path.read_text()) == asdict(value)
    generation.end()
    assert read_origin(rec, observer, '2P', FRAME, TIME).reason == 'outside_exact_update_clock'
    assert rec.ledger.snapshot(handle) == before and stream.getvalue() == lines


def test_closed_epoch_does_not_leak_still_provisional_origin() -> None:
    rec, _, _ = _recorder()
    handle = _open(rec)
    observer = ChainEndEpochObserver(rec)
    observer._ready['2P'] = EndEvidence(handle.instance_id, '2P', 1, 1, 0, 40, 'artificial', 2)
    assert read_origin(rec, observer, '2P', FRAME, TIME).status == 'READY'
    observer.mark_closed('2P')
    assert rec.ledger.snapshot(handle).status.value == 'provisional'
    assert read_origin(rec, observer, '2P', FRAME, TIME).reason == 'already_closed_once'
    assert read_origin(rec, observer, '2P', FRAME, TIME).status == 'TERMINAL_HOLD'
    assert read_origin(rec, observer, '2P', FRAME, TIME).instance_id == handle.instance_id


@pytest.mark.parametrize('fault', ['invalidated', 'generation', 'owner', 'foreign_handle'])
def test_invalid_owner_or_generation_is_not_current(fault: str) -> None:
    rec, generation, stream = _recorder()
    handle = _open(rec)
    observer = ChainEndEpochObserver(rec)
    if fault == 'invalidated':
        observer._invalidated.add(('2P', handle.instance_id))
    elif fault == 'generation':
        generation.set('2P', 3, 12)
    else:
        other, _, _ = _recorder()
        other_handle = _open(other)
        if fault == 'owner': observer.rec = other
        else: rec.active_handles['2P'] = other_handle
    before, lines = rec.ledger.snapshot(handle), stream.getvalue()
    if fault in ('owner', 'foreign_handle'):
        with pytest.raises((ValueError, RuntimeError)):
            read_origin(rec, observer, '2P', FRAME, TIME)
    else:
        assert read_origin(rec, observer, '2P', FRAME, TIME).status != 'READY'
    assert rec.ledger.snapshot(handle) == before and stream.getvalue() == lines


def test_initial_prediction_missing_distinguishes_same_call_and_too_late() -> None:
    rec, generation, _ = _recorder()
    rec._record_ledger_episode('2P', _event(_board()))
    observer = ChainEndEpochObserver(rec)
    assert read_origin(rec, observer, '2P', FRAME, TIME).reason == 'origin_prediction_pending'
    _begin(rec, generation, FRAME + 2, TIME + 2 / 60)
    assert read_origin(rec, observer, '2P', FRAME + 2, TIME + 2 / 60).status == 'TERMINAL_HOLD'


def test_later_total_prediction_is_not_substituted_for_missing_origin() -> None:
    rec, _, _ = _recorder()
    rec._record_ledger_episode('2P', _event(_board()))
    rec.record_start('2P', _event(_board(), 'formula_read'), _result(_board()))
    snapshot = rec.ledger.snapshot(rec.active_handles['2P'])
    assert snapshot.predictions[0].scope.value == 'total_from_instance_origin'
    assert snapshot.origin_prediction_revision is None
    value = read_origin(rec, ChainEndEpochObserver(rec), '2P', FRAME, TIME)
    assert value.status == 'TERMINAL_HOLD' and value.origin_json is None


def test_real_sync_removes_handle_without_silently_switching_requested_instance() -> None:
    rec, generation, _ = _recorder()
    handle = _open(rec)
    observer = ChainEndEpochObserver(rec)
    generation.set('2P', 3, 12)
    rec._sync_handle('2P', None)
    assert '2P' not in rec.active_handles
    assert rec.ledger.snapshot(handle).status.value == 'invalidated'
    old = read_origin(rec, observer, '2P', FRAME, TIME, handle.instance_id)
    assert old.status == 'TERMINAL_HOLD' and old.instance_id == handle.instance_id
    assert read_origin(rec, observer, '2P', FRAME, TIME).reason == 'origin_handle_missing'
    replacement = _open(rec)
    assert replacement.instance_id != handle.instance_id
    assert read_origin(rec, observer, '2P', FRAME, TIME).instance_id == replacement.instance_id
    assert read_origin(rec, observer, '2P', FRAME, TIME, handle.instance_id).status == 'TERMINAL_HOLD'
