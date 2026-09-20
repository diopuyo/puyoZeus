"""元selectorの責務外である実update時計・世代境界を限定CPUで確認する。"""
from scripts.chain_end_epoch_shadow_v1 import _origin_prediction
from test_diagnose_video38_prediction_ledger_shadow_v1 import _recorder, _open


def test_original_selector_is_snapshot_only_after_clock_ends() -> None:
    rec, generation, stream = _recorder()
    handle = _open(rec)
    snapshot = rec.ledger.snapshot(handle)
    original = _origin_prediction(snapshot)
    generation.end()
    before = stream.getvalue()
    assert not rec._clock_active()
    assert _origin_prediction(snapshot) == original
    assert rec.ledger.snapshot(handle) == snapshot and stream.getvalue() == before


def test_generation_change_needs_caller_check_without_sync_mutation() -> None:
    rec, generation, stream = _recorder()
    handle = _open(rec)
    snapshot = rec.ledger.snapshot(handle)
    generation.set('2P', 3, 12)
    before = stream.getvalue()
    assert snapshot.generation != rec._current_generation('2P')
    assert _origin_prediction(snapshot) is not None
    assert rec.ledger.snapshot(handle) == snapshot and stream.getvalue() == before
