"""B13の保存記録分類と、cが時刻追従より余計に捨てない不変条件を検証する。"""
from pathlib import Path

import numpy as np
import pytest

from scripts.analyze_live_b13 import B12_BUILD, breakdown, drop_rows, validate_policy
from src.phase_j.live_degrade import EventPriority


@pytest.mark.parametrize('enabled', [False, True])
@pytest.mark.parametrize('until', [-1.0, 0.0, 0.1, 1.0, 10.0])
def test_c_never_adds_skips_to_latest_wins(enabled: bool, until: float) -> None:
    priority = EventPriority()
    priority.enabled, priority.until = enabled, until
    for stride, fps in ((1, 30), (2, 60)):
        for next_frame in range(0, 30, stride):
            for due in range(next_frame, 60, stride):
                selected = priority.select(next_frame, due, stride, fps)
                assert next_frame <= selected <= due
                assert (selected-next_frame) % stride == 0
                if due == next_frame:
                    assert selected == next_frame


def test_drop_evidence_does_not_invent_selector_timestamp() -> None:
    data = dict(t_sec=np.array([0, .1, .2]), captured_at=np.array([100, 100.1, 100.2]),
                recognized_at=np.array([100.09, 100.16, 100.23]),
                queue_put_ms=np.array([0, 20, 1]))
    rows = drop_rows(data, [-.1, 1/30, .1+1/30, .2+1/30], dict(fps=30, stride=1))
    assert [row['evidence'] for row in rows] == [
        'unobserved_source_or_other_work', 'recognition_already_late',
        'queue_wait_sufficient', 'unobserved_source_or_other_work']
    assert all(row['category'] == 'deadline_catch_up' for row in rows)


@pytest.mark.parametrize('change', [dict(realtime=False), dict(source='directshow'),
                                  dict(assets={}), dict(assets=dict(app_build_id='unknown'))])
def test_policy_classification_rejects_unknown_build(change: dict) -> None:
    metrics = dict(realtime=True, source='video', assets=dict(app_build_id=B12_BUILD))
    validate_policy(metrics)
    metrics.update(change)
    with pytest.raises(ValueError):
        validate_policy(metrics)


@pytest.mark.parametrize('fault', [None, 'missing', 'duplicate', 'wrong_count'])
def test_breakdown_covers_every_slot_once(tmp_path: Path, fault: str | None) -> None:
    import json
    source = np.arange(6)/30
    dropped = [float(source[2])]
    actual = np.delete(source, 2)
    if fault == 'missing':
        actual = actual[:-1]
    elif fault == 'duplicate':
        actual[-1] = actual[-2]
    data = dict(t_sec=actual, captured_at=100+actual, recognized_at=100+actual+.04,
                acquired_at=100+actual+.01, recognition_ms=np.full(len(actual), 30),
                queue_put_ms=np.zeros(len(actual)))
    np.savez(tmp_path/'recognition.npz', **data)
    metrics = dict(realtime=True, source='video', assets=dict(app_build_id=B12_BUILD),
        frame_bounds=dict(fps=30, start=0, end=6, stride=1), expected_frames=6,
        capture_dropped_in_measured_window=2 if fault == 'wrong_count' else 1)
    (tmp_path/'metrics.json').write_text(json.dumps(metrics))
    (tmp_path/'recognition.json').write_text(json.dumps(dict(dropped_times=dropped)))
    if fault is not None:
        with pytest.raises(ValueError):
            breakdown(tmp_path, 0, .2)
    else:
        result = breakdown(tmp_path, 0, .2)
        assert result['total']['expected'] == 6
        assert result['total']['recognized'] == 5
        assert result['total']['deadline_catch_up'] == 1
        assert result['total']['event_policy_extra_skip'] == 0


@pytest.mark.parametrize('fault', [None, 'asset', 'config', 'drop', 'frame'])
def test_full_reference_validation_and_b8_method(tmp_path: Path, fault: str | None) -> None:
    import json
    from scripts.analyze_live_b13 import impact
    from tests.test_live_b8 import sample_data
    paths = [tmp_path/'reference', tmp_path/'candidate']
    for index, path in enumerate(paths):
        path.mkdir()
        data = sample_data([0, 2, 4]) if index == 0 else sample_data([0, 4], [0, 2])
        for key in ('recognition_ms', 'cpu_ms', 'queue_put_ms'):
            data[key] = np.ones(len(data['t_sec']))
        if fault == 'frame' and index == 0:
            data['t_sec'][-1] = .05
        np.savez(path/'recognition.npz', **data)
        metrics = dict(assets=dict(app_build_id='changed' if fault == 'asset' and index else B12_BUILD),
            realtime=bool(index), capture_dropped_in_measured_window=int(fault == 'drop' or index == 1),
            frame_bounds=dict(start=0, end=3, stride=1, fps=30))
        (path/'metrics.json').write_text(json.dumps(metrics))
        (path/'recognition_config.json').write_text(json.dumps(dict(stable=4 if fault == 'config' and index else 3)))
        (path/'events.jsonl').write_text('')
    if fault:
        with pytest.raises(ValueError):
            impact(*paths, 0, .1)
    else:
        result = impact(*paths, 0, .1)
        assert result['boards']['placements'] == dict(reference_placements=2, matched=1, omitted=1, unresolved=0)
        assert result['physics']['chain_start']['matched'] == 0
