"""CPU配分とフレーム脱落監査の境界条件を検証する。"""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import os

import numpy as np
import pytest

from src.phase_j.live_cpu import configure_environment, apply_runtime, THREAD_ENV, NICE_ENV, POOL_ENV
from src.phase_j.live_audit import RecognitionAudit


@pytest.fixture(autouse=True)
def isolate_cpu_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """直接更新する環境変数も各テスト終了後に元へ戻す。"""
    for key in (*POOL_ENV, THREAD_ENV, NICE_ENV):
        before = os.environ.get(key)
        monkeypatch.setenv(key, before or '')
        if before is None:
            monkeypatch.delenv(key)


@pytest.mark.parametrize('threads', [-1, True, 1.5])
def test_invalid_threads(threads: object) -> None:
    with pytest.raises(ValueError):
        configure_environment(threads, 0)


@pytest.mark.parametrize('nice', [-1, 20, True, 1.5])
def test_invalid_nice(nice: object) -> None:
    with pytest.raises(ValueError):
        configure_environment(0, nice)


@pytest.mark.parametrize('threads', [0, 1, 2])
def test_environment(monkeypatch: pytest.MonkeyPatch, threads: int) -> None:
    for key in (*POOL_ENV, THREAD_ENV, NICE_ENV):
        monkeypatch.delenv(key, raising=False)
    configure_environment(threads, 10)
    assert os.environ[THREAD_ENV] == str(threads)
    assert os.environ[NICE_ENV] == '10'
    assert all(os.environ.get(key) == (str(threads) if threads else None) for key in POOL_ENV)


@pytest.mark.parametrize('role,lower,expected', [('recognition', True, 0),
    ('evaluation', False, 0), ('evaluation', True, 10), ('mc', True, 10)])
def test_recognition_never_lowers_priority(monkeypatch: pytest.MonkeyPatch,
                                          role: str, lower: bool, expected: int) -> None:
    monkeypatch.setenv(THREAD_ENV, '0')
    monkeypatch.setenv(NICE_ENV, '10')
    calls = []
    monkeypatch.setattr(os, 'nice', lambda delta: calls.append(delta) or 0, raising=False)
    apply_runtime(role, lower)
    assert sum(calls) == expected


def test_disabled_audit_does_not_deserialize() -> None:
    notice = SimpleNamespace(result=Mock(side_effect=AssertionError('不要な展開')))
    audit = RecognitionAudit(None)
    audit.append(notice, None, 0, 0, True)
    audit.queue_wait(0)
    audit.save({}, None)
    assert not audit.rows


def sample_data(counts: list[int], frames: list[int] | None = None) -> dict:
    boards = np.zeros((len(counts), 2, 13, 6), dtype=np.int8)
    for i, count in enumerate(counts):
        for cell in range(count):
            boards[i, 0, 12-cell//6, cell%6] = 1
    stamps = np.array(frames if frames is not None else list(range(len(counts))))/30
    return dict(boards=boards, raw=boards.copy(), t_sec=stamps,
        active=np.ones(len(counts), dtype=bool), stable=np.ones((len(counts), 2), dtype=bool),
        captured_at=stamps+100, recognized_at=stamps+100.02)


def test_alignment_uses_time_not_compressed_row_number() -> None:
    from scripts.analyze_live_b8 import board_events
    event = board_events(sample_data([0, 2, 2], [0, 30, 60]), 0)[1]
    assert event['frames'] == [30, 60]
    assert event['indices'] == [1, 2]


def test_missing_placement_requires_bracketing_anchors() -> None:
    from scripts.analyze_live_b8 import board_comparison
    report = board_comparison(sample_data([0, 2, 4]), sample_data([0, 4], [0, 2]))
    assert report['placements'] == dict(reference_placements=2, matched=1, omitted=1, unresolved=0)


@pytest.mark.parametrize('pairs', [[(0, 0, 'exact')], [(2, 1, 'exact')],
    [(0, 0, 'exact'), (2, 2, 'exact')]])
def test_unmatched_does_not_automatically_mean_missed(pairs: list) -> None:
    from scripts.analyze_live_b8 import classify_missing
    report = classify_missing([1], pairs, 3, 3)
    assert report['omitted'] == 0
    assert report['unresolved'] == 1


@pytest.mark.parametrize('change', ['ojama', 'erased', 'recolored', 'epoch', 'inactive'])
def test_only_pure_two_colored_cell_placements(change: str) -> None:
    from scripts.analyze_live_b8 import board_events, placement_indices
    sequence = board_events(sample_data([2, 4]), 0)
    if change == 'ojama':
        sequence[1]['board'][12, 2:] = 9
    elif change == 'erased':
        sequence[1]['board'][12, 0] = 0
    elif change == 'recolored':
        sequence[1]['board'][12, 0] = 2
    elif change == 'epoch':
        sequence[1]['epoch'] += 1
    else:
        sequence[1]['active'] = False
    assert placement_indices(sequence) == []


def test_raw_arrival_proxy_and_confirmation_latency_are_separate() -> None:
    import json
    from scripts.analyze_live_b8 import board_comparison
    reference = sample_data([0, 0, 2, 2])
    reference['raw'][1] = reference['boards'][2]
    candidate = sample_data([0, 2], [0, 7])
    report = board_comparison(reference, candidate)
    assert report['raw_proxy_placement_reference_frames']['P50'] == pytest.approx(1)
    assert report['raw_proxy_placement_candidate_frames']['P50'] == pytest.approx(6)
    assert report['placement_additional_delay_frames']['P50'] == pytest.approx(5)
    assert report['raw_proxy_placement_candidate_wall_frames']['P50'] == pytest.approx(6.6)
    json.dumps(report, allow_nan=False)


def test_one_raw_frame_is_not_an_arrival_proof() -> None:
    from scripts.analyze_live_b8 import board_comparison
    report = board_comparison(sample_data([0, 2]), sample_data([0, 2]))
    assert report['raw_proxy_unavailable'] == 1


@pytest.mark.parametrize('delta', [-.5, 0., .5])
def test_event_shift_has_signed_and_absolute_distributions(delta: float) -> None:
    from scripts.analyze_live_b8 import timing_result
    report = timing_result([dict(side='1P', t=1)], [dict(side='1P', t=1+delta)], [(0, 0)], 't')
    assert report['signed_ms']['P50'] == pytest.approx(delta*1000)
    assert report['absolute_ms']['P95'] == pytest.approx(abs(delta)*1000)


@pytest.mark.parametrize('field,value', [('side', '2P'), ('game', 2), ('t', 9)])
def test_event_matching_never_crosses_side_game_or_far_time(field: str, value: object) -> None:
    from scripts.analyze_live_b8 import match_times
    a = dict(side='1P', game=1, t=1)
    assert match_times([a], [dict(a, **{field: value})], 't') == []


def test_event_matching_chooses_closest_without_reusing_event() -> None:
    from scripts.analyze_live_b8 import match_times
    a = [dict(side='1P', game=1, t=t) for t in (0, 1, 2)]
    b = [dict(side='1P', game=1, t=1.1)]
    assert match_times(a, b, 't') == [(1, 0)]


def test_missing_timing_stays_out_of_denominator() -> None:
    from scripts.analyze_live_b8 import timing_result
    report = timing_result([dict(side='1P', t=1)], [dict(side='1P', t=None)], [(0, 0)], 't')
    assert report['matched'] == 0
    assert report['reference_events'] == 1 and report['candidate_events'] == 0


def test_raw_proxy_requires_consecutive_source_frames() -> None:
    from scripts.analyze_live_b8 import board_comparison
    reference = sample_data([0, 0, 2], [0, 1, 3])
    reference['raw'][1] = reference['boards'][2]
    report = board_comparison(reference, reference)
    assert report['raw_proxy_unavailable'] == 1


def test_early_confirmation_is_not_counted_as_meeting_latency_limit() -> None:
    from scripts.analyze_live_b8 import board_comparison
    reference = sample_data([0, 0, 2], [0, 4, 5])
    reference['raw'][1] = reference['boards'][2]
    report = board_comparison(reference, sample_data([0, 2], [0, 3]))
    assert report['raw_proxy_rejected_early'] == 1
    assert report['raw_proxy_placement_candidate_frames']['count'] == 0


def test_audit_serializes_copies_and_queue_wait(tmp_path: Path) -> None:
    import json
    grid = np.zeros((13, 6), dtype=np.int8)
    side = SimpleNamespace(confirmed_board=SimpleNamespace(_grid=grid),
        cnn_board=None, state=SimpleNamespace(name='STABLE'), score=None)
    notice = SimpleNamespace(frame=3, t_sec=.1, recognized_at=10.2, dropped_before=2,
        result=lambda: SimpleNamespace(p1=side, p2=side, is_match_active=True))
    audit = RecognitionAudit(tmp_path/'audit.npz')
    audit.append(notice, SimpleNamespace(acquired_at=10.1, captured_at=10.0), .2, .3, False)
    grid[:] = 1
    audit.queue_wait(.01)
    audit.save({'nice': 0}, SimpleNamespace(dropped=2, dropped_times=[.03, .06]))
    with np.load(tmp_path/'audit.npz') as rows:
        assert not rows['boards'].any()
        assert (rows['raw'] == -1).all()
        assert rows['queue_put_ms'].item() == pytest.approx(10)
        assert not rows['ready'].item()
    assert json.loads((tmp_path/'audit.json').read_text())['dropped'] == 2


def test_final_measurement_is_one_cpu_realtime_case() -> None:
    from scripts.measure_live_b8 import measurement_command
    command = measurement_command(Path('out'))
    for flag, expected in [('--cnn-device', 'cpu'), ('--cpu-threads', '1'),
                           ('--evaluation-nice', '10'), ('--start-sec', '2580'), ('--end-sec', '2700')]:
        assert command[command.index(flag)+1] == expected
    assert '--realtime' in command and '--recognition-audit' in command


def test_default_preserves_explicit_existing_thread_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in POOL_ENV:
        monkeypatch.setenv(key, '3')
    configure_environment(0, 0)
    assert all(os.environ[key] == '3' for key in POOL_ENV)


@pytest.mark.parametrize('invalid', [None, 'gpu', 'priority', 'threads'])
def test_final_measurement_checks_actual_runtime(invalid: str | None) -> None:
    from scripts.measure_live_b8 import verify_runtime
    runtime = {role: dict(nice=nice, torch_threads=1, torch_interop_threads=1, opencv_threads=1)
               for role, nice in [('recognition', 0), ('evaluation', 10)]}
    result = dict(gpu=None, runtime=runtime)
    if invalid == 'gpu':
        result['gpu'] = 'NVIDIA'
    elif invalid == 'priority':
        runtime['recognition']['nice'] = 10
    elif invalid == 'threads':
        runtime['evaluation']['torch_threads'] = 2
    assert verify_runtime(result) == (invalid is None)


def test_audit_window_excludes_warmup_and_end(tmp_path: Path) -> None:
    from scripts.analyze_live_b8 import load_audit
    path = tmp_path/'recognition.npz'
    np.savez(path, t_sec=[2599., 2600., 2639., 2640.], frame=[0, 1, 2, 3])
    data = load_audit(path, 2600., 2640.)
    assert data['frame'].tolist() == [1, 2]
