"""B9の故障境界・HOLD・計装・終了保護を検査する。"""
from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from src.phase_j.live_faults import FaultInjector, FaultPlan
from src.phase_j.live_source import CapturedFrame, VideoFileSource
from src.phase_j.live_video_session import VerifiedVideo
from scripts.analyze_live_b9 import lines, trends, fault_result


@pytest.fixture(autouse=True)
def isolate_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.phase_j.live_cpu import THREAD_ENV, NICE_ENV, POOL_ENV
    for key in (*POOL_ENV, THREAD_ENV, NICE_ENV, 'CUDA_VISIBLE_DEVICES'):
        before = os.environ.get(key)
        monkeypatch.setenv(key, before or '')
        if before is None:
            monkeypatch.delenv(key)


@pytest.mark.parametrize('kind,at,duration', [('unknown', 1, 1), ('black', -1, 1),
    ('black', float('nan'), 1), ('black', 0, 0), ('black', 0, -1), ('black', 0, float('inf'))])
def test_fault_rejects_invalid_plan(kind: str, at: float, duration: float) -> None:
    with pytest.raises(ValueError):
        FaultPlan(kind, at, duration)


def test_other_video_requires_existing_file() -> None:
    with pytest.raises(ValueError):
        FaultPlan('other', 0, video='missing-b9-video.avi')


@pytest.mark.parametrize('kind', ['black', 'resolution', 'repeat'])
def test_transform_is_bounded_and_emits_once(kind: str) -> None:
    events = []
    fault = FaultInjector(FaultPlan(kind, 1, 2), events.append)
    image = np.full((18, 32, 3), 200, np.uint8)
    assert fault.transform(image, 0) is image
    changed = fault.transform(image, 1)
    second = fault.transform(np.zeros_like(image), 2)
    assert fault.transform(image, 3) is image
    fault.transform(image, 4)
    assert [r['phase'] for r in events] == ['start', 'end']
    if kind == 'black':
        assert not changed.any()
    elif kind == 'resolution':
        assert changed.shape == (720, 1280, 3)
    else:
        assert np.array_equal(changed, second)


def test_stall_holds_before_wait_and_waits_once() -> None:
    seen = []
    fault = FaultInjector(FaultPlan('stall', 2, 5), lambda r: seen.append(r['phase']))
    hold = lambda phase: seen.append(phase)
    sleep = lambda duration: seen.append(duration)
    fault.before(1, sleep, hold)
    fault.before(2, sleep, hold)
    fault.before(3, sleep, hold)
    assert seen == ['start', 'verifying', 5, 'end']


def test_pause_preserves_causal_drop_count() -> None:
    from tests.phase_j.test_live_source import Capture
    now = [0.]
    source = VideoFileSource(Capture(), 30, 0, 300, 1, True,
        clock=lambda: now[0], sleep=lambda dt: now.__setitem__(0, now[0]+dt),
        fault=FaultInjector(FaultPlan('stall', 0, 5)))
    frame = next(iter(source))
    assert frame.index == 150 and source.dropped == 150
    assert frame.media_sec == 5 and frame.captured_at <= now[0]


def frame(stamp: float, value: int = 100, size: tuple = (1920, 1080)) -> CapturedFrame:
    return CapturedFrame(round(stamp*30), stamp, stamp, stamp,
        np.full((18, 32, 3), value, np.uint8), source_size=size)


def test_720p_metadata_survives_normalization() -> None:
    from tests.phase_j.test_live_source import Capture
    source = VideoFileSource(Capture(), 30, 0, 1, 1,
        fault=FaultInjector(FaultPlan('resolution', 0, 1)))
    result = next(iter(source))
    assert result.source_size == (1280, 720) and result.image.shape == (1080, 1920, 3)


def test_continuous_verifier_rejects_midstream_black_and_recovers() -> None:
    session = Mock()
    source = [frame(0), frame(1, 0), frame(2, 101)]
    video = VerifiedVideo(source, session, verifier=lambda image: bool(image.any()), continuous=True)
    assert [r.media_sec for r in video] == [0, 2]
    session.hold.assert_called_once_with('no_puyo_screen')


def test_repeated_pixels_hold_until_changed() -> None:
    session = Mock()
    video = VerifiedVideo([frame(0), frame(.5), frame(1), frame(2, 101)], session,
                          verifier=lambda image: True, continuous=True)
    assert [r.media_sec for r in video] == [0, .5, 2]
    session.hold.assert_called_once_with('verifying')


def test_resolution_switch_gates_both_directions() -> None:
    session = Mock()
    source = [frame(0), frame(.1, 101, (1280, 720)), frame(.2, 102, (1280, 720)), frame(.3, 103)]
    video = VerifiedVideo(source, session, verifier=lambda image: True, continuous=True)
    assert [r.media_sec for r in video] == [0, .2]
    assert session.hold.call_count == 2


@pytest.mark.parametrize('phase', ['verifying', 'calibrating', 'no_puyo_screen'])
def test_pending_input_never_exposes_probability(phase: str) -> None:
    from src.phase_j.live_publish import LivePublisher, result_snapshot
    from tests.phase_j.test_live_publish import ASSETS, row
    publisher = LivePublisher(ASSETS, '127.0.0.1', 0)
    publisher.offer(row())
    publisher.input_pending(dict(phase=phase, at=1, progress=20))
    snap = result_snapshot(publisher.initial, publisher.latest, 2, 1, 1)
    assert snap.evaluations['practical']['p1_win_probability'] is None
    assert snap.display['input_status'] == phase


@pytest.mark.parametrize('args,threads,nice', [([], 1, 10),
    (['--cpu-threads', '2', '--evaluation-nice', '0'], 2, 0), (['--worker-mode', 'thread'], 1, 0)])
def test_live_defaults_and_explicit_override(monkeypatch: pytest.MonkeyPatch,
                                            args: list, threads: int, nice: int) -> None:
    from scripts.run_live_pipeline_20260928 import parse_args
    monkeypatch.setattr(sys, 'argv', ['live', *args])
    options = parse_args()
    assert options.cpu_threads == threads and options.evaluation_nice == nice
    assert options.cnn_device == 'cpu'


@pytest.mark.parametrize('args', [['--video-fault', 'black'],
    ['--source', 'video', '--realtime', '--video-fault', 'black', '--fault-duration-sec', '-1'],
    ['--source', 'video', '--realtime', '--video-fault', 'black', '--fault-at-sec', '2700'],
    ['--source', 'video', '--realtime', '--video-fault', 'other']])
def test_fault_cli_rejects_incomplete_or_outside_interval(monkeypatch: pytest.MonkeyPatch, args: list) -> None:
    from scripts.run_live_pipeline_20260928 import parse_args
    monkeypatch.setattr(sys, 'argv', ['live', *args])
    with pytest.raises(SystemExit):
        parse_args()


def test_partial_live_json_line_is_not_parsed(tmp_path: Path) -> None:
    path = tmp_path/'live.jsonl'
    path.write_text('{"ok":1}\n{"incomplete":', encoding='utf-8')
    assert lines(path) == [dict(ok=1)]


@pytest.mark.parametrize('values,expected', [([1, 2, 3], True), ([1, 1, 2], False),
    ([3, 2, 1], False), ([None, None, None], False)])
def test_growth_assessment_does_not_invent_missing_measurements(values: list, expected: bool) -> None:
    keys = ['rss_p50', 'handles_p50', 'vram_p50', 'queue_p95', 'drop_fraction', 'latency_p95_ms']
    result = trends([{key: value for key in keys} for value in values])
    assert result['rss_p50']['strictly_increasing'] == expected
    assert len(result['vram_p50']['values']) == sum(v is not None for v in values)


def test_absent_fault_events_cannot_pass(tmp_path: Path) -> None:
    assert not fault_result(tmp_path)['passed']


def test_match_start_events_are_not_every_frame() -> None:
    from src.phase_j.live_bridge import RecognitionBridge
    bridge = RecognitionBridge(False, Mock())
    side = SimpleNamespace(confirmed_board=None)
    for game, active, stamp in [(1, True, 1), (1, True, 2), (1, False, 3), (2, True, 4)]:
        bridge.counter_context(SimpleNamespace(p1=side, p2=side, is_match_active=active), game, stamp, stamp)
    assert [r['t_sec'] for r in bridge.game_starts] == [1, 4]


def test_ctrl_c_is_forwarded_and_waited(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.phase_j.live_lifetime import join_evaluation, STOP_TIMEOUT_SEC
    worker = Mock(pid=123)
    worker.join.side_effect = [KeyboardInterrupt, None]
    worker.is_alive.side_effect = [True, False]
    kill = Mock()
    monkeypatch.setattr(os, 'kill', kill)
    with pytest.raises(KeyboardInterrupt):
        join_evaluation(worker)
    kill.assert_called_once_with(123, signal.SIGINT)
    worker.join.assert_called_with(STOP_TIMEOUT_SEC)


def test_owned_process_cleans_up_on_monitor_error(tmp_path: Path) -> None:
    from scripts.measure_live_b9 import owned_process
    with (tmp_path/'child.log').open('w') as log:
        with pytest.raises(ValueError):
            with owned_process([sys.executable, '-c', 'import time; time.sleep(30)'], log) as child:
                raise ValueError('監視側の例外')
    assert child.poll() is not None


@pytest.mark.skipif(not sys.platform.startswith('linux'), reason='WSL/Linuxの親死監視')
def test_parent_kill_terminates_protected_child(tmp_path: Path) -> None:
    script = tmp_path/'parent.py'
    script.write_text('import os,time\nfrom src.phase_j.live_lifetime import protect_parent\n'
        'parent=os.getpid()\nchild=os.fork()\nif child==0:\n protect_parent(parent)\n time.sleep(30)\n'
        'else:\n print(child,flush=True)\n time.sleep(30)\n')
    parent = subprocess.Popen([sys.executable, str(script)], stdout=subprocess.PIPE, text=True,
                              env=dict(os.environ, PYTHONPATH=str(Path.cwd())), start_new_session=True)
    try:
        child = int(parent.stdout.readline())
        time.sleep(.1)
        parent.kill()
        parent.wait(timeout=3)
        deadline = time.monotonic()+3
        while time.monotonic() < deadline:
            path = Path(f'/proc/{child}/stat')
            if not path.exists() or path.read_text().rsplit(')', 1)[1].split()[0] == 'Z':
                break
            time.sleep(.05)
        else:
            pytest.fail('保護した子processが稼働したままです')
    finally:
        try:
            os.killpg(parent.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        parent.wait()


def write_lines(path: Path, rows: list[dict]) -> None:
    path.write_text(''.join(json.dumps(r)+'\n' for r in rows), encoding='utf-8')


def sse(stamp: float, media: float, phase: str, visible: bool = False) -> dict:
    return dict(received_at=stamp, payload=dict(identity=dict(stream_seq=stamp, match_id='1'),
        timing=dict(source_available_ms=media*1000, capture_monotonic_sec=media),
        display=dict(input_status=phase, status='live' if visible else 'hold'), evaluations=dict(practical=dict(
            availability='available' if visible else 'unavailable', p1_win_probability=.6 if visible else None))))


@pytest.mark.parametrize('problem', [None, 'no_hold', 'stale', 'invalid_status', 'missing_phase', 'invalid_frame', 'hidden'])
def test_fault_report_checks_sse_not_just_process_exit(tmp_path: Path, problem: str | None) -> None:
    write_lines(tmp_path/'faults.jsonl', [dict(kind='black', phase='start', at=10, t_sec=10),
                                        dict(kind='black', phase='end', at=30, t_sec=30)])
    rows = [sse(11, 10, 'no_puyo_screen'), sse(30.1, 30, 'calibrating'), sse(32, 31, 'ready', True)]
    if problem == 'no_hold':
        rows.insert(1, sse(30.05, 30, 'ready', True))
    elif problem == 'stale':
        rows[-1] = sse(32, 29, 'ready', True)
    elif problem == 'invalid_status':
        rows[0] = sse(11, 10, 'no_puyo_screen', True)
    elif problem == 'missing_phase':
        rows[0] = sse(11, 10, 'verifying')
    elif problem == 'invalid_frame':
        rows.insert(1, sse(12, 11, 'ready', True))
    elif problem == 'hidden':
        rows[1]['payload']['display']['status'] = 'waiting'
    write_lines(tmp_path/'sse.jsonl', rows)
    report = fault_result(tmp_path)
    assert report['passed'] == (problem is None)
    assert report['recovery_sec'] == 2 or problem == 'no_hold'


def test_five_minute_windows_separate_drop_and_latency_samples(tmp_path: Path) -> None:
    from scripts.analyze_live_b9 import windows
    write_lines(tmp_path/'runtime.jsonl', [dict(at=t, progress=dict(t_sec=t), queue_depth=2) for t in (1, 299, 301, 599, 601, 699)])
    write_lines(tmp_path/'resources.jsonl', [dict(at=t, rss_bytes=100, handles=4, vram_mib=None) for t in (2, 302, 602)])
    (tmp_path/'recognition.json').write_text(json.dumps(dict(dropped_times=[10, 300, 650])))
    write_lines(tmp_path/'sse.jsonl', [sse(2.1, 2, 'ready', True), sse(302.2, 302, 'ready', True)])
    result = windows(tmp_path, 0, 700)
    assert [r['expected'] for r in result] == [9000, 9000, 3000]
    assert [r['dropped'] for r in result] == [1, 1, 1]
    assert result[0]['latency_p95_ms'] == pytest.approx(100)
    assert result[2]['latency_p95_ms'] is None and result[0]['vram_p50'] is None


def test_game_first_display_uses_actual_sse_receive_time(tmp_path: Path) -> None:
    from scripts.analyze_live_b9 import games
    write_lines(tmp_path/'runtime.jsonl', [dict(game_starts=[dict(game=1, t_sec=2, captured_at=100)])])
    write_lines(tmp_path/'sse.jsonl', [sse(101, 2, 'calibrating'), sse(102, 3, 'ready', True)])
    result = games(tmp_path, 0, 20)
    assert result['detected'] == 1 and not result['count_matches']
    assert result['rows'][0]['first_probability_delay_sec'] == 2
