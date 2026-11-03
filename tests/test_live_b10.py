"""形式逸脱の持続遮断、停止優先、送信直前の失効、既存ログ再集計を検証する。"""
from __future__ import annotations

from io import BytesIO
import json
from pathlib import Path
from unittest.mock import Mock

import pytest
import numpy as np

from src.phase_j.live_video_session import VerifiedVideo
from src.phase_j.live_publish import LiveOverlayServer, LiveStreamState, LivePublisher, initial_snapshot, result_snapshot
from tests.phase_j.test_live_publish import ASSETS, row
from tests.test_live_b9 import frame, sse, write_lines
from scripts.analyze_live_b9 import windows, fault_result


@pytest.mark.parametrize('size', [(1280, 720), (1600, 900), (1920, 1200)])
@pytest.mark.parametrize('valid', [True, False])
def test_format_contract_blocks_whole_changed_interval(size: tuple, valid: bool) -> None:
    session, verifier = Mock(), Mock(return_value=valid)
    source = [frame(0, 1), frame(1, 2, size), frame(20, 3, size),
              frame(21, 4), frame(21.5, 5), frame(22.1, 6)]
    video = VerifiedVideo(source, session, verifier, continuous=True)
    assert [r.media_sec for r in video] == ([0, 22.1] if valid else [])
    assert verifier.call_count == 2
    assert all(call.args == ('verifying',) for call in session.hold.call_args_list[int(not valid):4+int(not valid)])


@pytest.mark.parametrize('size', [(1280, 720), (1920, 1080), (1600, 900)])
def test_new_session_can_verify_its_initial_format(size: tuple) -> None:
    source = [frame(t, t+1, size) for t in range(3)]
    assert len(list(VerifiedVideo(source, Mock(), lambda image: True, continuous=True))) == 3


@pytest.mark.parametrize('valid', [True, False])
def test_frozen_non_puyo_is_still_an_input_failure(valid: bool) -> None:
    session, verifier = Mock(), Mock(return_value=valid)
    source = [frame(0), frame(1), frame(19), frame(20, 101), frame(20.5, 102), frame(21.1, 103)]
    output = list(VerifiedVideo(source, session, verifier, continuous=True))
    assert [r.media_sec for r in output] == ([0, 21.1] if valid else [])
    assert verifier.call_count == 2
    assert sum(call.args == ('verifying',) for call in session.hold.call_args_list) == 4


def test_recovery_instability_restarts_verification() -> None:
    size = (1280, 720)
    source = [frame(0, 1), frame(1, 2, size), frame(2, 3), frame(2.5, 4, size),
              frame(3, 5), frame(3.9, 6), frame(4.1, 7)]
    assert [r.media_sec for r in VerifiedVideo(source, Mock(), lambda image: True, True)] == [0, 4.1]


@pytest.mark.parametrize('kind', ['resolution', 'repeat'])
def test_normal_video_also_blocks_input_failure(kind: str) -> None:
    abnormal = frame(1, 2, (1280, 720)) if kind == 'resolution' else frame(1, 1)
    source = [frame(0, 1), abnormal, frame(2, 3), frame(3.1, 4)]
    session, verifier = Mock(), Mock(return_value=True)
    assert [r.media_sec for r in VerifiedVideo(source, session, verifier)] == [0, 3.1]
    assert verifier.call_count == 2
    assert session.hold.call_count == 2


@pytest.mark.parametrize('kind', ['resolution', 'repeat'])
def test_device_uses_same_transport_guard(kind: str) -> None:
    from src.phase_j.live_device import DirectShowSource, DeviceConfig
    from tests.phase_j.test_live_device import FakeDevice
    clock, holds = [0.], []
    capture = FakeDevice(None)
    capture.isOpened = lambda: True
    def read() -> tuple:
        at = clock[0]
        size = (180, 320, 3) if kind == 'resolution' and 1 <= at < 3 else (108, 192, 3)
        value = 50 if kind == 'repeat' and 1 <= at < 3 else int(at*100) % 255
        return True, np.full(size, value, np.uint8)
    def sleep(seconds: float) -> None:
        clock[0] += seconds
    capture.read = read
    source = DirectShowSource(DeviceConfig('test', 0), 4.5, lambda: None,
        lambda *args: capture, lambda image: True, lambda: clock[0], sleep,
        on_status=lambda phase: holds.append((clock[0], phase)))
    frames = list(source)
    assert capture.released
    assert all(not 2.1 <= r.media_sec < 4 for r in frames)
    assert any(r.media_sec > 4 for r in frames)
    assert holds and all(phase == 'verifying' for _, phase in holds)
    assert all(r.source_size == (192, 108) for r in frames)


def payload(revision: int = 1, phase: str = 'ready', captured: float = 1.) -> dict:
    value = dict(row(), captured_at=captured, input_calibration=dict(phase=phase, progress=0))
    data = result_snapshot(initial_snapshot(ASSETS), value, 10, revision, None).to_mapping()
    data['identity']['stream_seq'] = revision
    return data


@pytest.mark.parametrize('captured,allowed', [(1., False), (2., False), (2.1, True)])
def test_invalidated_capture_is_not_sendable(captured: float, allowed: bool) -> None:
    server = LiveOverlayServer(LiveStreamState(), '127.0.0.1', 0)
    server.invalidated_at = 2.
    assert (server.sendable(payload(captured=captured)) is not None) == allowed


@pytest.mark.parametrize('phase', ['verifying', 'calibrating', 'no_puyo_screen'])
def test_latest_hold_replaces_prefetched_probability(phase: str) -> None:
    state = LiveStreamState()
    state._latest_payload = payload(2, phase)
    server = LiveOverlayServer(state, '127.0.0.1', 0)
    sent = server.sendable(payload())
    assert sent['identity']['stream_seq'] == 2
    assert sent['evaluations']['practical']['p1_win_probability'] is None


@pytest.mark.parametrize('publish_hold', [True, False])
def test_rate_limit_wait_rechecks_input_before_socket_write(monkeypatch: pytest.MonkeyPatch,
                                                            publish_hold: bool) -> None:
    state = LiveStreamState()
    server = LiveOverlayServer(state, '127.0.0.1', 0)
    handler = object.__new__(server._make_handler_class())
    handler.wfile = BytesIO()
    def invalidate(seconds: float) -> None:
        server.invalidated_at = 2.
        if publish_hold:
            state._latest_payload = payload(2, 'verifying')
    monkeypatch.setattr('src.phase_j.live_publish.time.sleep', invalidate)
    handler._write_sse('analysis', payload())
    content = handler.wfile.getvalue()
    if publish_hold:
        data = json.loads(content.splitlines()[1][len(b'data: '):])
        assert data['evaluations']['practical']['availability'] == 'unavailable'
    else:
        assert content == b'' and not server.sent


def test_ready_cannot_revive_capture_before_invalidated_epoch() -> None:
    publisher = LivePublisher(ASSETS, '127.0.0.1', 0)
    publisher.input_pending(dict(phase='verifying', progress=0, at=2.))
    publisher.input_pending(dict(phase='ready', progress=100, at=3.))
    assert publisher.server.sendable(payload()) is None
    assert publisher.server.sendable(payload(captured=3.1)) is not None


@pytest.mark.parametrize('start,expected', [(0., 20999), (2580.566, 20999), (2580.566, 21000)])
def test_window_json_uses_native_scalars_and_actual_total(tmp_path: Path, start: float, expected: int) -> None:
    (tmp_path/'metrics.json').write_text(json.dumps(dict(expected_frames=expected)))
    (tmp_path/'recognition.json').write_text(json.dumps(dict(dropped_times=[start+1, start+301, start+601])))
    rows = windows(tmp_path, start, start+700)
    encoded = json.dumps(rows, allow_nan=False)
    assert json.loads(encoded) == rows
    assert sum(r['expected'] for r in rows) == expected
    assert sum(r['dropped'] for r in rows) == 3
    assert rows[2]['expected'] == expected-18000


@pytest.mark.parametrize('kind', ['resolution', 'repeat'])
@pytest.mark.parametrize('source_time', [9., 12.])
def test_any_fault_window_probability_is_a_failure(tmp_path: Path, kind: str, source_time: float) -> None:
    write_lines(tmp_path/'faults.jsonl', [dict(kind=kind, phase='start', at=10., t_sec=10.),
                                        dict(kind=kind, phase='end', at=30., t_sec=30.)])
    write_lines(tmp_path/'sse.jsonl', [sse(10.1, 10, 'verifying'), sse(12, source_time, 'ready', True),
        sse(30.1, 30, 'calibrating'), sse(32, 31, 'ready', True)])
    result = fault_result(tmp_path)
    assert not result['passed'] and result['available_during_fault'] == 1
    assert result['invalid_frame_probability_count'] == int(source_time == 12)


@pytest.mark.parametrize('idle_ok', [True, False])
def test_retry_never_runs_long_or_accepted_cases(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                              idle_ok: bool) -> None:
    from scripts.measure_live_b10 import run_cases
    monkeypatch.setattr('os.nice', lambda value: 0, raising=False)
    run = Mock(return_value=dict(passed=True))
    idle = Mock(return_value=idle_ok)
    run_cases(tmp_path, 'test', idle=idle, run=run)
    assert [call.args[0] for call in run.call_args_list] == (['resolution', 'repeat'] if idle_ok else [])
    assert idle.call_count == 1


def test_retry_records_failure_without_skipping_second_case(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from scripts.measure_live_b10 import run_cases
    monkeypatch.setattr('os.nice', lambda value: 0, raising=False)
    run = Mock(side_effect=[ValueError('failure'), dict(passed=True)])
    run_cases(tmp_path, 'test', idle=lambda *args: True, run=run)
    result = json.loads((tmp_path/'status.json').read_text())
    assert not result['passed'] and result['results']['repeat']['passed']
