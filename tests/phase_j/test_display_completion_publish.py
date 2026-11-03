"""判定確定時だけの送信・空送りなし・旧送信経路の互換を確認する。"""
from __future__ import annotations

from io import BytesIO
from pathlib import Path
from queue import Empty
import time

import pytest

from src.phase_j.live_publish import LivePublisher, LiveOverlayServer, LiveStreamState
from src.phase_j.live_publish import initial_snapshot, result_snapshot
from tests.phase_j.test_live_publish import ASSETS, row


def candidate(frame: int = 1) -> dict:
    stamp = time.perf_counter()
    return dict(row(), frame=frame, judgment_id=(0, frame, 'static', .6, True),
                captured_at=stamp, recognized_at=stamp, evaluated_at=stamp)


def test_only_new_judgments_publish_and_close_does_not_repeat() -> None:
    publisher = LivePublisher(ASSETS, '127.0.0.1', 0, completion_driven=True)
    subscription = publisher.state.subscribe()
    publisher.thread.start()
    try:
        with pytest.raises(Empty):
            subscription.get(timeout=.02)
        publisher.offer(candidate())
        first = subscription.get(timeout=2.)
        publisher.offer(dict(candidate(), frame=2, evaluated_at=time.perf_counter()))
        assert not publisher.ready.is_set()
        publisher.offer(candidate(3))
        second = subscription.get(timeout=2.)
        assert first['evaluations']['practical']['p1_win_probability'] == .6
        assert second['identity']['stream_seq'] == first['identity']['stream_seq'] + 1
    finally:
        publisher.close()
    assert len(publisher.publications) == 2
    assert subscription.empty()


def test_wait_state_does_not_send_periodic_empty_snapshots() -> None:
    publisher = LivePublisher(ASSETS, '127.0.0.1', 0, completion_driven=True)
    subscription = publisher.state.subscribe()
    publisher.thread.start()
    try:
        status = dict(phase='calibrating', progress=10, at=time.perf_counter())
        publisher.input_pending(status)
        snapshot = subscription.get(timeout=2.)
        assert snapshot['display']['input_status'] == 'calibrating'
        publisher.input_pending(dict(status, at=time.perf_counter()))
        assert not publisher.ready.is_set()
        with pytest.raises(Empty):
            subscription.get(timeout=.6)
    finally:
        publisher.close()
    assert len(publisher.publications) == 1


def test_failure_is_published_once() -> None:
    from types import SimpleNamespace
    publisher = LivePublisher(ASSETS, '127.0.0.1', 0, completion_driven=True)
    subscription = publisher.state.subscribe()
    publisher.thread.start()
    try:
        publisher.evaluation_hold(SimpleNamespace(**candidate()), 0)
        assert subscription.get(timeout=2.)['display']['message'] == '評価エラー'
    finally:
        publisher.close()
    assert len(publisher.publications) == 1


def test_event_sse_has_no_periodic_wait_or_duplicate(monkeypatch: pytest.MonkeyPatch) -> None:
    state = LiveStreamState()
    server = LiveOverlayServer(state, '127.0.0.1', 0, completion_driven=True)
    handler = server._make_handler_class().__new__(server._make_handler_class())
    handler.wfile = BytesIO()
    handler.last_analysis = time.perf_counter()
    def no_sleep(seconds: float) -> None:
        raise AssertionError('イベント送信では周期待ちしない')
    monkeypatch.setattr(time, 'sleep', no_sleep)
    first = result_snapshot(initial_snapshot(ASSETS), candidate(), time.perf_counter(), 1, None)
    state.publish(first)
    handler._write_sse('analysis', first.to_mapping())
    handler._write_sse('analysis', first.to_mapping())
    handler._write_sse('heartbeat', dict(ts=0.))
    assert handler.wfile.getvalue().count(b'event: analysis') == 1
    assert handler.wfile.getvalue().count(b'event: heartbeat') == 1


def test_event_html_uses_connection_heartbeat() -> None:
    on = LiveOverlayServer(LiveStreamState(), '127.0.0.1', 0, completion_driven=True)
    off = LiveOverlayServer(LiveStreamState(), '127.0.0.1', 0)
    assert 'const COMPLETION_DRIVEN = true;' in on._make_handler_class().overlay_html
    assert 'const COMPLETION_DRIVEN = false;' in off._make_handler_class().overlay_html


def test_defaults_are_adopted() -> None:
    from src.production_config import DISPLAY_COMPLETION_DRIVEN_DEFAULT, DISPLAY_LATENCY_CANDIDATES
    assert DISPLAY_COMPLETION_DRIVEN_DEFAULT is True
    assert all(flag.adopted == '2026-10-10' for flag in DISPLAY_LATENCY_CANDIDATES)


def test_replaced_candidate_does_not_repeat_new_input_hold() -> None:
    publisher = LivePublisher(ASSETS, '127.0.0.1', 0, completion_driven=True)
    publisher.offer(candidate())
    previous = publisher.latest
    publisher.input_pending(dict(phase='verifying', progress=0, at=time.perf_counter()))
    publisher._publish(previous, 0, None)
    assert not publisher.publications


@pytest.mark.parametrize('value', ['true', 1, None, True])
def test_launcher_rejects_invalid_or_non_b3_event_flag(tmp_path: Path, value: object) -> None:
    from src.phase_j.launcher import parse_user_config, LauncherConfigError
    if value is True:
        assert parse_user_config(dict(device_name='OBS', display_completion_driven=value),
                                 tmp_path).display_completion_driven is False
        return
    with pytest.raises(LauncherConfigError, match='display_completion_driven'):
        parse_user_config(dict(device_name='OBS', display_completion_driven=value), tmp_path)


def test_launcher_forwards_explicit_event_flag(tmp_path: Path) -> None:
    from src.phase_j.launcher import parse_user_config, build_pipeline_config
    (tmp_path / 'model.joblib').touch()
    (tmp_path / 'timing.json').touch()
    user = parse_user_config(dict(device_name='OBS', b3_judgment=True, b3_async=True,
        b3_model='model.joblib', b3_timing_table='timing.json', display_completion_driven=True), tmp_path)
    assert build_pipeline_config(user)['display_completion_driven'] is True
