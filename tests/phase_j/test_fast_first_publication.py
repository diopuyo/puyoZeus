"""較正引継ぎと複数試合を、配信thread・Hub・実HTTPまで通して検査する。"""
from __future__ import annotations

from dataclasses import replace
import json
import pickle
from queue import Queue
import time
from typing import Any
from urllib.request import urlopen

import pytest

from src.phase_j.live_device_session import DeviceSession
from src.phase_j.live_publish import LivePublisher
from tests.phase_j.test_live_fast_first import session  # noqa: F401
from tests.phase_j.test_live_calibration import samples
from tests.phase_j.test_live_publish import ASSETS, row

MATCH_COUNT = 4
TIMEOUT_SEC = 2.0


def public_latest(publisher: LivePublisher, subscription: Queue) -> dict[str, Any]:
    """購読通知で公開完了を待ってから、OBSと同じHTTP最新値を読む。"""
    published = subscription.get(timeout=TIMEOUT_SEC)
    url = 'http://%s:%s/latest' % publisher.server.address()
    with urlopen(url, timeout=TIMEOUT_SEC) as response:
        latest = json.load(response)
    assert latest == published
    assert publisher.error is None and publisher.thread.is_alive()
    return latest


def input_events(session: DeviceSession, publisher: LivePublisher,
                 subscription: Queue) -> list[dict[str, Any]]:
    snapshots = []
    while not session.queue.empty():
        kind, status = session.queue.get_nowait()
        assert kind == 'hold'
        publisher.input_pending(status)
        if status['phase'] != 'ready':
            snapshots.append(public_latest(publisher, subscription))
    return snapshots


def publish_result(publisher: LivePublisher, subscription: Queue, game: int,
                   active: bool, probability: float | None) -> dict[str, Any]:
    """評価済み合成入力を通常のofferへ渡す。前試合の死亡値も経路へ含める。"""
    frame = len(publisher.publications) + 1
    stamp = time.perf_counter()
    candidate = dict(row(), game=game, frame=frame, t_sec=float(frame),
        captured_at=stamp, recognized_at=stamp, evaluated_at=stamp,
        match_active=active, raw_probability=probability,
        probability=probability, source='confirmed_death' if probability == 0 else 'static',
        judgment_id=(game, frame, active, probability))
    publisher.offer(candidate)
    return public_latest(publisher, subscription)


@pytest.mark.parametrize('enabled', [False, True])
@pytest.mark.parametrize('transport_hold', [False, True])
def test_four_matches_through_latest(session: DeviceSession, enabled: bool,
                                   transport_hold: bool) -> None:
    session.warmup.device = replace(session.warmup.device, fast_first_judgment=enabled)
    publisher = LivePublisher(ASSETS, '127.0.0.1', 0, completion_driven=True)
    subscription = publisher.state.subscribe()
    publisher.start()
    try:
        input_events(session, publisher, subscription)
        session.update(0, 0., None)
        input_events(session, publisher, subscription)
        for game in range(MATCH_COUNT):
            start = publish_result(publisher, subscription, game, True, None)
            assert start['display']['match_phase'] == 'playing'
            assert start['evaluations']['practical']['p1_win_probability'] is None
            assert 'b3_contents' not in start['evaluations']
            value = publish_result(publisher, subscription, game, True, .6)
            assert value['identity']['match_id'] == str(game)
            assert value['display']['input_status'] == 'ready'
            assert value['evaluations']['practical']['p1_win_probability'] == .6
            publish_result(publisher, subscription, game, True, 0.)
            end = publish_result(publisher, subscription, game, False, None)
            assert end['display']['match_phase'] == 'between_matches'
            assert end['evaluations']['practical']['availability'] == 'unavailable'
            if transport_hold:
                session.hold('no_puyo_screen')
                held = input_events(session, publisher, subscription)
                assert all(s['display']['calibration_progress'] < 100 for s in held)
                assert all(s['evaluations']['practical']['p1_win_probability'] is None for s in held)
                samples(session.warmup, session.warmup.target)
                session.warmup.advance(session.pipe._reader._classifier)
                session.update(game + 1, float(game + 1), None)
                input_events(session, publisher, subscription)
    finally:
        publisher.close()


@pytest.mark.parametrize('phase', ['verifying', 'no_puyo_screen', 'overlay_feedback'])
def test_retained_calibration_has_valid_public_progress(session: DeviceSession, phase: str) -> None:
    session.hold(phase)
    assert session.warmup.ready and session.warmup.progress == 100
    statuses = [session.queue.get_nowait()[1] for _ in range(session.queue.qsize())]
    assert all(status['progress'] < 100 for status in statuses)
    session.update(1, 1., None)
    assert session.last_status['phase'] == 'ready'
    assert session.last_status['progress'] == 100


@pytest.mark.parametrize('phase', [None, 'verifying', 'no_puyo_screen', 'overlay_feedback'])
@pytest.mark.parametrize('progress', [0, 41, 100])
def test_off_status_is_byte_identical(session: DeviceSession, phase: str | None,
                                    progress: int) -> None:
    warmup = session.warmup
    warmup.device = replace(warmup.device, fast_first_judgment=False)
    warmup.progress, warmup.ready = progress, progress == 100
    # 7036e39aのstatus()が返す辞書と、順序も含めて比較する。
    legacy = dict(phase=phase or ('ready' if warmup.ready else 'calibrating'),
                  progress=warmup.progress, cached=warmup.cached, mode=warmup.mode,
                  device_id=warmup.device.calibration_path.stem)
    assert pickle.dumps(warmup.status(phase)) == pickle.dumps(legacy)
