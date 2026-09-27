"""Phase J公開契約とSSE latest-winsの接続試験。"""
from __future__ import annotations

import json
import time
from urllib.request import urlopen

import pytest

from src.phase_j.live_publish import LiveOverlayServer, LiveStreamState, initial_snapshot, result_snapshot
from src.phase_j.validator import validate_snapshot

ASSETS = {name: 'test' for name in ('app_build_id', 'recognition_model_hash',
    'recognition_config_hash', 'prediction_model_hash', 'calibration_hash')}


def row(source: str = 'G_fe') -> dict:
    return dict(frame=1, t_sec=1.0, game=0, captured_at=1.0, recognized_at=1.1,
                evaluated_at=1.2, queue_depth=0, raw_probability=0.6,
                probability=0.6, advantage=20.0, source=source, capture_gap=False,
                digest='sha256:test', hold=False)


@pytest.mark.parametrize('source', ['G_fe', 'S1', 'S3', 'S3_landing', 'unavoidable_death'])
@pytest.mark.parametrize('hold', [None, 1.3])
def test_public_dto_is_valid(source: str, hold: float | None) -> None:
    snapshot = result_snapshot(initial_snapshot(ASSETS), row(source), 1.5, 1, hold)
    report = validate_snapshot(snapshot)
    assert report.is_valid, report
    assert snapshot.evaluations['practical']['source'] == source


def test_unknown_probability_stays_hidden() -> None:
    data = row()
    data['raw_probability'] = None
    snapshot = result_snapshot(initial_snapshot(ASSETS), data, 1.5, 1, None)
    assert validate_snapshot(snapshot).is_valid
    assert snapshot.display['visibility'] == 'hidden'


def test_counter_pending_preserves_source_generation() -> None:
    from src.phase_j.live_counter import AsyncCounter
    from types import SimpleNamespace
    tracker = AsyncCounter(SimpleNamespace())
    tracker.generation, tracker.result_generation = 3, 1
    tracker.pending = True
    data = dict(row(), counter_search=tracker.status())
    snapshot = result_snapshot(initial_snapshot(ASSETS), data, 1.5, 1, None)
    assert validate_snapshot(snapshot).is_valid
    assert snapshot.evaluations['counter_search']['pending']
    assert snapshot.evaluations['counter_search']['result_generation'] == 1
    assert snapshot.evaluations['counter_search']['generation'] == 3
    data['counter_search']['pending'] = False
    invalid = result_snapshot(initial_snapshot(ASSETS), data, 1.5, 1, None)
    assert not validate_snapshot(invalid).is_valid


def test_pending_batch_is_one_job_with_all_notification_count() -> None:
    data = dict(row(), queue_depth=29)
    snapshot = result_snapshot(initial_snapshot(ASSETS), data, 1.5, 1, None)
    assert validate_snapshot(snapshot).is_valid
    assert snapshot.runtime['practical_queue_depth'] == 1
    assert snapshot.runtime['recognition_notification_queue_depth'] == 29


def test_slow_subscriber_gets_latest() -> None:
    state = LiveStreamState()
    queue = state.subscribe()
    initial = initial_snapshot(ASSETS)
    for revision in range(1, 5):
        state.publish(result_snapshot(initial, row(), 1.5, revision, None))
    assert queue.qsize() == 1
    assert queue.get()['identity']['reducer_revision'] == 4


def test_existing_http_sse_serves_phase_j() -> None:
    state = LiveStreamState()
    server = LiveOverlayServer(state, '127.0.0.1', 0)
    state.publish(result_snapshot(initial_snapshot(ASSETS), row(), 1.5, 1, None))
    server.start()
    url = 'http://%s:%s' % server.address()
    try:
        with urlopen(url + '/latest', timeout=2) as response:
            assert json.load(response)['evaluations']['practical']['source'] == 'G_fe'
        with urlopen(url + '/events', timeout=2) as response:
            assert response.readline().decode().strip() == 'event: analysis'
            assert json.loads(response.readline().decode()[6:])['schema_version'] == 'puyo-overlay-snapshot/v1'
    finally:
        server.stop()


def test_sse_connection_is_limited_to_two_hz() -> None:
    state = LiveStreamState()
    server = LiveOverlayServer(state, '127.0.0.1', 0)
    snapshot = result_snapshot(initial_snapshot(ASSETS), row(), 1.5, 1, None)
    state.publish(snapshot)
    server.start()
    try:
        with urlopen('http://%s:%s/events' % server.address(), timeout=2) as response:
            for _ in range(3):
                response.readline()
            first = time.perf_counter()
            state.publish(snapshot)
            response.readline()
            assert time.perf_counter()-first >= 0.45
    finally:
        server.stop()
