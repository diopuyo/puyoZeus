"""評価が止まっても公開側はholdへ進む。決着保持とOFF互換も固定する。"""
from __future__ import annotations

import json
from pathlib import Path
import time
from urllib.request import urlopen
from unittest.mock import patch

import pytest

from src.phase_j.live_publish import LivePublisher, initial_snapshot, result_snapshot
from src.phase_j.live_freshness import EVALUATION_FRESHNESS_SEC
from src.phase_j.launcher import build_pipeline_config, parse_user_config, LauncherConfigError
from src.phase_j.validator import validate_snapshot
from tests.phase_j.test_live_publish import ASSETS, row

TEST_WAIT_SEC = 4.0
TEST_POLL_SEC = 0.02


@pytest.mark.parametrize('age,expected', [(0., 'live'), (1.999, 'live'), (2., 'hold'), (30., 'hold')])
def test_age_gate_changes_status_only(age: float, expected: str) -> None:
    data = row()
    snapshot = result_snapshot(initial_snapshot(ASSETS), data, data['evaluated_at']+age,
                               1, None, freshness_gate=True)
    assert validate_snapshot(snapshot).is_valid
    assert snapshot.display['status'] == expected
    assert snapshot.evaluations['practical']['p1_win_probability'] == data['probability']
    assert data == row()
    if expected == 'hold':
        assert snapshot.display['primary_hold_reason'] == 'calculation_pending'
        assert '評価更新待ち' in snapshot.display['message']


@pytest.mark.parametrize('probability', [0., 1.])
def test_death_source_is_exempt_but_model_extremes_are_not(probability: float) -> None:
    data = dict(row('confirmed_death'), probability=probability, raw_probability=probability)
    snapshot = result_snapshot(initial_snapshot(ASSETS), data, 100., 1, None, True)
    assert snapshot.display['status'] == 'live'
    data['source'] = 'G_fe'
    assert result_snapshot(initial_snapshot(ASSETS), data, 100., 1, None, True).display['status'] == 'hold'


def test_frozen_b3_media_clock_does_not_hide_stale_age() -> None:
    data = dict(row(), b3_state_t_sec=1., b3_state_frame=1)
    snapshot = result_snapshot(initial_snapshot(ASSETS), data, 31.2, 1, None, True)
    assert snapshot.timing['calculation_age_ms'] == pytest.approx(30000.)
    assert snapshot.evaluations['practical']['calculation_age_ms'] == pytest.approx(30000.)
    assert validate_snapshot(snapshot).is_valid


@pytest.mark.parametrize('changes', [dict(match_active=False), dict(raw_probability=None),
    dict(input_calibration=dict(phase='verifying', progress=0)), dict(evaluation_error=True)])
def test_existing_nonlive_reason_kept(changes: dict) -> None:
    initial, data = initial_snapshot(ASSETS), dict(row(), **changes)
    with patch('src.phase_j.live_publish.utc_now', return_value='2026-10-12T00:00:00+00:00'):
        assert result_snapshot(initial, data, 100., 1, None, True) == result_snapshot(
            initial, data, 100., 1, None, False)


def wait_status(publisher: LivePublisher, expected: str) -> dict:
    deadline = time.perf_counter()+TEST_WAIT_SEC
    while time.perf_counter() < deadline:
        latest = publisher.state.latest()
        if latest and latest['display']['status'] == expected:
            return latest
        assert publisher.error is None
        time.sleep(TEST_POLL_SEC)
    raise AssertionError(f'{expected}へ更新されません')


@pytest.mark.parametrize('completed,evaluation', [(True, True), (False, True), (False, False)])
def test_worker_timeout_publishes_hold_and_new_evaluation_recovers(completed: bool, evaluation: bool) -> None:
    publisher = LivePublisher(ASSETS, '127.0.0.1', 0, completion_driven=completed,
                              evaluation_driven=evaluation, freshness_gate=True)
    publisher.start()
    try:
        stamp = time.perf_counter()
        data = dict(row(), captured_at=stamp, recognized_at=stamp, evaluated_at=stamp)
        publisher.offer(data)
        wait_status(publisher, 'live')
        wait_status(publisher, 'hold')
        # 周期公開経路は既存のholdが先行するため、鮮度理由まで待つ。
        deadline = stamp+TEST_WAIT_SEC
        while '評価更新待ち' not in publisher.state.latest()['display'].get('message', ''):
            assert time.perf_counter() < deadline
            time.sleep(TEST_POLL_SEC)
        host, port = publisher.server.address()
        with urlopen(f'http://{host}:{port}/latest', timeout=2) as response:
            latest = json.load(response)
        assert latest['display']['status'] == 'hold'
        assert latest['evaluations']['practical']['p1_win_probability'] == data['probability']
        publisher.offer(dict(data, frame=2, evaluated_at=time.perf_counter()))
        wait_status(publisher, 'live')
    finally:
        publisher.close()


def test_completion_off_keeps_old_live_without_timeout() -> None:
    publisher = LivePublisher(ASSETS, '127.0.0.1', 0, completion_driven=True)
    publisher.start()
    try:
        stamp = time.perf_counter()-EVALUATION_FRESHNESS_SEC*2
        publisher.offer(dict(row(), captured_at=stamp, recognized_at=stamp, evaluated_at=stamp))
        latest = wait_status(publisher, 'live')
        time.sleep(TEST_POLL_SEC*2)
        assert publisher.state.latest() == latest
    finally:
        publisher.close()


@pytest.mark.parametrize('enabled', [True, False])
def test_json_roundtrip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, enabled: bool) -> None:
    from scripts.run_live_pipeline_20260928 import parse_args
    config = build_pipeline_config(parse_user_config(
        dict(device_name='OBS', evaluation_freshness_gate=enabled), tmp_path))
    assert ('evaluation_freshness_gate' in config) == enabled
    path = tmp_path / 'pipeline.json'
    path.write_text(json.dumps(config), encoding='utf-8')
    monkeypatch.setattr('sys.argv', ['live', '--config', str(path)])
    assert parse_args().evaluation_freshness_gate is enabled


@pytest.mark.parametrize('value', [None, 1, 'true'])
def test_config_bool_required(tmp_path: Path, value: object) -> None:
    with pytest.raises(LauncherConfigError):
        parse_user_config(dict(device_name='OBS', evaluation_freshness_gate=value), tmp_path)
