"""実機を使わず、較正だけの引継ぎと空盤面からの通知再開を検証する。"""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from queue import Queue
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from src.board import Board
from src.phase_j.live_calibration import ColorWarmup
from src.phase_j.live_device import DeviceConfig, DirectShowSource
from src.phase_j.live_device_session import DeviceSession
from src.phase_j.launcher import parse_user_config, build_pipeline_config, LauncherConfigError
from tests.phase_j.test_live_calibration import samples


@pytest.fixture
def session(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> DeviceSession:
    monkeypatch.setenv('PUYO_CALIBRATION_DIR', str(tmp_path / 'calibration'))
    hsv = SimpleNamespace(_ranges={'base': 1}, _native_ranges_cache=None, _native_params_cache=None,
                          set_color_ranges_from_simple=Mock())
    classifier = SimpleNamespace(_hsv=hsv, predict_proba_and_hsv_grid=Mock(return_value=(None, None)))
    pipe = SimpleNamespace(reset=Mock(), update=Mock(), _online_hsv=None,
                           _reader=SimpleNamespace(_classifier=classifier))
    result = SimpleNamespace(is_match_active=True,
        p1=SimpleNamespace(state=SimpleNamespace(name='STABLE'), confirmed_board=Board()),
        p2=SimpleNamespace(state=SimpleNamespace(name='STABLE'), confirmed_board=Board()))
    pipe.update.return_value = result
    instance = DeviceSession(pipe, tmp_path / 'input.json', 1, Queue())
    instance.switch(DeviceConfig('OBS', 0, fast_first_judgment=True))
    samples(instance.warmup, instance.warmup.target)
    instance.warmup.advance(classifier)
    return instance


@pytest.mark.parametrize('phase', ['no_puyo_screen', 'verifying', 'overlay_feedback'])
@pytest.mark.parametrize('verification', [True, False])
def test_hold_keeps_calibration_but_resets_history_once(
        session: DeviceSession, phase: str, verification: bool) -> None:
    session.warmup.verification_only = verification
    warmup = session.warmup
    session.pipe._reader._classifier._hsv._ranges = {'calibrated': 2}
    for _ in range(3):
        session.hold(phase)
    assert session.warmup is warmup and warmup.ready
    assert not session.publishing_ready
    assert session.epoch == 1 and session.pipe.reset.call_count == 2
    assert session.pipe._reader._classifier._hsv._ranges == {'calibrated': 2}
    result = session.update(1, 0., None)
    assert session.publishing_ready and result.p1.confirmed_board == session.pipe.update.return_value.p1.confirmed_board
    assert session.pipe.reset.call_count == 2
    assert session.pipe._online_hsv_injected_colors == {1, 2, 3, 4}


def test_empty_then_first_three_placements_are_not_gated(session: DeviceSession) -> None:
    session.hold('no_puyo_screen')
    for hand in range(4):
        result = session.pipe.update.return_value
        for side in (result.p1, result.p2):
            side.confirmed_board = Board()
            for column in range(hand * 2):
                side.confirmed_board.set(12, column, 1)
        assert session.update(hand, float(hand), None) is result
        assert session.publishing_ready
    assert session._recheck is not None and not session._recheck.ready


@pytest.mark.parametrize('change', ['name', 'index', 'generation'])
def test_real_input_switch_always_rechecks(session: DeviceSession, change: str) -> None:
    config = session.warmup.device
    if change == 'name':
        config = replace(config, name='Another')
    if change == 'index':
        config = replace(config, index=1)
    session.switch(config)
    assert not session.warmup.ready and not session.publishing_ready
    assert session.epoch == 1
    assert session.pipe._reader._classifier._hsv._ranges == {'base': 1}
    session.update(0, 0., None)
    assert not session.publishing_ready


@pytest.mark.parametrize('size', [(1280, 720), (640, 480)])
def test_format_change_invalidates_even_if_returned_to_original(
        session: DeviceSession, size: tuple[int, int]) -> None:
    source = DirectShowSource(session.warmup.device, 1, Mock())
    source.on_format_change = session.invalidate_format
    source._check_format((1920, 1080))
    assert not session._calibration_invalid
    source._check_format(size)
    source._check_format((1920, 1080))
    session.hold('verifying')
    assert not session.warmup.ready and session.epoch == 1


@pytest.mark.parametrize('shift', [0, 10])
def test_existing_profile_comparison_runs_without_blocking(
        session: DeviceSession, shift: int, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr('src.phase_j.live_device_session.FAST_FIRST_BACKGROUND_RECHECK', False)
    session.hold('no_puyo_screen')
    session.update(0, 0., None)
    assert session.publishing_ready
    probe = session._recheck
    samples(probe, probe.target, shift=shift)
    session.update(1, 1., None)
    if shift:
        assert session.warmup is probe and probe.changed_profile
        assert not session.publishing_ready and not probe.ready
        assert probe.target == probe.full_target and session.epoch == 1
        samples(probe, probe.full_target, shift=shift)
        session.update(2, 2., None)
        assert session.publishing_ready
    else:
        assert session.publishing_ready and session.epoch == 1
        assert session._recheck is None and not session._needs_recheck


@pytest.mark.parametrize('phase', ['no_puyo_screen', 'verifying', 'overlay_feedback'])
def test_off_still_discards_calibration(session: DeviceSession, phase: str) -> None:
    session.warmup.device = replace(session.warmup.device, fast_first_judgment=False)
    previous = session.warmup
    session.hold(phase)
    assert session.warmup is not previous and not session.warmup.ready
    assert session.warmup.cached and session.epoch == 1
    session.hold(phase)
    assert session.epoch == 1
    session.update(0, 0., None)
    assert not session.publishing_ready


def test_partial_samples_survive_without_enabling_publication(session: DeviceSession) -> None:
    session.switch(session.warmup.device)
    samples(session.warmup, 1)
    session.warmup.advance(session.pipe._reader._classifier)
    warmup = session.warmup
    session.hold('no_puyo_screen')
    session.update(0, 0., None)
    assert session.warmup is warmup and not session.publishing_ready


def test_format_change_during_hold_does_not_duplicate_boundary(session: DeviceSession) -> None:
    session.hold('no_puyo_screen')
    session.invalidate_format()
    session.hold('verifying')
    assert session.epoch == 1 and not session.warmup.ready


@pytest.mark.parametrize('verification', [True, False])
def test_drift_completed_in_one_observation_still_discards_old_result(
        session: DeviceSession, verification: bool) -> None:
    session.warmup.verification_only = verification
    session.hold('no_puyo_screen')
    session.update(0, 0., None)
    probe = session._recheck
    samples(probe, probe.full_target, shift=10)
    session.update(1, 1., None)
    if verification:
        assert probe.ready and session.warmup is probe and session.publishing_ready
        assert session.epoch == 1
        return
    assert probe.ready and session.warmup is probe and not session.publishing_ready
    assert session.epoch == 1
    if not verification:
        assert session.pipe._reader._classifier._hsv.set_color_ranges_from_simple.call_count == 2
    session.update(2, 2., None)
    assert session.publishing_ready


@pytest.mark.parametrize('enabled', [False, True])
def test_launcher_roundtrip_and_off_has_no_new_key(tmp_path: Path, enabled: bool) -> None:
    data = dict(device_name='OBS', fast_first_judgment=enabled)
    config = build_pipeline_config(parse_user_config(data, tmp_path))
    assert ('fast_first_judgment' in config) == enabled
    path = tmp_path / 'pipeline.json'
    path.write_text(json.dumps(config), encoding='utf-8')
    assert DeviceConfig.load(path).fast_first_judgment == enabled
    if not enabled:
        assert config == build_pipeline_config(parse_user_config(dict(device_name='OBS'), tmp_path))


@pytest.mark.parametrize('value', ['true', 1, None])
def test_non_boolean_config_rejected(tmp_path: Path, value: object) -> None:
    with pytest.raises(LauncherConfigError):
        parse_user_config(dict(device_name='OBS', fast_first_judgment=value), tmp_path)
    with pytest.raises(ValueError):
        DeviceConfig('OBS', 0, fast_first_judgment=value)


def test_generated_config_is_accepted_by_live_parser(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from scripts.run_live_pipeline_20260928 import parse_args
    config = build_pipeline_config(parse_user_config(dict(device_name='OBS', fast_first_judgment=True), tmp_path))
    path = tmp_path / 'pipeline.json'
    path.write_text(json.dumps(config), encoding='utf-8')
    monkeypatch.setattr('sys.argv', ['live', '--config', str(path)])
    assert parse_args().input_config == path


def test_candidate_remains_unadopted() -> None:
    from src import production_config as config
    assert config.FAST_FIRST_JUDGMENT_DEFAULT is False
    assert config.FAST_FIRST_JUDGMENT_CANDIDATES[0].adopted == '2026-10-11'
    assert config.FAST_FIRST_JUDGMENT_CANDIDATES[0] not in config.LIVE_BRIDGE_ADOPTED


def test_verification_recollection_never_interrupts_publication(session: DeviceSession) -> None:
    session.warmup.verification_only = True
    session.hold('no_puyo_screen')
    session.update(0, 0., None)
    probe = session._recheck
    previous = session.warmup
    resets = session.pipe.reset.call_count
    samples(probe, probe.target, shift=10)
    session.update(1, 1., None)
    assert probe.changed_profile and not probe.ready
    assert session.warmup is previous and session.publishing_ready
    samples(probe, probe.full_target, shift=10)
    session.update(2, 2., None)
    assert session.warmup is probe and probe.ready and session.publishing_ready
    assert session.pipe.reset.call_count == resets
    assert session._recheck is None and not session._needs_recheck
