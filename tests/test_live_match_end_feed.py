"""保留専用の供給ゲート・保持・文字限定・既定OFFを検証する。"""
from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import cv2
import numpy as np
import pytest

from src.async_match_end_pair import AsyncMatchEndPairDetector
from src.match_end_pair import MatchEndPairDetector
from src.phase_j import live_match_end_feed as feed
from src.phase_j import live_speed
from src.phase_j.live_device import DeviceConfig, DirectShowSource, PuyoScreenVerifier
from src.phase_j.live_device_session import DeviceSession
from tests.test_async_match_end_pair import FRAME, _completed
from tests.test_lockdown_score_zero_then_move import _make_pipe, _run, FINAL, ZERO


@pytest.mark.parametrize('borders,scores,expected', [
    ((False, True), (True, True), True),
    ((True, False), (True, True), True),
    ((True, True), (False, False), True),
    ((False, True), (False, True), False),
    ((False, False), (True, True), True),
    ((False, False), (False, False), False),
])
def test_same_frame_screen_evidence(borders: tuple, scores: tuple, expected: bool,
                                    monkeypatch: pytest.MonkeyPatch) -> None:
    """片側の崩れ・得点欠測を許すが、枠も得点もない別画面を拒否する。"""
    from src.image_reader import DEFAULT_P1_REGION, DEFAULT_P2_REGION
    from src.phase_j import distribution_input, live_device
    verifier = object.__new__(PuyoScreenVerifier)
    verifier.regions = (DEFAULT_P1_REGION, DEFAULT_P2_REGION)
    verifier.grid = SimpleNamespace(detect=lambda image: None)
    verifier.score_ok = lambda image, side: scores[0 if side == '1P' else 1]
    border_iter = iter(borders)
    monkeypatch.setattr(live_device, 'frame_border_matches', lambda *args, **kw: next(border_iter))
    monkeypatch.setattr(distribution_input, 'has_feedback', lambda image: False)
    assert feed.eligible(FRAME, verifier) is expected


def test_feedback_rejected_before_ocr(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.phase_j import distribution_input
    monkeypatch.setattr(distribution_input, 'has_feedback', lambda image: True)
    assert not feed.eligible(FRAME, object.__new__(PuyoScreenVerifier))


def test_displaced_grid_is_not_a_border_hold(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.image_reader import DEFAULT_P1_REGION, DEFAULT_P2_REGION
    from src.phase_j import distribution_input
    monkeypatch.setattr(distribution_input, 'has_feedback', lambda image: False)
    verifier = object.__new__(PuyoScreenVerifier)
    verifier.regions = (DEFAULT_P1_REGION, DEFAULT_P2_REGION)
    verifier.grid = SimpleNamespace(detect=lambda image: SimpleNamespace(
        top_left=(500, 500), bottom_right=(900, 900)))
    assert not feed.eligible(FRAME, verifier)


@pytest.mark.parametrize('ready,allowed', [(False, True), (True, False), (True, True)])
def test_transport_and_current_frame_gate(ready: bool, allowed: bool,
                                         monkeypatch: pytest.MonkeyPatch) -> None:
    source = DirectShowSource(DeviceConfig('test', 0), 1, lambda: None)
    source._match_end_hold_verifier = object()
    callback, gate = Mock(), Mock(return_value=allowed)
    source.on_match_end_hold = callback
    monkeypatch.setattr(feed, 'eligible', gate)
    source._feed_match_end_hold(FRAME, 12, ready)
    assert callback.call_args.args[0] is (FRAME if ready and allowed else None)
    assert callback.call_args.args[1] == 12
    assert gate.call_count == int(ready)


def test_held_end_is_latched_and_zero_twice_releases(monkeypatch: pytest.MonkeyPatch) -> None:
    pipe, _ = _make_pipe([FINAL, ZERO, ZERO], False, enable_match_end_pair=True)
    detector = AsyncMatchEndPairDetector({})
    pipe._match_end_detector = detector
    session = SimpleNamespace(pipe=pipe)
    monkeypatch.setattr(detector, '_score_crop',
                        lambda crop, key: 0.9 if key in ('yatta_1p', 'batan_2p') else 0.1)
    try:
        feed.feed(session, FRAME, 1)
        _completed(detector, 1)
        feed.feed(session, FRAME, 2)
        assert detector.is_locked(2) and pipe._post_match_lockdown_active
        assert pipe._post_match_lockdown_started_time == 1
        _run(pipe, 3)
        assert not detector.is_locked(20)
        assert detector._completed is None and detector._pending is None
    finally:
        detector.close()


def test_rejected_frame_never_reaches_detector() -> None:
    session = SimpleNamespace(pipe=SimpleNamespace(_match_end_detector=Mock()))
    feed.feed(session, None, 1)
    session.pipe._match_end_detector.update.assert_not_called()
    assert not session._match_end_hold_eligible


def test_eligible_hold_does_not_reset_pipeline() -> None:
    session = object.__new__(DeviceSession)
    session._match_end_hold_eligible = True
    session.warmup, session.pipe, session.emit = None, Mock(), Mock()
    session.publishing_ready = True
    session.hold('no_puyo_screen')
    session.pipe.reset.assert_not_called()
    assert not session.publishing_ready


@pytest.mark.parametrize('flag', ['0', '1'])
def test_text_template_is_exact_crop(flag: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('PUYO_MATCH_END_YATTA_TEXT_TEMPLATE', flag)
    rng = np.random.default_rng(42)
    yatta = rng.integers(0, 256, (370, 520), dtype=np.uint8)
    batan = rng.integers(0, 256, (300, 470), dtype=np.uint8)
    templates = dict(match_end_yatta=yatta, match_end_batan=batan)
    detector = MatchEndPairDetector(templates)
    expected = yatta[210:365, 70:500] if flag == '1' else yatta
    np.testing.assert_array_equal(detector._templates['match_end_yatta'], expected)
    assert detector._templates['match_end_batan'] is batan
    assert templates['match_end_yatta'] is yatta
    image = FRAME.copy()
    image[300:455, 200:630] = cv2.cvtColor(yatta[210:365, 70:500], cv2.COLOR_GRAY2BGR)
    image[300:600, 1200:1670] = cv2.cvtColor(batan, cv2.COLOR_GRAY2BGR)
    result = detector.detect(image)
    assert result.detected is (flag == '1')
    if flag == '1':
        assert result.winner == '1P'
    else:
        assert detector._templates is templates


def test_flags_default_off_and_environment_delivery(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(live_speed.os, 'environ', {})
    keys = ('live_match_end_feed_on_hold', 'match_end_yatta_text_template')
    # 後続の鍵 (live_capture_reconnect・live_board_vectorized など) が末尾へ追加されても、この 2 つは連続して並ぶ
    start = live_speed.TRIAL_FLAG_KEYS.index(keys[0])
    assert live_speed.TRIAL_FLAG_KEYS[start:start+2] == keys
    for key in keys:
        assert live_speed.SPEED_DEFAULTS[key] is False
        monkeypatch.delenv('PUYO_'+key.upper(), raising=False)
    assert not feed.enabled()
    assert DirectShowSource(DeviceConfig('test', 0), 1, lambda: None).on_match_end_hold is None
    options = argparse.Namespace(source='dshow', cpu_threads=1, **dict.fromkeys(keys, True))
    live_speed.configure(options, argparse.ArgumentParser())
    assert feed.enabled()
    record = live_speed.trial_flags_record(options)
    assert record['arm'] == 'OFF'
    assert all(record['flags'][key] and record['env']['PUYO_'+key.upper()] == '1' for key in keys)


@pytest.mark.parametrize('async_verify', ['0', '1'])
@pytest.mark.parametrize('allowed', [False, True])
def test_source_feeds_hold_without_recognizing(async_verify: str, allowed: bool,
                                              monkeypatch: pytest.MonkeyPatch) -> None:
    from tests.phase_j.test_live_device import FakeDevice
    monkeypatch.setenv('PUYO_LIVE_VERIFY_ASYNC', async_verify)
    monkeypatch.setenv('PUYO_LIVE_BORDER_GRACE', '0')
    clock = [0.0]
    calls = []

    def sleep(seconds: float) -> None:
        clock[0] += seconds

    source = DirectShowSource(DeviceConfig('test', 0), .15, lambda: None,
        lambda *args: FakeDevice(FRAME), lambda image: False, lambda: clock[0], sleep)
    source._match_end_hold_verifier = object()
    source.on_match_end_hold = lambda image, stamp: calls.append((image, stamp))
    monkeypatch.setattr(feed, 'eligible', lambda *args, **kwargs: allowed)
    assert list(source) == []
    assert calls and all(image is (FRAME if allowed else None) for image, _ in calls)


def test_release_drops_completed_positive_before_next_hold() -> None:
    detector = AsyncMatchEndPairDetector({})
    scores = dict(yatta_1p=.9, batan_2p=.9)
    detector._completed = (1, detector._result(scores, scores))
    detector.release()
    session = SimpleNamespace(pipe=SimpleNamespace(_match_end_detector=detector))
    try:
        feed.feed(session, FRAME, 2)
        assert not detector.is_locked(2)
    finally:
        detector.close()


@pytest.mark.parametrize('enabled', [False, True])
def test_json_launcher_forwards_both_flags(enabled: bool, tmp_path: Path) -> None:
    from src.phase_j import launcher
    data = dict(source='dshow', device_name='OBS', live_match_end_feed_on_hold=enabled,
                match_end_yatta_text_template=enabled)
    config = launcher.build_pipeline_config(launcher.parse_user_config(data, tmp_path))
    for key in ('live_match_end_feed_on_hold', 'match_end_yatta_text_template'):
        assert config.get(key, False) is enabled


@pytest.mark.parametrize('flag', ['0', '1'])
@pytest.mark.parametrize('pair_mode', [False, True])
def test_session_wiring_only_with_both_flags(flag: str, pair_mode: bool, tmp_path: Path,
                                            monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('PUYO_LIVE_MATCH_END_FEED_ON_HOLD', flag)
    pipe = SimpleNamespace(_reader=SimpleNamespace(_classifier=SimpleNamespace(
        _hsv=SimpleNamespace(_ranges={}))), _online_hsv=None, _enable_match_end_pair=pair_mode)
    session = DeviceSession(pipe, tmp_path/'input.json', 1, Mock())
    assert (session.source.on_match_end_hold is not None) is (flag == '1' and pair_mode)


@pytest.mark.parametrize('phase', ['verifying', 'overlay_feedback', 'no_puyo_screen'])
def test_rejected_hold_keeps_legacy_reset(phase: str) -> None:
    session = object.__new__(DeviceSession)
    session._match_end_hold_eligible = False
    session._held = False
    session.warmup = SimpleNamespace(ready=True, progress=1,
        device=SimpleNamespace(fast_first_judgment=False), status=lambda value: dict(phase=value))
    session.switch, session.emit = Mock(), Mock()
    session.hold(phase)
    session.switch.assert_called_once_with(session.warmup.device, advance_epoch=True)
