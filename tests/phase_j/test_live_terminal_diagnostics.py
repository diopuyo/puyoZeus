"""追加計装の保存と、判断・状態通知の同一性を検証する。"""
from __future__ import annotations

from contextlib import ExitStack
from pathlib import Path
import pickle
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from src.match_end_detector import MatchEndDetector, MatchEndDetectionResult
from src.phase_j.live_device import DeviceConfig, DirectShowSource
from src.phase_j.live_input_guard import FrameContinuity
from src.phase_j.live_observation_diagnostics import archive_terminal_observation
from src.phase_j.live_process import recognition_logs
from tests.phase_j.test_live_fast_first import session


def test_match_end_records_existing_detection_without_extra_scan() -> None:
    detector = MatchEndDetector({})
    result = MatchEndDetectionResult(True, 'match_end_yatta', .83)
    detector.detect = Mock(return_value=result)
    assert detector.update(None, 3.)
    detector.detect.assert_called_once_with(None)
    assert detector.last_detection is result and detector.last_detection_t == 3.
    assert detector.is_locked(4.) and not detector.is_locked(100.)
    detector.reset()
    assert detector.last_detection is None and not detector.is_locked(4.)


def test_spool_exists_without_full_recognition_audit(tmp_path: Path) -> None:
    death = SimpleNamespace(scores=[.1, .7], streak=[0, 2], frames=10, fast=True,
                            full_matches=[0, 1], local_hits=[0, 1])
    detector = SimpleNamespace(last_detection=MatchEndDetectionResult(True, 'yatta', .8),
                               last_detection_t=1.)
    owner = SimpleNamespace(_live_snapshot_inputs=SimpleNamespace(terminal=death),
        _match_end_detector=detector, _live_match_off_diagnostics=dict(
            score_zero_both=False, post_match_lockdown_active=True, effective_hard_off=True))
    notice = SimpleNamespace(frame=3, t_sec=1.2, captured_at=10., recognized_at=10.1)
    with ExitStack() as stack:
        recognition_logs(stack, SimpleNamespace(), str(tmp_path), owner)
        archive_terminal_observation(owner, notice, False)
        death.scores[1], death.streak[1] = .2, 0
        owner._live_match_off_diagnostics['effective_hard_off'] = False
    with (tmp_path/'terminal_observations.pickle').open('rb') as stream:
        saved = pickle.load(stream)
    assert saved['observed_death']['scores'] == [.1, .7]
    assert saved['observed_death']['streak'] == [0, 2]
    assert saved['match_end']['score'] == .8
    assert saved['match_end']['observed_t_sec'] == 1.
    assert saved['match_off']['effective_hard_off'] and not saved['ready']


def test_continuity_reasons_do_not_change_return_values() -> None:
    guard = FrameContinuity(1.)
    image = np.zeros((16, 16, 3), np.uint8)
    assert guard.ready(image, (16, 16), 0.)
    assert not guard.ready(image, (16, 16), 1.)
    assert guard.diagnostics == dict(wrong_size=False, repeated=True, recovery_wait=False)
    assert not guard.ready(image+1, (16, 16), 1.1)
    assert guard.diagnostics['recovery_wait']
    assert guard.ready(image+2, (16, 16), 2.2)
    assert not guard.ready(image+3, (32, 16), 2.3)
    assert guard.diagnostics['wrong_size']


@pytest.mark.parametrize('duplicate', [False, True])
def test_device_exposes_duplicate_or_verifier_failure(duplicate: bool) -> None:
    clock, statuses = [0.], []
    source = DirectShowSource(DeviceConfig('OBS', 0, verify_fail_streak=3), 5., lambda: None,
        verifier=lambda image: duplicate or clock[0] < 1., clock=lambda: clock[0],
        sleep=lambda seconds: clock.__setitem__(0, clock[0]+seconds),
        on_status=lambda phase: statuses.append((phase, dict(source.input_diagnostics))))
    capture = SimpleNamespace(set=lambda *args: None, release=lambda: None)
    source._capture = lambda: capture
    source._read = lambda _: (clock[0], np.full((16, 16, 3),
        0 if duplicate else int(clock[0]*100) % 256, np.uint8), (16, 16))
    list(source)
    assert statuses
    first = statuses[0][1]
    assert first['continuity']['repeated'] is duplicate
    assert first['transport_ready'] is not duplicate
    assert first['verifier_fail_streak'] == (0 if duplicate else 3)


def test_diagnostics_do_not_add_status_transitions(session: object) -> None:
    while not session.queue.empty():
        session.queue.get()
    diagnostics = dict(continuity=dict(repeated=True), verifier_fail_streak=0)
    session.remember_input_diagnostics(diagnostics)
    session.hold('verifying')
    values = []
    while not session.queue.empty():
        values.append(session.queue.get()[1])
    assert values and all(v['input_diagnostics'] == diagnostics for v in values)
    session.remember_input_diagnostics(dict(verifier_fail_streak=7))
    session.hold('verifying')
    assert session.queue.empty()
