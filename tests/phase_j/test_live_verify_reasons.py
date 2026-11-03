"""実機を使わず、理由の記録と判定・短絡順序の不変性を検査する。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from src.phase_j import live_device as device
from src.phase_j import live_verify_reasons as reasons
from src.phase_j.live_border_grace import verify_screen
from src.phase_j.live_screen_worker import ScreenWorker

IMAGE = np.zeros((1080, 1920, 3), np.uint8)
SIZE = (1920, 1080)
WIDTH, HEIGHT = 384, 720


def verifier_for(case: str, relaxed: bool = False) -> device.PuyoScreenVerifier:
    verifier = object.__new__(device.PuyoScreenVerifier)
    verifier.relaxed = relaxed
    verifier.regions = [SimpleNamespace(x=100, y=100, width=WIDTH, height=HEIGHT)]*2
    offset = 81 if case == 'grid' else 80 if case == 'boundary' else 0
    grid = SimpleNamespace(top_left=(20+offset, 20), bottom_right=(404, 740))
    verifier.grid = SimpleNamespace(detect=lambda image: None if case.startswith('border') else grid)
    verifier.score = SimpleNamespace(
        read_side=lambda image, side: (None if case in ('ocr', 'formula') or
                                      (case == 'p2_ocr' and side == '2P') else 123, 1.),
        read_formula_side=lambda image, side: SimpleNamespace(valid=case == 'formula'))
    return verifier


def setup_case(monkeypatch: pytest.MonkeyPatch, case: str) -> None:
    from src.phase_j import distribution_input
    monkeypatch.setattr(distribution_input, 'has_feedback', lambda image: case == 'feedback')
    lines = [(20, 20, 404, 20), (20, 740, 404, 740),
             (20, 20, 20, 740), (404, 20, 404, 740)]
    if case == 'border_coverage':
        lines[0] = (20, 20, 120, 20)
    if case == 'border_position':
        lines[0] = (20, 45, 404, 45)
    monkeypatch.setattr(device.cv2, 'HoughLinesP', lambda *args, **kwargs:
                        None if case == 'border_none' else np.array(lines).reshape(-1, 1, 4))


@pytest.mark.parametrize('case', ['pass', 'ocr', 'p2_ocr', 'grid', 'boundary', 'feedback',
                                 'formula', 'border_pass', 'border_coverage', 'border_position', 'border_none'])
@pytest.mark.parametrize('relaxed', [False, True])
def test_on_off_same_result_and_named_reason(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                           case: str, relaxed: bool) -> None:
    setup_case(monkeypatch, case)
    monkeypatch.setenv(reasons.OUTPUT_ENV, str(tmp_path))
    monkeypatch.setenv(reasons.FLAG_ENV, '0')
    expected = verify_screen(verifier_for(case, relaxed), IMAGE, 1., SIZE, 0)
    path = tmp_path/reasons.LOG_FILE
    assert not path.exists()
    monkeypatch.setenv(reasons.FLAG_ENV, '1')
    actual = verify_screen(verifier_for(case, relaxed), IMAGE, 1., SIZE, 0)
    row = json.loads(path.read_text(encoding='utf-8'))
    assert actual == expected == row['passed']
    assert row['fail_streak'] == int(not actual)
    assert row['relaxed'] == relaxed and row['captured_at'] == 1.
    failure = {'ocr': '1P.score', 'p2_ocr': '2P.score', 'grid': '1P.grid_position',
               'feedback': 'feedback', 'formula': '1P.score',
               'border_coverage': '1P.border', 'border_position': '1P.border',
               'border_none': '1P.border'}.get(case)
    assert row['failed_conditions'][:1] == ([] if actual else [failure])
    if case == 'ocr':
        assert row['checks']['1P.score']['value'] is None
        assert row['checks']['2P.score'] is None
    if case == 'grid':
        assert row['checks']['1P.grid_position']['error_px'] == 81
    if case == 'border_coverage':
        assert row['checks']['1P.border']['edges']['top']['coverage'] == 100/WIDTH
        if not relaxed:
            assert '1P.border.top.coverage' in row['failed_conditions']
    if case == 'border_position':
        assert 25 in row['checks']['1P.border']['edges']['top']['position_offsets_px']
        if not relaxed:
            assert '1P.border.top.position' in row['failed_conditions']


def test_worker_logs_each_verification_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    setup_case(monkeypatch, 'ocr')
    monkeypatch.setenv(reasons.FLAG_ENV, '1')
    monkeypatch.setenv(reasons.OUTPUT_ENV, str(tmp_path))
    worker = ScreenWorker(verifier_for('ocr'))
    try:
        for stamp in (1., 2.):
            assert worker.submit(IMAGE, stamp, SIZE)
            worker.future.result()
            assert worker.poll().passed is False
        worker.invalidate()
        assert worker.submit(IMAGE, 3., SIZE)
        worker.future.result()
        worker.poll()
    finally:
        worker.close()
    rows = [json.loads(line) for line in (tmp_path/reasons.LOG_FILE).read_text().splitlines()]
    assert [row['fail_streak'] for row in rows] == [1, 2, 1]
    assert [row['captured_at'] for row in rows] == [1., 2., 3.]


def test_streak_hold_sequence_unchanged(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from tests.phase_j.test_live_device import _run_streak
    monkeypatch.setenv(reasons.OUTPUT_ENV, str(tmp_path))
    script = [True, False, False, False, True, False]
    monkeypatch.setenv(reasons.FLAG_ENV, '0')
    expected = _run_streak(script, 3)
    monkeypatch.setenv(reasons.FLAG_ENV, '1')
    assert _run_streak(script, 3) == expected
    rows = [json.loads(line) for line in (tmp_path/reasons.LOG_FILE).read_text().splitlines()]
    assert [row['fail_streak'] for row in rows[:6]] == [0, 1, 2, 3, 0, 1]


def test_config_launcher_and_trial_flag(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from src.phase_j import live_speed
    from src.phase_j.launcher import UserConfig
    assert not live_speed.SPEED_DEFAULTS['live_verify_reason_log']
    assert not UserConfig.__dataclass_fields__['live_verify_reason_log'].default
    keys = live_speed.TRIAL_FLAG_KEYS
    start = keys.index('live_verify_reason_log')
    assert keys[start:start+4] == ('live_verify_reason_log', 'b3_warning_async',
                                   'live_native_hsv_main', 'live_verify_async')   # 後ろに別の鍵が続いてよい
    parser = argparse.ArgumentParser()
    live_speed.add_arguments(parser)
    options = parser.parse_args(['--live-verify-reason-log'])
    options.output, options.source, options.cpu_threads = tmp_path, 'dshow', 1
    monkeypatch.setattr('os.environ', dict())
    live_speed.configure(options, parser)
    assert reasons.enabled()
    assert Path(__import__('os').environ[reasons.OUTPUT_ENV]) == tmp_path
    row = json.loads((tmp_path/live_speed.TRIAL_FLAGS_FILE).read_text())
    assert row['flags']['live_verify_reason_log'] is True
    assert row['env'][reasons.FLAG_ENV] == '1'


def test_save_failure_does_not_change_result(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    setup_case(monkeypatch, 'pass')
    monkeypatch.setenv(reasons.FLAG_ENV, '1')
    monkeypatch.setenv(reasons.OUTPUT_ENV, str(tmp_path/'missing'))
    assert verify_screen(verifier_for('pass'), IMAGE, 1., SIZE, 0)


@pytest.mark.parametrize('on', [False, True])
def test_short_circuit_does_not_repeat_ocr(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, on: bool) -> None:
    setup_case(monkeypatch, 'ocr')
    monkeypatch.setenv(reasons.FLAG_ENV, str(int(on)))
    monkeypatch.setenv(reasons.OUTPUT_ENV, str(tmp_path))
    verifier = verifier_for('ocr')
    calls = []
    def read(image: np.ndarray, side: str) -> tuple[None, float]:
        calls.append(side)
        return None, 0.
    verifier.score.read_side = read
    assert not verify_screen(verifier, IMAGE, 1., SIZE, 0)
    assert calls == ['1P']


def test_feedback_correlation_is_observed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from src.phase_j.distribution_input import marker_image, MARKER_TOLERANCE
    monkeypatch.setenv(reasons.FLAG_ENV, '1')
    monkeypatch.setenv(reasons.OUTPUT_ENV, str(tmp_path))
    image = IMAGE.copy()
    marker = marker_image()
    image[100:100+marker.shape[0], 100:100+marker.shape[1]] = marker
    assert not verify_screen(verifier_for('pass'), image, 1., SIZE, 0)
    row = json.loads((tmp_path/reasons.LOG_FILE).read_text())
    assert row['checks']['feedback']['max_correlation'] >= MARKER_TOLERANCE
    assert row['failed_conditions'] == ['feedback']


@pytest.mark.parametrize('arm', ['OFF', 'X', 'ALL'])
def test_diagnostic_flag_does_not_change_trial_arm(arm: str) -> None:
    from src.phase_j.live_speed import classify_arm, X_TRIAL_KEYS, TRIAL_FLAG_KEYS
    keys = {'OFF': (), 'X': X_TRIAL_KEYS, 'ALL': TRIAL_FLAG_KEYS[:-1]}[arm]
    flags = dict.fromkeys(keys, True)
    assert classify_arm(flags) == arm
    assert classify_arm(dict(flags, live_verify_reason_log=True)) == arm


def test_grace_result_and_raw_failure_are_separate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from src.phase_j.live_border_grace import BorderGraceVerifier
    setup_case(monkeypatch, 'border_pass')
    monkeypatch.setenv(reasons.FLAG_ENV, '1')
    monkeypatch.setenv(reasons.OUTPUT_ENV, str(tmp_path))
    verifier = BorderGraceVerifier(verifier_for('border_pass'))
    assert verify_screen(verifier, IMAGE, 1., SIZE, 0)
    monkeypatch.setattr(device.cv2, 'HoughLinesP', lambda *args, **kwargs: None)
    assert verify_screen(verifier, IMAGE, 2., SIZE, 0)
    rows = [json.loads(line) for line in (tmp_path/reasons.LOG_FILE).read_text().splitlines()]
    assert rows[1]['passed'] and rows[1]['fail_streak'] == 0
    assert rows[1]['grace']['grace'] and not rows[1]['grace']['strict']
    assert '1P.border.top.coverage' in rows[1]['failed_conditions']
