"""左右の組・因果窓・得点リセット待ちと配信配線を資産なしで試験する。"""
from __future__ import annotations

import argparse
import json
from tests.launcher_legacy import legacy_user_config
import os
import pickle
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from src.match_end_detector import MatchEndDetector
from src.match_end_pair import MatchEndPairDetector, PAIR_ROIS
from src.phase_j import launcher, live_speed
from src.phase_j.live_observation_diagnostics import recognition_diagnostics
from src.recognition_pipeline import RecognitionPipeline
from tests.test_live_r37_flags import _load_pipeline
from tests.test_lockdown_score_zero_then_move import _make_pipe, _run, FINAL, ZERO, FIRST
from tests.test_recognition_pipeline import _FakeScoreTrackerSeq
from src.score_ocr import ScoreDelta

FRAME = np.zeros((1080, 1920, 3), dtype=np.uint8)


class _NullableScoreTracker(_FakeScoreTrackerSeq):
    """OCR欠測では確定得点を保持し、新規読み取りはNoneを返す。"""

    def update(self, frame: np.ndarray) -> ScoreDelta:
        cur = self._scores[min(self._idx, len(self._scores) - 1)]
        if cur is not None:
            return super().update(frame)
        self._idx += 1
        return ScoreDelta(side='1P', prev_score=self._last_score, cur_score=None, delta=0)


def _mock_scores(detector: MatchEndPairDetector, monkeypatch: pytest.MonkeyPatch,
                 **scores: float) -> None:
    monkeypatch.setattr(detector, '_score_roi', lambda frame, key: scores.get(key, 0.1))


def _sweep(detector: MatchEndPairDetector, t_sec: float) -> bool:
    """同一時刻の4認識コマで一巡させる。"""
    for _ in PAIR_ROIS:
        locked = detector.update(FRAME, t_sec)
    return locked


@pytest.mark.parametrize('winner', ['1P', '2P'])
def test_real_ncc_both_sides(winner: str) -> None:
    """人工画像を実NCCで照合し、左右の探索領域と勝者を検証する。"""
    rng = np.random.default_rng(42)
    templates = {f'match_end_{kind}': rng.integers(0, 255, (20, 35), dtype=np.uint8)
                 for kind in ('yatta', 'batan')}
    frame = FRAME.copy()
    for kind, side in [('yatta', winner), ('batan', '2P' if winner == '1P' else '1P')]:
        x, y, _, _ = PAIR_ROIS[f'{kind}_{side.lower()}']
        frame[y:y + 20, x:x + 35] = cv2.cvtColor(templates[f'match_end_{kind}'], cv2.COLOR_GRAY2BGR)
    detector = MatchEndPairDetector(templates)
    for index in range(len(PAIR_ROIS)):
        locked = detector.update(frame, 10 + index / 30)
    assert locked
    assert detector.last_detection.winner == winner
    assert detector.is_locked(100), '固定5秒で解除しない'


@pytest.mark.parametrize('key', list(PAIR_ROIS))
def test_single_never_locks(key: str, monkeypatch: pytest.MonkeyPatch) -> None:
    detector = MatchEndPairDetector({})
    _mock_scores(detector, monkeypatch, **{key: 0.9})
    assert not _sweep(detector, 1)


@pytest.mark.parametrize('delay, expected', [(0, True), (0.5, True), (0.501, False), (-0.1, False)])
def test_causal_window(delay: float, expected: bool, monkeypatch: pytest.MonkeyPatch) -> None:
    detector = MatchEndPairDetector({})
    _mock_scores(detector, monkeypatch, yatta_2p=0.8)
    assert not _sweep(detector, 1)
    _mock_scores(detector, monkeypatch, batan_1p=0.8)
    # 次の照合をばたん1Pに合わせ、やった2Pの最新観測を上書きしない。
    detector._next_roi = list(PAIR_ROIS).index('batan_1p')
    assert detector.update(FRAME, 1 + delay) is expected


@pytest.mark.parametrize('scores', [dict(yatta_1p=0.9, batan_1p=0.9),
                                  dict(yatta_1p=0.49, batan_2p=0.9),
                                  dict.fromkeys(PAIR_ROIS, 0.9)])
def test_wrong_side_weak_or_ambiguous(scores: dict[str, float], monkeypatch: pytest.MonkeyPatch) -> None:
    detector = MatchEndPairDetector({})
    _mock_scores(detector, monkeypatch, **scores)
    assert not _sweep(detector, 1)


def _pair_pipe(pairs: list[tuple[int | None, int | None]], r37: bool,
               monkeypatch: pytest.MonkeyPatch, **kwargs: bool) -> tuple[RecognitionPipeline, MatchEndPairDetector]:
    pipe, _ = _make_pipe(pairs, r37, enable_match_end_pair=True, **kwargs)
    pipe._score_tracker_1p = _NullableScoreTracker([p[0] for p in pairs])
    pipe._score_tracker_2p = _NullableScoreTracker([p[1] for p in pairs])
    detector = MatchEndPairDetector({})
    detector.lock_started_t = 5.0
    detector._last_detected_t = 5.0
    _mock_scores(detector, monkeypatch)
    pipe._match_end_detector = detector
    return pipe, detector


@pytest.mark.parametrize('r37', [False, True])
def test_zero_twice_releases_and_r37_starts(r37: bool, monkeypatch: pytest.MonkeyPatch) -> None:
    seq = [FINAL] * 2 + [ZERO] * 2 + [FIRST] * 4
    pipe, detector = _pair_pipe(seq, r37, monkeypatch)
    _run(pipe, 3)
    assert detector.is_locked(50)
    _run(pipe, 1, start=3)
    assert not detector.is_locked(50)
    assert pipe._post_match_lockdown_active is r37
    results = _run(pipe, 4, start=4)
    if r37:
        assert pipe._lockdown_score_start_confirmed
        assert results[-1].is_match_active


@pytest.mark.parametrize('gap', [(None, None), (None, 0), (0, None), (1, 0), (0, 1)])
def test_zero_read_streak_breaks(gap: tuple[int | None, int | None], monkeypatch: pytest.MonkeyPatch) -> None:
    pipe, detector = _pair_pipe([FINAL, ZERO, gap, ZERO, ZERO], False, monkeypatch)
    _run(pipe, 4)
    assert detector.is_locked(50)
    _run(pipe, 1, start=4)
    assert not detector.is_locked(50)


def test_missing_zero_uses_existing_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    pipe, detector = _pair_pipe([FINAL] * 3, False, monkeypatch)
    pipe.update(0, 5, FRAME)
    assert pipe.update(1, 11, FRAME).match_end_locked
    assert not pipe.update(2, 50, FRAME).match_end_locked
    assert not detector.is_locked(50)


def test_pixel_auxiliary_releases_pair(monkeypatch: pytest.MonkeyPatch) -> None:
    pipe, detector = _pair_pipe([FINAL] * 2, False, monkeypatch)
    pipe.update(0, 5, FRAME)
    pipe._score_zero_detector.both_zero = True
    # 実際の補助解除条件を直接与え、解除結果が組にも伝わることを検査。
    monkeypatch.setattr(pipe, '_board_shows_real_gameplay', lambda frame: True)
    pipe._post_match_lockdown_raw_active_since = 5
    pipe._update_post_match_lockdown_latch(FRAME, 10, True, True, None, None, False)
    assert not pipe._release_pair_fallback(10, True)
    assert not detector.is_locked(10)


def test_diagnostics_only_when_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    detector = MatchEndPairDetector({})
    _mock_scores(detector, monkeypatch, yatta_1p=0.7, batan_2p=0.6)
    _sweep(detector, 2)
    owner = SimpleNamespace(_match_end_detector=detector, _enable_match_end_pair=True)
    result = recognition_diagnostics(owner)['match_end']
    assert result['winner'] == '1P' and result['detected']
    assert set(result['ncc']) == set(PAIR_ROIS)
    assert result['scanned_roi'] == 'batan_2p'
    assert result['ncc_observed_t_sec'] == dict.fromkeys(PAIR_ROIS, 2)
    owner._enable_match_end_pair = False
    assert set(recognition_diagnostics(owner)['match_end']) == {
        'score', 'detected', 'template_name', 'observed_t_sec'}


@pytest.mark.parametrize('enabled', [False, True])
def test_json_to_pipeline_and_trial_log(enabled: bool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from scripts.run_live_pipeline_20260928 import parse_args
    monkeypatch.setattr(os, 'environ', {})
    monkeypatch.setenv('PUYO_LIVE_MATCH_END_PAIR', str(int(not enabled)))
    path = tmp_path / 'puyo_live.json'
    config = legacy_user_config(dict(device_name='test', live_match_end_pair=enabled))
    path.write_text(json.dumps(config), encoding='utf-8')
    user = launcher.load_user_config(path)
    config_path = tmp_path / 'config.json'
    config_path.write_text(json.dumps(launcher.build_pipeline_config(user)), encoding='utf-8')
    monkeypatch.setattr(sys, 'argv', ['live', *launcher.build_pipeline_argv(user, config_path)])
    parse_args()
    config = live_speed.recognition_config({}, True)
    pipe = _load_pipeline(pickle.loads(pickle.dumps(config)), monkeypatch)
    assert pipe._enable_match_end_pair is enabled
    record = live_speed.trial_flags_record(argparse.Namespace(live_match_end_pair=enabled))
    flags = list(record['flags'])
    start = flags.index('live_match_end_pair')
    assert flags[start:start+7] == ['live_match_end_pair', 'live_verify_reason_log',
                                    'b3_warning_async', 'live_native_hsv_main', 'live_verify_async',
                                    'live_match_end_feed_on_hold', 'match_end_yatta_text_template']   # 後ろに別の鍵が続いてよい
    assert record['env']['PUYO_LIVE_MATCH_END_PAIR'] == str(int(enabled))
    assert record['arm'] == 'OFF'


def test_off_keeps_old_single_detection_and_timer(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.match_end_detector import MatchEndDetectionResult
    detector = MatchEndDetector({})
    monkeypatch.setattr(detector, 'detect', lambda frame: MatchEndDetectionResult(True, 'yatta', 0.9))
    assert detector.update(FRAME, 1)
    assert detector.is_locked(6)
    assert not detector.is_locked(6.001)


def test_factory_pair_bypasses_image_free(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr('src.image_free_ui.enabled', lambda: True)
    assert isinstance(MatchEndDetector.load_default(tmp_path, pair_mode=True), MatchEndPairDetector)


def test_reset_clears_window(monkeypatch: pytest.MonkeyPatch) -> None:
    detector = MatchEndPairDetector({})
    _mock_scores(detector, monkeypatch, yatta_1p=0.8)
    _sweep(detector, 1)
    detector.reset()
    _mock_scores(detector, monkeypatch, batan_2p=0.8)
    assert not _sweep(detector, 1.1)


def test_r37_cannot_start_after_only_one_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    pipe, detector = _pair_pipe([FINAL, ZERO] + [FIRST] * 5, True, monkeypatch)
    results = _run(pipe, 7)
    assert detector.is_locked(10)
    assert not pipe._lockdown_score_start_confirmed
    assert not any(result.is_match_active for result in results)


@pytest.mark.parametrize('r37', [False, True])
def test_full_clear_does_not_preempt_two_reads(r37: bool, monkeypatch: pytest.MonkeyPatch) -> None:
    pipe, detector = _pair_pipe([FINAL] * 2 + [ZERO] * 4 + [FIRST] * 5, r37, monkeypatch)
    pipe._enable_match_start_full_clear = True
    _run(pipe, 3)
    assert detector.is_locked(10)
    _run(pipe, 1, start=3)
    assert not detector.is_locked(10)
    results = _run(pipe, 7, start=4)
    if r37:
        assert results[-1].is_match_active


def test_pair_scans_each_frame_despite_throttle(monkeypatch: pytest.MonkeyPatch) -> None:
    pipe, detector = _pair_pipe([FINAL] * 4, False, monkeypatch)
    pipe._enable_large_roi_throttle = True
    results = _run(pipe, 4)
    assert detector.last_detection_t == results[-1].time_sec


def test_fallback_without_latch_still_has_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    pipe, detector = _pair_pipe([FINAL] * 2, False, monkeypatch)
    pipe._enable_post_match_lockdown_latch = False
    assert pipe.update(0, 11, FRAME).match_end_locked
    assert not pipe.update(1, 50, FRAME).match_end_locked
    assert not detector.is_locked(50)


def test_moving_auxiliary_still_works_after_announcement(monkeypatch: pytest.MonkeyPatch) -> None:
    pipe, detector = _pair_pipe([FINAL], False, monkeypatch, enable_lockdown_score_moving_release=True)
    pipe.update(0, 5, FRAME)
    monkeypatch.setattr(pipe, '_board_shows_real_gameplay', lambda frame: True)
    pipe._update_post_match_lockdown_latch(FRAME, 6, True, False, *FINAL, True)
    assert pipe._post_match_lockdown_active
    pipe._update_post_match_lockdown_latch(FRAME, 11, True, False, *FINAL, True)
    assert not pipe._release_pair_fallback(11, True)
    assert not detector.is_locked(11)


def test_one_ncc_per_frame_and_four_frame_cycle(monkeypatch: pytest.MonkeyPatch) -> None:
    """実NCC呼出し数と、診断に残る各領域の時刻を確かめる。"""
    template = np.random.default_rng(7).integers(0, 255, (20, 35), dtype=np.uint8)
    detector = MatchEndPairDetector({'yatta': template, 'batan': template})
    original = cv2.matchTemplate
    calls: list[tuple[int, ...]] = []

    def match(roi: np.ndarray, target: np.ndarray, method: int) -> np.ndarray:
        calls.append(roi.shape)
        return original(roi, target, method)

    monkeypatch.setattr(cv2, 'matchTemplate', match)
    observed = dict.fromkeys(PAIR_ROIS, None)
    for index, key in enumerate(list(PAIR_ROIS) * 2):
        detector.update(FRAME, index / 30)
        observed[key] = index / 30
        assert len(calls) == index + 1
        assert detector.last_detection.scanned_roi == key
        assert detector.last_detection.ncc_observed_t_sec == observed


@pytest.mark.parametrize('winner', ['1P', '2P'])
@pytest.mark.parametrize('offset', range(len(PAIR_ROIS)))
def test_end_detected_within_one_cycle(winner: str, offset: int,
                                      monkeypatch: pytest.MonkeyPatch) -> None:
    detector = MatchEndPairDetector({})
    _mock_scores(detector, monkeypatch)
    for index in range(len(PAIR_ROIS) + offset):
        assert not detector.update(FRAME, index / 30)
    loser = '2P' if winner == '1P' else '1P'
    _mock_scores(detector, monkeypatch,
                 **{f'yatta_{winner.lower()}': 0.8, f'batan_{loser.lower()}': 0.8})
    for step in range(1, len(PAIR_ROIS) + 1):
        if detector.update(FRAME, (index + step) / 30):
            break
    assert detector.last_detection.winner == winner


def test_latest_negative_replaces_positive(monkeypatch: pytest.MonkeyPatch) -> None:
    detector = MatchEndPairDetector({})
    _mock_scores(detector, monkeypatch, yatta_1p=0.8)
    assert not _sweep(detector, 1)
    _mock_scores(detector, monkeypatch, batan_2p=0.8)
    assert not _sweep(detector, 1.1)
    assert detector.last_detection.ncc['yatta_1p'] == 0.1


def test_expired_ncc_keeps_diagnostic_timestamp(monkeypatch: pytest.MonkeyPatch) -> None:
    detector = MatchEndPairDetector({})
    _mock_scores(detector, monkeypatch, yatta_1p=0.8)
    assert not _sweep(detector, 1)
    detector._next_roi = list(PAIR_ROIS).index('batan_2p')
    _mock_scores(detector, monkeypatch, batan_2p=0.8)
    assert not detector.update(FRAME, 1.501)
    assert detector.last_detection.ncc['yatta_1p'] == 0.8
    assert detector.last_detection.ncc_observed_t_sec['yatta_1p'] == 1


def test_off_pipeline_preserves_scan_interval(monkeypatch: pytest.MonkeyPatch) -> None:
    pipe, _ = _make_pipe([FINAL] * 20, False, enable_match_end_pair=False)
    detector = MatchEndDetector({})
    calls: list[float] = []
    monkeypatch.setattr(detector, 'update', lambda frame, t: calls.append(t) or False)
    pipe._match_end_detector = detector
    pipe._enable_large_roi_throttle = True
    results = _run(pipe, 20)
    assert calls == [result.time_sec for index, result in enumerate(results)
                     if pipe._should_run_large_roi_scan(index)]
