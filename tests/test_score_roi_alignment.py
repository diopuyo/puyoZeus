"""式枠較正の証拠条件・既存互換・配信配線を検証する。"""
from __future__ import annotations

from copy import deepcopy
import inspect
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from src.score_ocr import ScoreOcr, _crop_score_roi, SCORE_1P_REGION, SCORE_2P_REGION
from src.score_roi_alignment import (
    CANDIDATES, FORMULA_SAMPLES, ScoreRoiAlignment, candidate_scores, select_offset,
    FIXED_ROI_NCC_MIN, ZERO_OFFSET_INDEX, select_final_offset,
)


def fixture(offsets: tuple = ((0, 0), (0, 0))) -> tuple:
    """背景と区別できる合成数字と×を固定乱数で配置する。"""
    rng = np.random.default_rng(93)
    templates = {i: rng.integers(0, 256, (50, 40), dtype=np.uint8) for i in range(10)}
    mult = rng.integers(0, 256, (50, 40), dtype=np.uint8)
    ocr = ScoreOcr(templates, mult_template=mult)
    frame = np.zeros((1080, 1920), dtype=np.uint8)
    for region, (dx, dy) in zip((SCORE_1P_REGION, SCORE_2P_REGION), offsets):
        for cell, digit in ((2, 4), (3, 0), (4, -1), (7, 1)):
            y, x = region[0]+8+dy, region[2]+40*cell+dx
            frame[y:y+50, x:x+40] = mult if digit == -1 else templates[digit]
    return ocr, np.repeat(frame[:, :, None], 3, axis=2)


@pytest.mark.parametrize('offset', [(0, 0), (0, -2), (-6, -6), (6, 6), (2, -3)])
@pytest.mark.parametrize('side', ['1P', '2P'])
def test_scores_match_reference(offset: tuple, side: str) -> None:
    ocr, frame = fixture((offset, offset))
    actual = candidate_scores(frame, side, ocr)
    reference = np.array([ocr._match_mult_cell(_crop_score_roi(frame, side, p), side)
                          for p in CANDIDATES])
    np.testing.assert_allclose(actual, reference, atol=1e-6)
    assert select_offset(actual) == select_offset(reference) == offset


@pytest.mark.parametrize('trim', [1, 4])
def test_sliding_template_matches_reference(trim: int) -> None:
    """セルより小さいテンプレでも、候補内の最大値を同じ窓から採る。"""
    ocr, frame = fixture(((0, -2), (0, 0)))
    ocr._mult_template_gray = ocr._mult_template_gray[trim:-trim, trim:-trim].copy()
    actual = candidate_scores(frame, '1P', ocr)
    reference = np.array([ocr._match_mult_cell(_crop_score_roi(frame, '1P', p), '1P')
                          for p in CANDIDATES])
    np.testing.assert_allclose(actual, reference, atol=1e-6)
    assert select_offset(actual) == select_offset(reference)


@pytest.mark.parametrize('offset', [(0, 0), (0, -2), (-6, 6), (6, -6)])
def test_exactly_twelve_and_independent_sides(offset: tuple) -> None:
    ocr, frame = fixture((offset, (0, 0)))
    alignment = ScoreRoiAlignment(ocr)
    for i in range(FORMULA_SAMPLES-1):
        alignment.observe(frame, i/5)
        assert ocr._region_offsets == {'1P': (0, 0), '2P': (0, 0)}
    alignment.observe(frame, (FORMULA_SAMPLES-1)/5)
    assert alignment.ready
    assert ocr._region_offsets == {'1P': offset, '2P': (0, 0)}
    assert all(ocr.read_formula_side(frame, side).valid for side in ('1P', '2P'))
    alignment._sample = Mock(side_effect=AssertionError('確定後の再探索'))
    alignment.observe(frame, 100)


def test_zero_offset_exact_output() -> None:
    ocr, frame = fixture()
    reference = deepcopy(ocr)
    alignment = ScoreRoiAlignment(ocr)
    for i in range(20):
        alignment.observe(frame, i/5)
        assert ocr.read(frame) == reference.read(frame)
        for side in ('1P', '2P'):
            assert ocr.read_side_detail(frame, side) == reference.read_side_detail(frame, side)
            assert ocr.read_formula_side(frame, side) == reference.read_formula_side(frame, side)


def test_throttling_and_no_formula() -> None:
    ocr, frame = fixture()
    alignment = ScoreRoiAlignment(ocr)
    alignment.observe(frame, 0)
    for i in range(1, 6):
        alignment.observe(frame, i/30)
    assert alignment.counts == {'1P': 1, '2P': 1}
    for i in range(100):
        alignment.observe(np.zeros_like(frame), 1+i/5)
    assert alignment.counts == {'1P': 1, '2P': 1}


def test_mult_alone_does_not_count() -> None:
    ocr, frame = fixture()
    frame[:, :515] = 0
    alignment = ScoreRoiAlignment(ocr)
    alignment.observe(frame, 0)
    assert alignment.counts == {'1P': 0, '2P': 1}


def test_missing_template_and_bad_frame() -> None:
    alignment = ScoreRoiAlignment(ScoreOcr())
    alignment.observe(np.zeros((1080, 1920), np.uint8), 0)
    alignment.observe(None, 1)
    assert not alignment.ready
    assert alignment.counts == {'1P': 0, '2P': 0}


def test_tie_chooses_zero_then_vertical() -> None:
    scores = np.ones(len(CANDIDATES))
    assert select_offset(scores) == (0, 0)
    scores[:] = 0
    for p in [(1, 0), (0, -1), (2, 2)]:
        scores[CANDIDATES.index(p)] = 1
    assert select_offset(scores) == (0, -1)


@pytest.mark.parametrize('fixed,expected', [
    (0.70, (0, -2)), (0.79, (0, -2)),
    (FIXED_ROI_NCC_MIN, (0, 0)), (0.88, (0, 0)), (0.92, (0, 0)),
])
def test_fixed_window_priority(fixed: float, expected: tuple[int, int]) -> None:
    scores = np.zeros(len(CANDIDATES))
    scores[ZERO_OFFSET_INDEX] = fixed
    scores[CANDIDATES.index((0, -2))] = 0.97
    assert select_final_offset(scores) == expected


def test_priority_uses_mean_not_last_sample(monkeypatch: pytest.MonkeyPatch) -> None:
    """最後の低相関コマだけで、十分読めていた固定枠を移動しない。"""
    import src.score_roi_alignment as module
    ocr, frame = fixture(((0, -2), (0, -2)))
    alignment = ScoreRoiAlignment(ocr)
    scores = np.zeros(len(CANDIDATES))
    scores[CANDIDATES.index((0, -2))] = 0.97
    monkeypatch.setattr(module, 'candidate_scores', lambda *args: scores.copy())
    for index in range(FORMULA_SAMPLES):
        scores[ZERO_OFFSET_INDEX] = 0.88 if index < FORMULA_SAMPLES-1 else 0.70
        alignment.observe(frame, index/5)
    assert alignment.ready
    assert alignment.offsets == dict.fromkeys(('1P', '2P'), (0, 0))


def test_pipeline_defaults_and_reset() -> None:
    from src.recognition_pipeline import RecognitionPipeline
    from tests.test_recognition_pipeline import _StubImageReader, _StubMatchDetector
    from src.board import Board
    assert inspect.signature(RecognitionPipeline).parameters['enable_score_roi_alignment'].default is False
    assert inspect.signature(RecognitionPipeline.load_default).parameters['enable_score_roi_alignment'].default is False
    ocr, frame = fixture(((0, -2), (0, 0)))
    pipe = RecognitionPipeline(_StubImageReader(Board(), Board()), _StubMatchDetector(),
                               score_ocr=ocr, enable_score_roi_alignment=True)
    for i in range(FORMULA_SAMPLES):
        pipe._score_roi_alignment.observe(frame, i/5)
    pipe.reset()
    assert pipe._score_roi_alignment.ready
    assert ocr._region_offsets['1P'] == (0, -2)


def test_flag_delivery(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.phase_j.live_speed import SPEED_DEFAULTS, recognition_config, validate
    from src.production_config import LIVE_SCORE_ROI_ALIGNMENT_DEFAULT, LIVE_SCORE_ROI_ALIGNMENT_DELIVERY
    from src.phase_j.launcher import UserConfig
    assert SPEED_DEFAULTS['align_score_roi'] is LIVE_SCORE_ROI_ALIGNMENT_DEFAULT is True
    assert UserConfig.align_score_roi is True
    assert LIVE_SCORE_ROI_ALIGNMENT_DELIVERY[0].flag == 'align_score_roi'
    assert validate({'align_score_roi': False})['align_score_roi'] is False
    monkeypatch.delenv('PUYO_ALIGN_SCORE_ROI', raising=False)
    assert 'enable_score_roi_alignment' not in recognition_config({}, False)
    monkeypatch.setenv('PUYO_ALIGN_SCORE_ROI', '1')
    assert recognition_config({}, False)['enable_score_roi_alignment'] is True
    with pytest.raises(ValueError):
        validate({'align_score_roi': 'true'})


def test_status_delivery() -> None:
    from src.phase_j.live_device_session import DeviceSession
    ocr, _ = fixture()
    session = DeviceSession.__new__(DeviceSession)
    session.pipe = SimpleNamespace(_score_roi_alignment=ScoreRoiAlignment(ocr))
    session.epoch, session.last_status = 0, None
    session.queue, session._input_diagnostics = Mock(), {}
    session.emit(dict(phase='ready', progress=100))
    assert session.queue.put.call_args.args[0][1]['score_roi_alignment']['1P']['ready'] is False


@pytest.mark.parametrize('offset', [(0, 0), (0, -2)])
def test_pipeline_update_applies_before_shared_cells(offset: tuple) -> None:
    from src.recognition_pipeline import RecognitionPipeline
    from tests.test_recognition_pipeline import _StubImageReader, _StubMatchDetector
    from src.board import Board
    ocr, frame = fixture((offset, offset))
    pipe = RecognitionPipeline(_StubImageReader(Board(), Board()), _StubMatchDetector(),
                               score_ocr=ocr, enable_score_roi_alignment=True,
                               enable_formula_value_read=True)
    for i in range(FORMULA_SAMPLES):
        pipe.update(i, i/5, frame)
    assert pipe._score_roi_alignment.ready
    assert pipe._score_ocr._region_offsets == dict.fromkeys(('1P', '2P'), offset)
    assert pipe._formula_last_read_1p.valid
    assert pipe._formula_last_read_2p.valid


@pytest.mark.parametrize('phase', ['calibrating', 'ready'])
def test_public_status_is_schema_valid(phase: str) -> None:
    from tests.phase_j.test_live_publish import ASSETS, row
    from src.phase_j.live_publish import initial_snapshot, result_snapshot
    from src.phase_j.validator import validate_snapshot
    ocr, _ = fixture()
    status = ScoreRoiAlignment(ocr).status()
    data = dict(row(), input_calibration=dict(phase=phase,
                progress=100 if phase == 'ready' else 42, score_roi_alignment=status))
    snapshot = result_snapshot(initial_snapshot(ASSETS), data, 2, 1, None)
    assert validate_snapshot(snapshot).is_valid
    assert snapshot.to_mapping()['display']['score_roi_alignment'] == status


def test_b3_input_uses_same_offsets(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.phase_j.live_b3 import capture
    from src.board import Board
    from tests.test_recognition_pipeline import _make_pipe
    pipe = _make_pipe(Board(), Board())
    ocr, frame = fixture(((0, -2), (0, 0)))
    pipe._score_ocr = ocr
    pipe._score_roi_alignment = ScoreRoiAlignment(ocr)
    for i in range(FORMULA_SAMPLES):
        pipe._score_roi_alignment.observe(frame, i/5)
    other, _ = fixture()
    pipe._b3_capture = SimpleNamespace(ocr=other, update=Mock(return_value=({}, ())))
    pipe._b3_epoch = (None, True)
    pipe._b3_queues = SimpleNamespace(update=Mock(return_value=()), current_game=1)
    monkeypatch.setenv('PUYO_B3_JUDGMENT', '1')
    capture(pipe, SimpleNamespace(), SimpleNamespace(image=frame, index=20, media_sec=4), (0, 0))
    assert other.read_formula_side(frame, '1P').valid
    assert other._region_offsets == ocr._region_offsets


@pytest.mark.parametrize('arguments,enabled', [([], True), (['--align-score-roi'], True),
                                             (['--no-align-score-roi'], False)])
def test_cli_optional_flag(arguments: list[str], enabled: bool) -> None:
    import argparse
    from src.phase_j.live_speed import add_arguments
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    assert parser.parse_args(arguments).align_score_roi is enabled


@pytest.mark.parametrize('enabled', [True, False])
def test_product_flag_reaches_worker(enabled: bool, monkeypatch: pytest.MonkeyPatch,
                                    tmp_path: Path) -> None:
    """未指定の製品ONと明示OFFがJSON・CLI・子process用設定を通過する。"""
    import argparse
    from src.phase_j.launcher import parse_user_config, build_pipeline_config
    from src.phase_j.live_speed import add_arguments, configure, recognition_config
    data = dict(device_name='test')
    if not enabled:
        data['align_score_roi'] = False
    config = build_pipeline_config(parse_user_config(data, tmp_path))
    assert config['align_score_roi'] is enabled
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    options = parser.parse_args([])
    options.align_score_roi = config['align_score_roi']
    options.cpu_threads, options.source = 1, 'dshow'
    # configureが更新する環境を試験終了時に復元する。
    from src.phase_j.live_speed import SPEED_DEFAULTS, WORKERS_KEY
    for key in (*SPEED_DEFAULTS, WORKERS_KEY):
        monkeypatch.setenv('PUYO_'+key.upper(), '0')
    configure(options, parser)
    effective = recognition_config({}, True)
    assert effective.get('enable_score_roi_alignment', False) is enabled
