"""固定枠維持時の較正前後を、合成入力で公開DTOまで比較する。"""
from __future__ import annotations

from copy import deepcopy
import json
from typing import Any

import cv2
import numpy as np
import pytest

from scripts.measure_live_score_roi import json_value
from src.board import Board
from src.score_ocr import ScoreOcr, SCORE_1P_REGION, SCORE_2P_REGION
from src.score_roi_alignment import CANDIDATES, candidate_scores, select_offset

FPS = 30
DURATION_SEC = 180
DELAY_SEC = 60
CYCLE_FRAMES = FPS * 4
CELL_HEIGHT, CELL_WIDTH = 50, 40
TEMPLATE_TOP = 7  # 固定枠より1px上。最良候補は-1でも固定枠を維持する。


def synthetic_inputs() -> tuple[ScoreOcr, tuple[np.ndarray, ...]]:
    """縦相関を持つ文字で、固定窓NCC約0.88・最良窓約1を再現する。"""
    rng = np.random.default_rng(93)
    patches = []
    for _ in range(11):
        patch = rng.integers(0, 256, (CELL_HEIGHT, CELL_WIDTH), dtype=np.uint8)
        patch = cv2.GaussianBlur(patch, (1, 9), 3)
        patch[:5], patch[-5:] = 0, 0
        patches.append(patch)
    ocr = ScoreOcr(dict(enumerate(patches[:10])), mult_template=patches[-1])
    frames = []
    for cells in (((2, 4), (3, 0), (4, 10), (7, 1)),
                  tuple(enumerate((0, 0, 0, 0, 0, 1, 2, 0))),
                  tuple(enumerate((0,) * 8))):
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        for region in (SCORE_1P_REGION, SCORE_2P_REGION):
            for cell, digit in cells:
                y, x = region[0]+TEMPLATE_TOP, region[2]+CELL_WIDTH*cell
                frame[y:y+CELL_HEIGHT, x:x+CELL_WIDTH] = patches[digit][:, :, None]
        frames.append(frame)
    return ocr, tuple(frames)


def make_pair(ocr: ScoreOcr, enabled: bool) -> tuple[Any, Any, Any]:
    from src.phase_j.live_speed import validate
    from src.recognition_pipeline import RecognitionPipeline
    from src.b3_observation import ObservationCapture
    from src.b3_live_display import B3Display
    from tests.test_recognition_pipeline import _StubImageReader, _StubMatchDetector
    pipe = RecognitionPipeline(_StubImageReader(Board(), Board()), _StubMatchDetector(),
        score_ocr=deepcopy(ocr), enable_score_roi_alignment=validate(
            {} if enabled else {'align_score_roi': False})['align_score_roi'],
        enable_formula_value_read=True, enable_match_start_full_clear=True)
    pipe._live_share_scores = True
    observation = ObservationCapture(skip_icon_reading=True, enable_warning_parallel=False)
    return pipe, observation, B3Display()


def published(pair: tuple, index: int, frame: np.ndarray, initial: Any) -> tuple:
    """認識・実OCR・会計・式段・平滑化・公開経路。推論だけ同一のテスト値。"""
    from src.phase_j.live_publish import result_snapshot
    from src.phase_j.live_score_share import shared_scores
    from tests.phase_j.test_live_publish import row
    pipe, observation, display = pair
    stamp = index/FPS
    result = pipe.update(index, stamp, frame)
    observation.ocr._region_offsets = dict(pipe._score_ocr._region_offsets)
    from types import SimpleNamespace
    source = SimpleNamespace(image=frame, index=index, media_sec=stamp)
    scores = shared_scores(pipe, observation.ocr, source)
    counts = tuple(pipe.tsumo_count(side) for side in ('1P', '2P'))
    record, events = observation.update(result, frame, index, stamp, counts, scores=scores)
    display.boundary(record['game_idx'])
    display.chain.observe(record)
    display.prediction(0.25 if index % CYCLE_FRAMES < FPS else 0.75, 1, stamp)
    advantage, probability = display.update(stamp)
    contents = display.chain.decorate(dict(zip(('p1', 'p2'), record['sides'])))
    data = dict(row(), frame=index, t_sec=stamp, probability=probability,
                advantage=advantage, raw_probability=display.raw, b3_contents=contents,
                captured_at=stamp, recognized_at=stamp, evaluated_at=stamp,
                input_calibration=dict(phase='ready', progress=100))
    alignment = pipe._score_roi_alignment
    if alignment is not None:
        data['input_calibration']['score_roi_alignment'] = alignment.status()
    dto = result_snapshot(initial, data, stamp, index, None).to_mapping()
    assert dto['display']['status'] == 'live'
    # 要求された「合わせ中」の診断表示だけはON専用。勝率等は全項目比較する。
    dto['display'].pop('score_roi_alignment', None)
    formulas = (pipe._formula_last_read_1p, pipe._formula_last_read_2p)
    encoded = json.dumps((result, formulas, scores, pipe._live_raw_scores, record, events, dto),
                         default=json_value, sort_keys=True)
    return encoded, record, probability


def test_three_minute_zero_offset_all_outputs(monkeypatch: pytest.MonkeyPatch) -> None:
    """元動画29/65は削除済みのため、180秒・5400コマの合成入力で代替する。"""
    from src.phase_j import live_publish
    from tests.phase_j.test_live_publish import ASSETS
    ocr, frames = synthetic_inputs()
    scores = candidate_scores(frames[0], '1P', ocr)
    assert select_offset(scores) == (0, -1)
    assert 0.80 < scores[CANDIDATES.index((0, 0))] < 0.92
    assert ocr.read_formula_side(frames[0], '1P').valid
    monkeypatch.setattr(ScoreOcr, 'load_default', lambda **kwargs: deepcopy(ocr))
    monkeypatch.setattr(live_publish, 'utc_now', lambda: '2026-10-17T00:00:00+00:00')
    initial = live_publish.initial_snapshot(ASSETS)
    pairs = [make_pair(ocr, enabled) for enabled in (False, True)]
    valid = steps = searching = locked = 0
    probabilities = set()
    try:
        for index in range(FPS * DURATION_SEC):
            phase = (index % CYCLE_FRAMES) // FPS
            frame = frames[0 if phase == 0 else (1 if phase < 3 else 2)]
            if index < FPS * DELAY_SEC:
                frame = frames[1]
            outputs = [published(pair, index, frame, initial) for pair in pairs]
            assert outputs[0][0] == outputs[1][0], index
            alignment = pairs[1][0]._score_roi_alignment
            searching += int(not alignment.ready)
            locked += int(alignment.ready)
            valid += sum(side['formula']['valid'] for side in outputs[0][1]['sides'])
            steps += sum(side['formula_step'] is not None for side in outputs[0][1]['sides'])
            probabilities.add(outputs[0][2])
            if index % (FPS * 30) == 0:
                print(f'合成入力 {index/FPS:.0f}秒: 差0', flush=True)
        assert alignment.offsets == dict.fromkeys(('1P', '2P'), (0, 0))
        assert searching > FPS * DELAY_SEC and locked > 0
        assert valid > 0 and steps > 0 and len(probabilities) > 1
        print(dict(frames=FPS*DURATION_SEC, searching=searching, locked=locked,
                   valid_side_frames=valid, steps=steps, mismatches=0))
    finally:
        for _, observation, _ in pairs:
            observation.close()
