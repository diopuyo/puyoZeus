"""第22試合の実画像と状態機械で連鎖境界を検証する。"""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from src.board import Board
from src.board_state_machine import BoardState, DetectorSignals, StateContext
from src.chain_boundary_visual import incoming_ojama, verified_next_motion
from src.next_slide_detector import NextSlideDetector
from src.recognition_pipeline import RecognitionPipeline
from src.state_detectors import ChainPhaseDetector

FIXTURE = Path(__file__).parent / 'fixtures/material_boundary_20261117.npz'
ROIS = {'next1': (130, 420, 700, 850), 'next2': (130, 420, 1080, 1230),
        'top1': (160, 218, 272, 656), 'top2': (160, 218, 1252, 1636)}


def frame(index: int) -> np.ndarray:
    """必要な領域のみ保存した実画像を元座標へ戻す。"""
    image = np.zeros((1080, 1920, 3), dtype=np.uint8)
    with np.load(FIXTURE) as data:
        for name, (y0, y1, x0, x1) in ROIS.items():
            key = f'f{index}_{name}'
            if key in data:
                image[y0:y1, x0:x1] = data[key]
    return image


@pytest.mark.parametrize('index,expected', [(5558, False), (5616, False),
                                           (5708, False), (5900, True)])
def test_next_real_boundaries(index: int, expected: bool) -> None:
    assert verified_next_motion(frame(index-2), frame(index), '2P') == expected
    detector = NextSlideDetector(side='2P')
    assert detector.update(frame(index-2), frame(index), verify_motion=True).slide_motion == expected


@pytest.mark.parametrize('index', [5758, 5790])
def test_false_ojama_has_no_incoming_shape(index: int) -> None:
    assert not incoming_ojama(frame(index-2), frame(index), '1P')


def test_real_ojama_and_static_top_are_distinct() -> None:
    assert incoming_ojama(frame(6292), frame(6294), '1P')
    assert not incoming_ojama(frame(6294), frame(6294), '1P')
    assert not incoming_ojama(frame(6292), frame(6294), '2P')


@pytest.mark.parametrize('side', ['1P', '2P'])
def test_unavailable_frame_is_not_a_signal(side: str) -> None:
    assert not incoming_ojama(None, frame(6294), side)
    assert not verified_next_motion(np.zeros((1, 1, 3), np.uint8), frame(5900), side)


def test_ojama_signal_has_priority_over_held_event() -> None:
    ctx = StateContext(state=BoardState.CHAIN)
    signals = DetectorSignals(0.0, Board(), True, chain_event=object())
    detector = ChainPhaseDetector()
    assert detector.detect(ctx, signals) == BoardState.CHAIN
    signals = replace(signals, chain_end_ojama_signal=True)
    assert detector.detect(ctx, signals) == BoardState.OJAMA_FALL


def test_rejected_effect_does_not_consume_cooldown() -> None:
    detector = NextSlideDetector(side='2P')
    detector.update(frame(5556), frame(5558), verify_motion=True)
    assert detector.update(frame(5898), frame(5900), verify_motion=True).slide_motion


@pytest.mark.parametrize('enabled,active', [(False, True), (True, False), (True, True)])
def test_pipeline_ojama_clears_only_enabled_active_side(enabled: bool, active: bool) -> None:
    pipe = object.__new__(RecognitionPipeline)
    pipe._chain_end_ojama_signal = enabled
    pipe._active_chain_1p = event = SimpleNamespace(chain_count=4)
    pipe._active_chain_2p = other = SimpleNamespace(chain_count=5)
    pipe._prev_frame = frame(6292)
    result = pipe._visual_ojama_chain_end('1P', frame(6294), active)
    assert result == (enabled and active)
    assert pipe._active_chain_1p is (None if result else event)
    assert pipe._active_chain_2p is other
    if result:
        assert pipe._last_chain_event_for_settle_1p is event
        assert not pipe._visual_ojama_chain_end('1P', frame(6294), active)

@pytest.mark.parametrize('index,ends', [(5558, False), (5616, False),
                                       (5708, False), (5900, True)])
def test_pipeline_next_guard_preserves_or_ends_event(monkeypatch: pytest.MonkeyPatch, index: int, ends: bool) -> None:
    from tests.test_recognition_pipeline import _make_pipe_with_slide_exit_guard, _make_chain_event
    from tests.test_recognition_pipeline import _prime_match_active
    monkeypatch.setenv('PUYO_NEXT_SLIDE_EFFECT_GUARD', '1')
    pipe = _make_pipe_with_slide_exit_guard(None, False)
    _prime_match_active(pipe, frames=35)
    event = _make_chain_event(is_all_clear=False, chain_count=5)
    pipe._active_chain_2p = event
    pipe._chain_until_2p = pipe._chain_event_max_until_2p = 110.0
    pipe._prev_frame = frame(index-2)
    pipe.update(index, index/60, frame(index))
    assert (pipe._active_chain_2p is None) == ends


def test_pipeline_ojama_reaches_state_machine(monkeypatch: pytest.MonkeyPatch) -> None:
    from tests.test_recognition_pipeline import _make_pipe_with_slide_exit_guard, _make_chain_event
    from tests.test_recognition_pipeline import _prime_match_active
    monkeypatch.setenv('PUYO_CHAIN_END_OJAMA_SIGNAL', '1')
    pipe = _make_pipe_with_slide_exit_guard(None, False)
    _prime_match_active(pipe, frames=35)
    pipe._active_chain_1p = _make_chain_event(is_all_clear=False, chain_count=4)
    pipe._chain_until_1p = pipe._chain_event_max_until_1p = 110.0
    pipe._prev_frame = frame(6292)
    result = pipe.update(6294, 104.9, frame(6294))
    assert pipe._active_chain_1p is None
    assert result.p1.state == BoardState.OJAMA_FALL
    assert result.p1.chain_event is None


def test_flags_are_off_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    from tests.test_recognition_pipeline import _make_pipe_with_slide_exit_guard
    monkeypatch.delenv('PUYO_CHAIN_END_OJAMA_SIGNAL', raising=False)
    monkeypatch.delenv('PUYO_NEXT_SLIDE_EFFECT_GUARD', raising=False)
    pipe = _make_pipe_with_slide_exit_guard(None, False)
    assert not pipe._chain_end_ojama_signal
    assert not pipe._next_slide_effect_guard



def test_blank_next_cannot_verify_motion() -> None:
    image = np.zeros((1080, 1920, 3), dtype=np.uint8)
    assert not verified_next_motion(image, image, '1P')
    assert not verified_next_motion(image, image, '2P')
