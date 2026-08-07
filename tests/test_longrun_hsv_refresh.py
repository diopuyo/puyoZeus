"""長時間劣化修正 A+B (`enable_online_hsv_refresh`) のテスト (2026-08-06)。

docs/LONGRUN_DEGRADATION_INVESTIGATION_2026-08-06.md §1/§4。

構成:
    1. フラグ既定値・格納の確認
    2. Fix A: reset() での OnlineHsvCalibrator 較正クリア (フラグON/OFF)
    3. Fix A: reset API が無い場合の再生成フォールバック
    4. Fix B: inject後もupdate()+再inject判定が継続する (フラグON/OFF)
    5. backwards compat (フラグOFFで既存挙動 bit-identical)
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from src.recognition_pipeline import RecognitionPipeline

_ROOT = Path(__file__).resolve().parent.parent


# =============================================================================
# stub helpers (test_visualize_recognition_inject.py の設計を踏襲)
# =============================================================================


@dataclass
class _StubMatchResult:
    state: object
    bg_value: float = 100.0
    bg_saturation: float = 50.0
    samples: int = 1


class _StubMatchDetector:
    """常に IN_MATCH を返すスタブ。"""

    def detect(self, frame: np.ndarray) -> _StubMatchResult:
        from src.match_state import MatchState
        return _StubMatchResult(state=MatchState.IN_MATCH)


class _StubImageReader:
    """固定 empty board を返すスタブ。set_color_ranges_from_simple も spy 対象。"""

    def __init__(self) -> None:
        from src.board import Board
        self._p1 = Board()
        self._p2 = Board()
        self._classifier = MagicMock()
        self._classifier._hsv = MagicMock()
        self._classifier._hsv.set_color_ranges_from_simple = MagicMock()

    def read_both_boards(
        self, frame: np.ndarray,
        p1_roi_offset: tuple[float, float] = (0.0, 0.0),
        p2_roi_offset: tuple[float, float] = (0.0, 0.0),
        skip_tier1_1p: bool = False, skip_tier1_2p: bool = False,
        telop_result: object | None = None,
    ) -> tuple[object, object]:
        return self._p1.copy(), self._p2.copy()


def _dummy_frame() -> "np.ndarray":
    return np.zeros((1080, 1920, 3), dtype=np.uint8)


def _make_pipe(
    enable_online_hsv_refresh: bool = False,
    online_hsv: object = None,
) -> RecognitionPipeline:
    """テスト用の最小構成 pipeline を組み立てる。"""
    return RecognitionPipeline(
        image_reader=_StubImageReader(),  # type: ignore[arg-type]
        match_state_detector=_StubMatchDetector(),  # type: ignore[arg-type]
        score_ocr=None, chain_tracker_1p=None, chain_tracker_2p=None,
        stable_frame_count=2, online_hsv=online_hsv,
        enable_online_hsv_refresh=enable_online_hsv_refresh,
    )


# =============================================================================
# 1. フラグ既定値・格納
# =============================================================================


def test_enable_online_hsv_refresh_default_false() -> None:
    """enable_online_hsv_refresh 未指定時は既定 False (backwards compat)。"""
    pipe = _make_pipe()
    assert pipe._enable_online_hsv_refresh is False


def test_enable_online_hsv_refresh_explicit_true_stored() -> None:
    """enable_online_hsv_refresh=True を明示すると格納される。"""
    pipe = _make_pipe(enable_online_hsv_refresh=True)
    assert pipe._enable_online_hsv_refresh is True


# =============================================================================
# 2. Fix A: reset() での較正クリア
# =============================================================================


def test_reset_clears_online_hsv_state_when_flag_true() -> None:
    """flag=True で reset() すると OnlineHsvCalibrator.reset() が呼ばれ、

    _online_hsv_injected / _online_hsv_injected_colors もクリアされる。
    """
    mock_calibrator = MagicMock()
    pipe = _make_pipe(enable_online_hsv_refresh=True, online_hsv=mock_calibrator)
    pipe._online_hsv_injected = True
    pipe._online_hsv_injected_colors = {1, 2, 3}

    pipe.reset()

    mock_calibrator.reset.assert_called_once()
    assert pipe._online_hsv_injected is False
    assert pipe._online_hsv_injected_colors == set()


def test_reset_does_not_clear_online_hsv_state_when_flag_false() -> None:
    """flag=False (既定) では reset() が較正状態に触れない (bit-identical)。"""
    mock_calibrator = MagicMock()
    pipe = _make_pipe(enable_online_hsv_refresh=False, online_hsv=mock_calibrator)
    pipe._online_hsv_injected = True
    pipe._online_hsv_injected_colors = {1, 2, 3}

    pipe.reset()

    mock_calibrator.reset.assert_not_called()
    assert pipe._online_hsv_injected is True
    assert pipe._online_hsv_injected_colors == {1, 2, 3}


def test_reset_noop_when_online_hsv_is_none_even_if_flag_true() -> None:
    """online_hsv=None の場合、flag=True でも reset() はエラーにならない。"""
    pipe = _make_pipe(enable_online_hsv_refresh=True, online_hsv=None)
    pipe.reset()  # 例外が出ないことの確認


# =============================================================================
# 3. Fix A: reset API が無い場合の再生成フォールバック
# =============================================================================


class _NoResetCalibrator:
    """reset() を持たない較正器スタブ (フォールバック経路のテスト用)。"""

    def __init__(self) -> None:
        self.marker = "original"


def test_reset_recreates_calibrator_when_reset_api_missing() -> None:
    """OnlineHsvCalibrator 相当のオブジェクトに reset() が無ければ、

    同じ型で再生成する (フォールバック経路)。
    """
    stub = _NoResetCalibrator()
    pipe = _make_pipe(enable_online_hsv_refresh=True, online_hsv=stub)

    pipe.reset()

    assert isinstance(pipe._online_hsv, _NoResetCalibrator)
    assert pipe._online_hsv is not stub  # 再生成されている (同一オブジェクトではない)
    assert pipe._online_hsv_injected is False
    assert pipe._online_hsv_injected_colors == set()


# =============================================================================
# 4. Fix B: inject後もupdate()+再inject判定が継続する
# =============================================================================


def _setup_pipe_for_inject(
    injected: bool, calibrator_ranges: dict, enable_online_hsv_refresh: bool,
) -> tuple[RecognitionPipeline, MagicMock]:
    """inject継続テスト用の pipeline + spy を組み立てる共通ヘルパ。"""
    from src.hybrid_classifier import HybridClassifier

    mock_calibrator = MagicMock()
    mock_calibrator.get_per_video_ranges.return_value = calibrator_ranges
    pipe = _make_pipe(
        enable_online_hsv_refresh=enable_online_hsv_refresh, online_hsv=mock_calibrator,
    )
    pipe._online_hsv_injected = injected
    pipe._reader._classifier.__class__ = HybridClassifier  # type: ignore[assignment]
    mock_proba = np.zeros((13, 6, 11), dtype=np.float32)
    mock_hsv = np.zeros((13, 6, 3), dtype=np.float32)
    pipe._reader._classifier.predict_proba_and_hsv_grid = MagicMock(
        return_value=(mock_proba, mock_hsv),
    )
    spy = pipe._reader._classifier._hsv.set_color_ranges_from_simple
    return pipe, spy


def test_online_hsv_inject_continues_after_injected_when_flag_true() -> None:
    """flag=True かつ既に injected=True でも、新しい色が現れれば inject が走る

    (Fix B: 凍結ガード撤廃、コメント本来の「段階的inject」意図)。
    """
    dummy_ranges = {1: (0, 20, 150, 255, 100, 255)}
    pipe, spy = _setup_pipe_for_inject(
        injected=True, calibrator_ranges=dummy_ranges, enable_online_hsv_refresh=True,
    )
    frame = _dummy_frame()
    for i in range(6):
        pipe.update(i, float(i) * 0.05, frame)
    spy.assert_called()  # 既に injected=True だったが新色1で再度呼ばれる


def test_online_hsv_inject_stays_suppressed_when_flag_false_even_with_new_colors() -> None:
    """flag=False (既定) では injected=True の間、新しい色が現れても

    inject は走らない (従来の完全凍結、bit-identical regression確認)。
    """
    dummy_ranges = {1: (0, 20, 150, 255, 100, 255)}
    pipe, spy = _setup_pipe_for_inject(
        injected=True, calibrator_ranges=dummy_ranges, enable_online_hsv_refresh=False,
    )
    frame = _dummy_frame()
    for i in range(6):
        pipe.update(i, float(i) * 0.05, frame)
    spy.assert_not_called()


# =============================================================================
# 5. backwards compat (フラグOFFでbit-identical)
# =============================================================================


def test_online_hsv_refresh_explicit_false_restores_legacy_pipeline_output() -> None:
    """enable_online_hsv_refresh=False を明示しても、未指定時と完全に同じ結果になる。"""
    pipe_default = _make_pipe()
    pipe_explicit = _make_pipe(enable_online_hsv_refresh=False)
    frame = _dummy_frame()
    for i in range(4):
        r1 = pipe_default.update(i, 0.05 * i, frame)
        r2 = pipe_explicit.update(i, 0.05 * i, frame)
        assert r1.p1.state == r2.p1.state
        assert r1.p2.state == r2.p2.state
        assert r1.p1.confirmed_board == r2.p1.confirmed_board
        assert r1.p2.confirmed_board == r2.p2.confirmed_board
