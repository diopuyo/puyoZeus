"""W1.1 StatePipeline のテスト。"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from src.board import Board
from src.state_pipeline import GameState, StatePipeline


def test_pipeline_construction_default() -> None:
    """既定構成でロードできる。"""
    pipeline = StatePipeline()
    assert pipeline._image_reader is not None
    # NextDetector / ScoreOcr は環境次第で None もありえる
    # 内部状態
    assert pipeline._prev_score_p1 is None
    assert pipeline._pending_ojama_p1 == 0


def test_extract_blank_frame() -> None:
    """全黒フレームでもエラーなく GameState を返す。"""
    pipeline = StatePipeline()
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    state = pipeline.extract(frame, t_sec=0.0)
    assert isinstance(state, GameState)
    assert state.t_sec == 0.0
    # 黒一色なので試合中判定外、両盤面 EMPTY
    assert isinstance(state.board_p1, Board)
    assert isinstance(state.board_p2, Board)


def test_reset_clears_internal_state() -> None:
    """reset() で score 履歴・pending ojama がリセットされる。"""
    pipeline = StatePipeline()
    # 内部状態を直接いじって reset を確認
    pipeline._prev_score_p1 = 10000
    pipeline._pending_ojama_p1 = 5
    pipeline._leftover_p2 = 30
    pipeline.reset()
    assert pipeline._prev_score_p1 is None
    assert pipeline._pending_ojama_p1 == 0
    assert pipeline._leftover_p2 == 0


def test_score_delta_increments_opponent_pending() -> None:
    """1P score 増 → 2P pending_ojama 増加。"""
    pipeline = StatePipeline(use_match_end_detector=False, use_telop_detector=False)
    pipeline.reset(match_start_sec=0.0)
    # 1P が連鎖発火 (score 0 → 5040)
    pipeline._update_ojama_pending(score_p1=0, score_p2=0, t_sec=0.0)
    pipeline._update_ojama_pending(score_p1=5040, score_p2=0, t_sec=10.0)
    # 5040 / 70 = 72 個 ojama (rate_base=70)
    assert pipeline._pending_ojama_p2 > 0
    assert pipeline._pending_ojama_p1 == 0  # 1P 自身は受けない


def test_score_decrease_does_not_decrease_pending() -> None:
    """OCR ノイズで score が逆行しても pending は減らない (差分は 0 扱い)。"""
    pipeline = StatePipeline(use_match_end_detector=False, use_telop_detector=False)
    pipeline.reset(match_start_sec=0.0)
    pipeline._update_ojama_pending(score_p1=10000, score_p2=10000, t_sec=0.0)
    pipeline._update_ojama_pending(score_p1=9500, score_p2=10000, t_sec=1.0)
    assert pipeline._pending_ojama_p1 == 0
    assert pipeline._pending_ojama_p2 == 0


def test_extract_on_real_frame_does_not_crash() -> None:
    """実フレーム (video_01 試合中) でもクラッシュなく動く。"""
    cap = cv2.VideoCapture("data/frames/video_01.mp4")
    if not cap.isOpened():
        pytest.skip("video not available")
    cap.set(cv2.CAP_PROP_POS_MSEC, 220 * 1000)
    ok, fr = cap.read()
    cap.release()
    if not ok or fr is None:
        pytest.skip("frame fetch failed")
    pipeline = StatePipeline()
    pipeline.reset(match_start_sec=200.0)
    state = pipeline.extract(fr, t_sec=220.0)
    assert isinstance(state, GameState)
    assert state.t_sec == 220.0


def test_game_state_immutable() -> None:
    """GameState は frozen dataclass、書き換え不可。"""
    pipeline = StatePipeline()
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    state = pipeline.extract(frame, t_sec=0.0)
    with pytest.raises(Exception):
        state.t_sec = 999.0  # type: ignore[misc]


def test_extract_with_resize() -> None:
    """720p フレームでも 1080p にリサイズして処理される。"""
    pipeline = StatePipeline()
    frame_720p = np.zeros((720, 1280, 3), dtype=np.uint8)
    state = pipeline.extract(frame_720p, t_sec=0.0)
    assert isinstance(state, GameState)


def test_match_end_lockdown_flag() -> None:
    """is_match_end_locked フラグが立つ条件。"""
    pipeline = StatePipeline()
    # 黒フレームではロックダウンしない
    state = pipeline.extract(np.zeros((1080, 1920, 3), dtype=np.uint8), 0.0)
    assert state.is_match_end_locked is False
