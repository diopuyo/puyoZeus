"""src/board_recognition_pipeline.py のテスト。"""
from __future__ import annotations

import numpy as np
import pytest

from src.board import (
    BOARD_COLS,
    BOARD_ROWS,
    Board,
    COLOR_EMPTY,
    HIDDEN_ROWS,
)
from src.board_recognition_pipeline import BoardRecognitionPipeline
from src.image_reader import ImageReader


def _empty_frame(h: int = 1080, w: int = 1920) -> np.ndarray:
    """全黒フレーム (HSV V<EMPTY_V_THRESHOLD)。"""
    return np.zeros((h, w, 3), dtype=np.uint8)


def test_pipeline_initial_call_returns_empty_board() -> None:
    """初回呼び出し → 全空盤面 (基本 ImageReader の挙動)。"""
    reader = ImageReader()
    pipeline = BoardRecognitionPipeline(reader)
    b1, b2 = pipeline.read(_empty_frame())
    for r in range(BOARD_ROWS):
        for c in range(BOARD_COLS):
            assert b1.get(r, c) == COLOR_EMPTY
            assert b2.get(r, c) == COLOR_EMPTY


def test_pipeline_with_all_layers_disabled_matches_image_reader() -> None:
    """全レイヤー OFF → ImageReader と完全一致。"""
    reader = ImageReader()
    pipeline = BoardRecognitionPipeline(
        reader, use_anim_filter=False, use_smoother=False,
        use_tracker=False, use_adaptive_bg=False,
    )
    frame = _empty_frame()
    pb1, pb2 = pipeline.read(frame)
    rb1, rb2 = reader.read_both_boards(frame)
    for r in range(BOARD_ROWS):
        for c in range(BOARD_COLS):
            assert pb1.get(r, c) == rb1.get(r, c)
            assert pb2.get(r, c) == rb2.get(r, c)


def test_pipeline_reset_clears_state() -> None:
    """reset() 後は初期状態に戻る。"""
    reader = ImageReader()
    pipeline = BoardRecognitionPipeline(reader)
    pipeline.read(_empty_frame())
    pipeline.read(_empty_frame())
    pipeline.reset()
    # reset 後 _last_b1 は None (内部状態確認、private アクセス)
    assert pipeline._last_b1 is None
    assert pipeline._last_b2 is None
    assert pipeline._frame_count == 0


def test_pipeline_anim_filter_holds_last_board() -> None:
    """連鎖アニメ判定で前盤面保持 (一致確認)。

    1 フレーム目で _last_b1 を確定後、2 フレーム目で大きな変化を与えると
    AnimationFilter が動いて _last_b1 のまま返るはず。
    結果が「f1 の認識結果と一致」を確認する (絶対値は ImageReader の挙動次第)。
    """
    reader = ImageReader()
    pipeline = BoardRecognitionPipeline(
        reader, use_anim_filter=True, use_smoother=False,
        use_tracker=False, use_adaptive_bg=False,
    )
    # 1 フレーム目: 全黒、認識結果を取得
    f1 = _empty_frame()
    expected_b1, expected_b2 = pipeline.read(f1)
    # 2 フレーム目: 全白 (大きな変化)
    f2 = np.full((1080, 1920, 3), 250, dtype=np.uint8)
    b1, b2 = pipeline.read(f2)
    # AnimationFilter で 前盤面 (expected) と一致するはず
    for r in range(BOARD_ROWS):
        for c in range(BOARD_COLS):
            assert b1.get(r, c) == expected_b1.get(r, c)
            assert b2.get(r, c) == expected_b2.get(r, c)


def test_pipeline_handles_invalid_frame() -> None:
    """None / 空配列 → 全空盤面。"""
    reader = ImageReader()
    pipeline = BoardRecognitionPipeline(reader)
    b1, b2 = pipeline.read(np.array([]))
    assert b1.get(HIDDEN_ROWS, 0) == COLOR_EMPTY
    assert b2.get(HIDDEN_ROWS, 0) == COLOR_EMPTY


def test_pipeline_initialize_background_with_empty_list() -> None:
    """空フレームリスト → 例外なく完了。"""
    reader = ImageReader()
    pipeline = BoardRecognitionPipeline(reader)
    pipeline.initialize_background([])
    # 何も起こらない (背景 FP セットなし)
    assert pipeline._adaptive_p1 is None or hasattr(
        pipeline._adaptive_p1, "cell_at",
    )
