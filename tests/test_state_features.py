"""W2.1 state_features のテスト。"""
from __future__ import annotations

import numpy as np

from src.board import (
    COLOR_BLUE,
    COLOR_EMPTY,
    COLOR_RED,
    COLOR_UNKNOWN,
    Board,
)
from src.state_features import (
    BOARD_FEATURE_DIM,
    NUM_COLOR_CLASSES,
    OJAMA_MAX,
    PAIR_FEATURE_DIM,
    TOTAL_FEATURE_DIM,
    encode_board,
    encode_ojama,
    encode_pair,
    encode_score,
    encode_state,
)
from src.state_pipeline import GameState


def _empty_state() -> GameState:
    return GameState(
        t_sec=0.0,
        board_p1=Board(),
        board_p2=Board(),
        next_p1=None, next_p2=None,
        dnext_p1=None, dnext_p2=None,
        score_p1=None, score_p2=None,
        score_confidence_p1=0.0, score_confidence_p2=0.0,
        pending_ojama_p1=0, pending_ojama_p2=0,
        is_match_end_locked=False, is_telop_visible=False,
    )


def test_encode_board_empty_returns_one_hot_empty() -> None:
    """空盤面は全セル EMPTY (idx 0) one-hot。"""
    b = Board()
    enc = encode_board(b)
    # 全 12*6 セルで idx 0 が 1.0
    assert enc[:, :, 0].sum() == 12 * 6
    # その他のクラスは 0
    assert enc[:, :, 1:].sum() == 0


def test_encode_board_with_red_cells() -> None:
    """赤セルは idx 1 に 1.0。"""
    b = Board()
    b.set(11, 2, COLOR_RED)
    b.set(12, 2, COLOR_RED)
    enc = encode_board(b)
    # 12 行 × 6 列で 70 セル EMPTY、2 セル RED
    assert enc[:, :, 0].sum() == 12 * 6 - 2
    assert enc[:, :, 1].sum() == 2


def test_encode_board_unknown_treated_as_empty() -> None:
    """UNKNOWN セルは EMPTY 扱い。"""
    b = Board()
    b.set(12, 2, COLOR_UNKNOWN)
    enc = encode_board(b)
    # UNKNOWN セルが EMPTY の idx 0 にマッピング
    assert enc[11, 2, 0] == 1.0  # visible row 11 (= row 12)


def test_encode_pair_none_returns_zeros() -> None:
    enc = encode_pair(None)
    assert enc.shape == (PAIR_FEATURE_DIM,)
    assert enc.sum() == 0


def test_encode_pair_returns_two_one_hots() -> None:
    enc = encode_pair((COLOR_RED, COLOR_BLUE))
    # top=RED (idx 1) + bot=BLUE (idx 2 + 7 offset)
    assert enc[1] == 1.0
    assert enc[NUM_COLOR_CLASSES + 2] == 1.0
    assert enc.sum() == 2.0


def test_encode_score_none_zero() -> None:
    assert encode_score(None) == 0.0


def test_encode_score_normalized() -> None:
    """log1p 正規化、0..1 範囲に収まる。"""
    # 99999 でほぼ 1 に近い値
    val = encode_score(99999)
    assert 0.5 < val < 1.0


def test_encode_ojama_clip() -> None:
    assert encode_ojama(0) == 0.0
    assert encode_ojama(100) == 100 / OJAMA_MAX
    # 200+ はクリップ
    assert encode_ojama(500) == 1.0
    assert encode_ojama(-1) == 0.0


def test_encode_state_total_dim() -> None:
    """encode_state は TOTAL_FEATURE_DIM 次元の vector を返す。"""
    state = _empty_state()
    vec = encode_state(state)
    assert vec.shape == (TOTAL_FEATURE_DIM,)
    assert vec.dtype == np.float32


def test_encode_state_with_real_data() -> None:
    """実データ風の state でエンコード成功。"""
    b1 = Board()
    b1.set(11, 2, COLOR_RED)
    b1.set(12, 2, COLOR_BLUE)
    b2 = Board()
    b2.set(12, 0, COLOR_RED)
    state = GameState(
        t_sec=10.0,
        board_p1=b1, board_p2=b2,
        next_p1=(COLOR_RED, COLOR_BLUE),
        next_p2=(COLOR_BLUE, COLOR_RED),
        dnext_p1=(COLOR_RED, COLOR_RED),
        dnext_p2=(COLOR_BLUE, COLOR_BLUE),
        score_p1=1000, score_p2=500,
        score_confidence_p1=0.9, score_confidence_p2=0.9,
        pending_ojama_p1=10, pending_ojama_p2=20,
        is_match_end_locked=False, is_telop_visible=True,
    )
    vec = encode_state(state)
    assert vec.shape == (TOTAL_FEATURE_DIM,)
    # 0 でない値を持つ (盤面・next・score・ojama 込み)
    assert vec.sum() > 0
