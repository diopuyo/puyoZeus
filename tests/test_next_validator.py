"""NextValidator のテスト."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest

from src.board import (
    BOARD_COLS,
    BOARD_ROWS,
    COLOR_BLUE,
    COLOR_EMPTY,
    COLOR_GREEN,
    COLOR_RED,
    Board,
)
from src.board_state_machine import BoardState
from src.self_supervised.next_validator import (
    CONTINUITY_MATCH_CONFIDENCE,
    NextValidator,
    PLACEMENT_MATCH_CONFIDENCE,
    STABLE_AGREE_MIN,
    STABLE_MATCH_CONFIDENCE,
    _color_count_delta,
    _delta_matches,
    _expected_delta_from_pair,
    _recover_pair_from_delta,
)


# ============================
# モック
# ============================


@dataclass
class _MockSide:
    state: BoardState = BoardState.STABLE
    confirmed_board: Any = None
    next_pair: tuple[int, int] | None = None
    dnext_pair: tuple[int, int] | None = None


@dataclass
class _MockResult:
    is_match_active: bool = True
    p1: _MockSide = None
    p2: _MockSide = None


def _empty_board() -> Board:
    return Board()


def _board_with(cells: dict[tuple[int, int], int]) -> Board:
    b = Board()
    for (r, c), color in cells.items():
        b.set(r, c, color)
    return b


def _frame_1080p() -> np.ndarray:
    return np.zeros((1080, 1920, 3), dtype=np.uint8)


# ============================
# helper の単体テスト
# ============================


def test_color_count_delta_no_change():
    """同一 board は delta なし."""
    b1 = _board_with({(11, 0): COLOR_RED})
    b2 = _board_with({(11, 0): COLOR_RED})
    assert _color_count_delta(b1, b2) == {}


def test_color_count_delta_addition():
    """puyo が +1 された場合 delta=+1."""
    b1 = _empty_board()
    b2 = _board_with({(11, 0): COLOR_RED, (11, 1): COLOR_BLUE})
    delta = _color_count_delta(b1, b2)
    assert delta == {COLOR_RED: 1, COLOR_BLUE: 1}


def test_expected_delta_from_pair_different_colors():
    pair = (COLOR_RED, COLOR_BLUE)
    assert _expected_delta_from_pair(pair) == {COLOR_RED: 1, COLOR_BLUE: 1}


def test_expected_delta_from_pair_same_color():
    pair = (COLOR_RED, COLOR_RED)
    assert _expected_delta_from_pair(pair) == {COLOR_RED: 2}


def test_delta_matches_match():
    actual = {COLOR_RED: 1, COLOR_BLUE: 1}
    expected = {COLOR_RED: 1, COLOR_BLUE: 1}
    assert _delta_matches(actual, expected)


def test_delta_matches_no_match():
    actual = {COLOR_RED: 1, COLOR_GREEN: 1}
    expected = {COLOR_RED: 1, COLOR_BLUE: 1}
    assert not _delta_matches(actual, expected)


def test_recover_pair_from_delta_two_colors():
    delta = {COLOR_RED: 1, COLOR_BLUE: 1}
    pair = _recover_pair_from_delta(delta)
    assert pair is not None
    assert set(pair) == {COLOR_RED, COLOR_BLUE}


def test_recover_pair_from_delta_same_color():
    delta = {COLOR_RED: 2}
    pair = _recover_pair_from_delta(delta)
    assert pair == (COLOR_RED, COLOR_RED)


def test_recover_pair_from_delta_invalid():
    delta = {COLOR_RED: 3}  # 3 puyo は invalid
    assert _recover_pair_from_delta(delta) is None
    assert _recover_pair_from_delta({}) is None


# ============================
# Validator 動作
# ============================


def test_validator_init_invalid_window():
    with pytest.raises(ValueError):
        NextValidator(history_window=2)


def test_validator_no_emit_when_inactive():
    v = NextValidator()
    res = _MockResult(is_match_active=False, p1=_MockSide(), p2=_MockSide())
    for i in range(5):
        v.update(i, i * 0.1, res, _frame_1080p())
    assert v.collect() == []


def test_validator_placement_trace_match():
    """STABLE → 配置 → STABLE で next_pair と delta が一致 → high confidence emit."""
    v = NextValidator()
    frame = _frame_1080p()
    # 1) 初 STABLE: 空盤面 + next=(R, B)
    b1 = _empty_board()
    res1 = _MockResult(p1=_MockSide(
        state=BoardState.STABLE, confirmed_board=b1,
        next_pair=(COLOR_RED, COLOR_BLUE),
    ), p2=_MockSide(state=BoardState.STABLE, confirmed_board=_empty_board()))
    v.update(0, 0.0, res1, frame)
    # 2) TSUMO_FALL (一時、emit 期待しない)
    res_t = _MockResult(p1=_MockSide(
        state=BoardState.TSUMO_FALL, confirmed_board=b1,
        next_pair=(COLOR_GREEN, COLOR_RED),
    ), p2=_MockSide(state=BoardState.STABLE, confirmed_board=_empty_board()))
    v.update(1, 0.1, res_t, frame)
    # 3) 次 STABLE: 盤面に R+B が 1 個ずつ追加 (+ 新 next pair)
    b2 = _board_with({(11, 0): COLOR_RED, (12, 0): COLOR_BLUE})
    res2 = _MockResult(p1=_MockSide(
        state=BoardState.STABLE, confirmed_board=b2,
        next_pair=(COLOR_GREEN, COLOR_RED),
    ), p2=_MockSide(state=BoardState.STABLE, confirmed_board=_empty_board()))
    v.update(2, 0.2, res2, frame)
    samples = v.collect()
    # 1 件以上 emit されているはず
    placement = [
        s for s in samples
        if s.metadata.get("source") == "placement_trace_match"
    ]
    assert len(placement) >= 1
    assert placement[0].confidence == PLACEMENT_MATCH_CONFIDENCE
    assert placement[0].label["top_color"] == COLOR_RED
    assert placement[0].label["bot_color"] == COLOR_BLUE


def test_validator_placement_trace_misread_correction():
    """next_pair が誤検出だった場合、配置色から正解を逆算して emit."""
    v = NextValidator()
    frame = _frame_1080p()
    # 初 STABLE: next=(R, R) と誤検出
    b1 = _empty_board()
    res1 = _MockResult(p1=_MockSide(
        state=BoardState.STABLE, confirmed_board=b1,
        next_pair=(COLOR_RED, COLOR_RED),  # 誤検出
    ), p2=_MockSide(state=BoardState.STABLE, confirmed_board=_empty_board()))
    v.update(0, 0.0, res1, frame)
    # 配置: 実際は R+B (= 正解)
    b2 = _board_with({(11, 0): COLOR_RED, (12, 0): COLOR_BLUE})
    res2 = _MockResult(p1=_MockSide(
        state=BoardState.STABLE, confirmed_board=b2,
        next_pair=(COLOR_GREEN, COLOR_RED),
    ), p2=_MockSide(state=BoardState.STABLE, confirmed_board=_empty_board()))
    v.update(1, 0.1, res2, frame)
    samples = v.collect()
    correct = [
        s for s in samples
        if s.metadata.get("source") == "placement_trace_correct"
    ]
    assert len(correct) >= 1
    # 正解は R + B
    pair = (correct[0].label["top_color"], correct[0].label["bot_color"])
    assert set(pair) == {COLOR_RED, COLOR_BLUE}


def test_validator_continuity_match():
    """直前 dnext == 現 next なら continuity emit."""
    v = NextValidator()
    frame = _frame_1080p()
    pair = (COLOR_RED, COLOR_BLUE)
    # frame 1: dnext=pair
    res1 = _MockResult(p1=_MockSide(
        state=BoardState.STABLE, confirmed_board=_empty_board(),
        next_pair=(COLOR_GREEN, COLOR_RED),
        dnext_pair=pair,
    ), p2=_MockSide(state=BoardState.STABLE, confirmed_board=_empty_board()))
    v.update(0, 0.0, res1, frame)
    # frame 2: next=pair (= 1 個前 dnext と一致)
    res2 = _MockResult(p1=_MockSide(
        state=BoardState.TSUMO_FALL, confirmed_board=_empty_board(),
        next_pair=pair,
        dnext_pair=(COLOR_GREEN, COLOR_BLUE),
    ), p2=_MockSide(state=BoardState.STABLE, confirmed_board=_empty_board()))
    v.update(1, 0.1, res2, frame)
    samples = v.collect()
    cont = [
        s for s in samples
        if s.metadata.get("source") == "continuity_match"
    ]
    assert len(cont) >= 1
    assert cont[0].confidence == CONTINUITY_MATCH_CONFIDENCE


def test_validator_reset_clears_state():
    v = NextValidator()
    frame = _frame_1080p()
    # 何度か update して emit
    b1 = _empty_board()
    b2 = _board_with({(11, 0): COLOR_RED, (12, 0): COLOR_BLUE})
    res1 = _MockResult(p1=_MockSide(
        state=BoardState.STABLE, confirmed_board=b1,
        next_pair=(COLOR_RED, COLOR_BLUE),
    ), p2=_MockSide(state=BoardState.STABLE, confirmed_board=_empty_board()))
    v.update(0, 0.0, res1, frame)
    res2 = _MockResult(p1=_MockSide(
        state=BoardState.STABLE, confirmed_board=b2,
        next_pair=(COLOR_GREEN, COLOR_RED),
    ), p2=_MockSide(state=BoardState.STABLE, confirmed_board=_empty_board()))
    v.update(1, 0.1, res2, frame)
    v.collect()
    # reset で内部 _stable_1p / history がクリア
    v.reset()
    # 同じ flow を再実行 → 再度 emit できる
    v.update(2, 0.2, res1, frame)
    v.update(3, 0.3, res2, frame)
    samples = v.collect()
    placement = [
        s for s in samples
        if s.metadata.get("source") == "placement_trace_match"
    ]
    assert len(placement) >= 1


# ============================
# Phase I 改良: STABLE 持続性 emit
# ============================


def test_validator_stable_persistence_emit():
    """STABLE 中に同一 next_pair が連続観測で MEDIUM emit."""
    v = NextValidator()
    frame = _frame_1080p()
    pair = (COLOR_RED, COLOR_BLUE)
    # 3 frame 連続で同じ STABLE + 同じ next_pair (盤面は変えない、同一 STABLE)
    res = _MockResult(p1=_MockSide(
        state=BoardState.STABLE, confirmed_board=_empty_board(),
        next_pair=pair,
    ), p2=_MockSide(state=BoardState.STABLE, confirmed_board=_empty_board()))
    for i in range(STABLE_AGREE_MIN):
        v.update(i, i * 0.2, res, frame)
    samples = v.collect()
    stable = [
        s for s in samples
        if s.metadata.get("source") == "stable_persistence"
    ]
    assert len(stable) >= 1
    s0 = stable[0]
    assert s0.confidence == STABLE_MATCH_CONFIDENCE
    assert s0.label["top_color"] == COLOR_RED
    assert s0.label["bot_color"] == COLOR_BLUE


def test_validator_stable_persistence_dedup():
    """同一 (side, pair) は 1 試合中 1 度のみ emit."""
    v = NextValidator()
    frame = _frame_1080p()
    pair = (COLOR_RED, COLOR_BLUE)
    res = _MockResult(p1=_MockSide(
        state=BoardState.STABLE, confirmed_board=_empty_board(),
        next_pair=pair,
    ), p2=_MockSide(state=BoardState.STABLE, confirmed_board=_empty_board()))
    # 6 frame 連続観測でも 1 度しか emit されない
    for i in range(6):
        v.update(i, i * 0.2, res, frame)
    samples = v.collect()
    stable_1p = [
        s for s in samples
        if s.metadata.get("source") == "stable_persistence"
        and s.input_data.get("side") == "1P"
    ]
    assert len(stable_1p) == 1


def test_validator_stable_persistence_different_pair_resets():
    """next_pair が変わった後は新たに emit 可能."""
    v = NextValidator()
    frame = _frame_1080p()
    pair_a = (COLOR_RED, COLOR_BLUE)
    pair_b = (COLOR_GREEN, COLOR_RED)
    res_a = _MockResult(p1=_MockSide(
        state=BoardState.STABLE, confirmed_board=_empty_board(),
        next_pair=pair_a,
    ), p2=_MockSide(state=BoardState.STABLE, confirmed_board=_empty_board()))
    res_b = _MockResult(p1=_MockSide(
        state=BoardState.STABLE, confirmed_board=_empty_board(),
        next_pair=pair_b,
    ), p2=_MockSide(state=BoardState.STABLE, confirmed_board=_empty_board()))
    for i in range(STABLE_AGREE_MIN):
        v.update(i, i * 0.2, res_a, frame)
    for i in range(STABLE_AGREE_MIN):
        v.update(STABLE_AGREE_MIN + i,
                 (STABLE_AGREE_MIN + i) * 0.2, res_b, frame)
    samples = v.collect()
    pairs_emitted = {
        (s.label["top_color"], s.label["bot_color"])
        for s in samples
        if s.metadata.get("source") == "stable_persistence"
        and s.input_data.get("side") == "1P"
    }
    assert pair_a in pairs_emitted
    assert pair_b in pairs_emitted


def test_validator_stable_persistence_disabled():
    """enable_stable_emit=False で持続性 emit 無効."""
    v = NextValidator(enable_stable_emit=False)
    frame = _frame_1080p()
    pair = (COLOR_RED, COLOR_BLUE)
    res = _MockResult(p1=_MockSide(
        state=BoardState.STABLE, confirmed_board=_empty_board(),
        next_pair=pair,
    ), p2=_MockSide(state=BoardState.STABLE, confirmed_board=_empty_board()))
    for i in range(STABLE_AGREE_MIN + 2):
        v.update(i, i * 0.2, res, frame)
    samples = v.collect()
    stable = [
        s for s in samples
        if s.metadata.get("source") == "stable_persistence"
    ]
    assert len(stable) == 0


def test_validator_stable_persistence_not_emit_during_action():
    """STABLE 以外の state では持続性 emit 走らない."""
    v = NextValidator()
    frame = _frame_1080p()
    pair = (COLOR_RED, COLOR_BLUE)
    # state=TSUMO_FALL 連続: persistence 不発
    res_t = _MockResult(p1=_MockSide(
        state=BoardState.TSUMO_FALL, confirmed_board=_empty_board(),
        next_pair=pair,
    ), p2=_MockSide(state=BoardState.STABLE, confirmed_board=_empty_board()))
    for i in range(STABLE_AGREE_MIN + 2):
        v.update(i, i * 0.2, res_t, frame)
    samples = v.collect()
    stable_1p = [
        s for s in samples
        if s.metadata.get("source") == "stable_persistence"
        and s.input_data.get("side") == "1P"
    ]
    assert len(stable_1p) == 0


def test_validator_stable_agree_min_validation():
    """stable_agree_min の異常値で ValueError."""
    with pytest.raises(ValueError):
        NextValidator(stable_agree_min=1)


def test_validator_stable_persistence_after_reset():
    """reset 後は同 pair でも再 emit 可能."""
    v = NextValidator()
    frame = _frame_1080p()
    pair = (COLOR_RED, COLOR_BLUE)
    res = _MockResult(p1=_MockSide(
        state=BoardState.STABLE, confirmed_board=_empty_board(),
        next_pair=pair,
    ), p2=_MockSide(state=BoardState.STABLE, confirmed_board=_empty_board()))
    for i in range(STABLE_AGREE_MIN):
        v.update(i, i * 0.2, res, frame)
    samples_before = v.collect()
    n_before = len([
        s for s in samples_before
        if s.metadata.get("source") == "stable_persistence"
        and s.input_data.get("side") == "1P"
    ])
    assert n_before == 1
    v.reset()
    for i in range(STABLE_AGREE_MIN):
        v.update(i + 100, (i + 100) * 0.2, res, frame)
    samples_after = v.collect()
    n_after = len([
        s for s in samples_after
        if s.metadata.get("source") == "stable_persistence"
        and s.input_data.get("side") == "1P"
    ])
    assert n_after == 1
