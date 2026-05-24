"""ChainValidator のテスト."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest

from src.board import COLOR_BLUE, COLOR_RED, Board
from src.board_state_machine import BoardState
from src.chain import ChainSimulator
from src.scoring import calculate_chain_score
from src.self_supervised.chain_validator import (
    EXPECTED_FRAMES_PER_CHAIN,
    SCORE_MATCH_CONFIDENCE,
    SCORE_MISMATCH_CONFIDENCE,
    ChainValidator,
)


# ============================
# モック
# ============================


@dataclass
class _MockChainEvent:
    before_board: Board
    chain_count: int
    end_sec: float = 0.0
    trigger_sec: float = 0.0
    total_erased: int = 0
    total_score: int = 0
    base_score: int = 0
    all_clear_bonus_applied: int = 0
    ojama_sent: int = 0
    leftover_score: int = 0
    is_all_clear: bool = False


@dataclass
class _MockSide:
    state: BoardState = BoardState.STABLE
    score: int | None = 0
    chain_event: Any | None = None


@dataclass
class _MockResult:
    is_match_active: bool = True
    p1: _MockSide = None
    p2: _MockSide = None


def _make_chainable_board() -> Board:
    """4 連結赤を含む盤面 (1 連鎖発火)."""
    b = Board()
    # 赤 4 連結 (column 0, rows 9..12)
    for r in range(9, 13):
        b.set(r, 0, COLOR_RED)
    return b


def _frame() -> np.ndarray:
    return np.zeros((1080, 1920, 3), dtype=np.uint8)


# ============================
# Validator 動作
# ============================


def test_validator_init():
    v = ChainValidator()
    assert v.collect() == []


def test_validator_no_emit_when_inactive():
    v = ChainValidator()
    res = _MockResult(is_match_active=False, p1=_MockSide(), p2=_MockSide())
    for i in range(5):
        v.update(i, i * 0.1, res, _frame())
    assert v.collect() == []


def test_validator_emits_on_chain_end_score_match():
    """CHAIN 終了時、score delta が simulate と一致すれば high confidence."""
    v = ChainValidator()
    chainable = _make_chainable_board()
    sim = ChainSimulator()
    cr = sim.simulate(chainable)
    score_result = calculate_chain_score(cr)
    expected_delta = score_result.total_score
    # 1) STABLE → CHAIN 開始 (score=1000, before_board セット)
    ce = _MockChainEvent(before_board=chainable, chain_count=cr.chain_count)
    res_chain = _MockResult(p1=_MockSide(
        state=BoardState.CHAIN, score=1000, chain_event=ce,
    ), p2=_MockSide(state=BoardState.STABLE))
    v.update(0, 0.0, res_chain, _frame())
    # 2) CHAIN 継続 (期待持続 frame に近い)
    n_frames = int(EXPECTED_FRAMES_PER_CHAIN * cr.chain_count)
    for i in range(1, n_frames):
        v.update(i, i / 60.0, res_chain, _frame())
    # 3) CHAIN 終了 → STABLE、score = 1000 + expected_delta
    res_end = _MockResult(p1=_MockSide(
        state=BoardState.STABLE, score=1000 + expected_delta,
    ), p2=_MockSide(state=BoardState.STABLE))
    v.update(n_frames, n_frames / 60.0, res_end, _frame())
    samples = v.collect()
    assert len(samples) == 1
    s = samples[0]
    assert s.confidence == SCORE_MATCH_CONFIDENCE
    assert s.metadata["score_match"] is True
    assert s.metadata["duration_match"] is True
    assert s.label["chain_count"] == cr.chain_count


def test_validator_emits_low_confidence_on_score_mismatch():
    """score delta が simulate と一致しなければ medium confidence."""
    v = ChainValidator()
    chainable = _make_chainable_board()
    sim = ChainSimulator()
    cr = sim.simulate(chainable)
    ce = _MockChainEvent(before_board=chainable, chain_count=cr.chain_count)
    res_chain = _MockResult(p1=_MockSide(
        state=BoardState.CHAIN, score=1000, chain_event=ce,
    ), p2=_MockSide(state=BoardState.STABLE))
    n_frames = int(EXPECTED_FRAMES_PER_CHAIN * cr.chain_count)
    for i in range(n_frames):
        v.update(i, i / 60.0, res_chain, _frame())
    # 終了時 score を意図的にずらす (delta=99999 過大)
    res_end = _MockResult(p1=_MockSide(
        state=BoardState.STABLE, score=1000 + 99999,
    ), p2=_MockSide(state=BoardState.STABLE))
    v.update(n_frames, n_frames / 60.0, res_end, _frame())
    samples = v.collect()
    assert len(samples) == 1
    s = samples[0]
    assert s.confidence == SCORE_MISMATCH_CONFIDENCE
    assert s.metadata["score_match"] is False


def test_validator_emits_low_confidence_on_duration_mismatch():
    """持続時間が想定外なら medium confidence."""
    v = ChainValidator()
    chainable = _make_chainable_board()
    sim = ChainSimulator()
    cr = sim.simulate(chainable)
    score_result = calculate_chain_score(cr)
    expected_delta = score_result.total_score
    ce = _MockChainEvent(before_board=chainable, chain_count=cr.chain_count)
    res_chain = _MockResult(p1=_MockSide(
        state=BoardState.CHAIN, score=1000, chain_event=ce,
    ), p2=_MockSide(state=BoardState.STABLE))
    # 1 frame だけ CHAIN
    v.update(0, 0.0, res_chain, _frame())
    res_end = _MockResult(p1=_MockSide(
        state=BoardState.STABLE, score=1000 + expected_delta,
    ), p2=_MockSide(state=BoardState.STABLE))
    v.update(1, 1 / 60.0, res_end, _frame())
    samples = v.collect()
    assert len(samples) == 1
    s = samples[0]
    assert s.metadata["duration_match"] is False
    # score は match だが duration ng → mismatch
    assert s.confidence == SCORE_MISMATCH_CONFIDENCE


def test_validator_skips_if_no_chain_event():
    """chain_event が None のまま CHAIN→STABLE 遷移しても crash しない."""
    v = ChainValidator()
    res_chain = _MockResult(p1=_MockSide(
        state=BoardState.CHAIN, score=1000, chain_event=None,
    ), p2=_MockSide(state=BoardState.STABLE))
    v.update(0, 0.0, res_chain, _frame())
    res_end = _MockResult(p1=_MockSide(
        state=BoardState.STABLE, score=1500,
    ), p2=_MockSide(state=BoardState.STABLE))
    v.update(1, 0.1, res_end, _frame())
    # crash 無し、emit 無し
    assert v.collect() == []


def test_validator_reset_clears_state():
    v = ChainValidator()
    chainable = _make_chainable_board()
    sim = ChainSimulator()
    cr = sim.simulate(chainable)
    ce = _MockChainEvent(before_board=chainable, chain_count=cr.chain_count)
    res_chain = _MockResult(p1=_MockSide(
        state=BoardState.CHAIN, score=1000, chain_event=ce,
    ), p2=_MockSide(state=BoardState.STABLE))
    v.update(0, 0.0, res_chain, _frame())
    v.reset()
    # reset 後 in_chain=False に戻ったか
    res_end = _MockResult(p1=_MockSide(
        state=BoardState.STABLE, score=2000,
    ), p2=_MockSide(state=BoardState.STABLE))
    v.update(1, 0.1, res_end, _frame())
    # 過去の chain は捨てられたので emit 無し
    assert v.collect() == []
