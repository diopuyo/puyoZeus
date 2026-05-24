"""InferenceBoardGenerator テスト (Phase B-5)."""

from __future__ import annotations

from dataclasses import dataclass

from src.board import (
    COLOR_BLUE, COLOR_RED, Board,
)
from src.board_state_machine import BoardState, StateContext
from src.inference_board import InferenceBoardGenerator


# ============================
# helper
# ============================


@dataclass
class _StubChainEvent:
    trigger_sec: float
    end_sec: float


def _empty_board() -> Board:
    return Board()


def _board_with_4connect_red() -> Board:
    """1 連鎖発火可能な盤面 (1 列に赤 4 個積み)."""
    b = Board()
    for r in range(9, 13):
        b.set(r, 0, COLOR_RED)
    return b


def _board_with_2chain() -> Board:
    """2 連鎖発火可能な盤面.

    最下段に赤 4 個 (横一列)、その上に青 4 個になるように積む。
    赤消去 → 青が降下して 4 連結 → 連鎖。
    """
    b = Board()
    # 赤 4 個 (最下段、列 0-3)
    for c in range(4):
        b.set(12, c, COLOR_RED)
    # 青 4 個 (1 段上、列 0-3)
    for c in range(4):
        b.set(11, c, COLOR_BLUE)
    return b


# ============================
# STABLE
# ============================


def test_stable_returns_confirmed_board() -> None:
    gen = InferenceBoardGenerator()
    board = _board_with_4connect_red()
    ctx = StateContext(state=BoardState.STABLE, confirmed_board=board)
    out = gen.generate(ctx, time_sec=10.0)
    assert out == board


def test_menu_returns_none() -> None:
    gen = InferenceBoardGenerator()
    ctx = StateContext(state=BoardState.MENU)
    assert gen.generate(ctx, time_sec=0.0) is None


# ============================
# Action states (hold confirmed_board)
# ============================


def test_tsumo_fall_holds_last_stable() -> None:
    gen = InferenceBoardGenerator()
    board = _board_with_4connect_red()
    ctx = StateContext(
        state=BoardState.TSUMO_FALL, confirmed_board=board,
    )
    assert gen.generate(ctx, time_sec=1.0) == board


def test_ojama_fall_holds_last_stable() -> None:
    gen = InferenceBoardGenerator()
    board = _board_with_4connect_red()
    ctx = StateContext(
        state=BoardState.OJAMA_FALL, confirmed_board=board,
    )
    assert gen.generate(ctx, time_sec=2.0) == board


def test_effect_holds_last_stable() -> None:
    gen = InferenceBoardGenerator()
    board = _board_with_4connect_red()
    ctx = StateContext(
        state=BoardState.EFFECT, confirmed_board=board,
    )
    assert gen.generate(ctx, time_sec=3.0) == board


# ============================
# CHAIN state
# ============================


def test_chain_invokes_simulator_at_start() -> None:
    gen = InferenceBoardGenerator()
    board = _board_with_4connect_red()
    ev = _StubChainEvent(trigger_sec=10.0, end_sec=11.0)
    ctx = StateContext(
        state=BoardState.CHAIN,
        confirmed_board=board,
        time_sec=10.0,
    )
    out = gen.generate(ctx, chain_event=ev, time_sec=10.0)
    # 連鎖発火直後は 1 段目の board_after が返る
    assert out is not None
    assert gen.chain_playback is not None
    assert gen.chain_playback.chain_result.chain_count == 1


def test_chain_returns_final_board_at_end() -> None:
    gen = InferenceBoardGenerator()
    board = _board_with_4connect_red()
    ev = _StubChainEvent(trigger_sec=10.0, end_sec=11.0)
    ctx = StateContext(
        state=BoardState.CHAIN, confirmed_board=board,
    )
    # progress=1.0 (= end_sec 到達) → final_board
    out = gen.generate(ctx, chain_event=ev, time_sec=11.0)
    pb = gen.chain_playback
    assert pb is not None
    assert out == pb.chain_result.final_board


def test_chain_progresses_through_steps() -> None:
    """2 連鎖盤面で progress=0.0/0.5/1.0 で異なる盤面が返る."""
    gen = InferenceBoardGenerator()
    board = _board_with_2chain()
    ev = _StubChainEvent(trigger_sec=10.0, end_sec=12.0)
    ctx = StateContext(
        state=BoardState.CHAIN, confirmed_board=board,
    )
    # progress=0 → 1 段目消去後
    b0 = gen.generate(ctx, chain_event=ev, time_sec=10.0)
    # progress=0.6 → 2 段目消去後 (idx=1)
    b1 = gen.generate(ctx, chain_event=ev, time_sec=11.2)
    # progress=1.0 → final_board
    b2 = gen.generate(ctx, chain_event=ev, time_sec=12.0)
    assert b0 is not None
    assert b1 is not None
    assert b2 is not None
    # 連鎖が進むほど puyo 数が減る
    assert b0.count_puyos() >= b1.count_puyos()
    assert b1.count_puyos() >= b2.count_puyos()


def test_chain_playback_clears_when_back_to_stable() -> None:
    gen = InferenceBoardGenerator()
    board = _board_with_4connect_red()
    ev = _StubChainEvent(trigger_sec=10.0, end_sec=11.0)
    ctx_chain = StateContext(
        state=BoardState.CHAIN, confirmed_board=board,
    )
    gen.generate(ctx_chain, chain_event=ev, time_sec=10.0)
    assert gen.chain_playback is not None

    # STABLE に戻る → playback クリア
    ctx_stable = StateContext(
        state=BoardState.STABLE, confirmed_board=board,
    )
    gen.generate(ctx_stable, time_sec=12.0)
    assert gen.chain_playback is None


def test_chain_no_simulate_for_non_eraseable_baseline() -> None:
    """4 連結なし → simulate 結果が chain_count=0、playback 不発."""
    gen = InferenceBoardGenerator()
    board = _empty_board()
    ev = _StubChainEvent(trigger_sec=10.0, end_sec=11.0)
    ctx = StateContext(
        state=BoardState.CHAIN, confirmed_board=board,
    )
    out = gen.generate(ctx, chain_event=ev, time_sec=10.0)
    # playback 起動失敗 → confirmed_board (= empty) を返す
    assert out == board
    assert gen.chain_playback is None


def test_reset_clears_playback() -> None:
    gen = InferenceBoardGenerator()
    board = _board_with_4connect_red()
    ev = _StubChainEvent(trigger_sec=10.0, end_sec=11.0)
    ctx = StateContext(
        state=BoardState.CHAIN, confirmed_board=board,
    )
    gen.generate(ctx, chain_event=ev, time_sec=10.0)
    assert gen.chain_playback is not None
    gen.reset()
    assert gen.chain_playback is None
