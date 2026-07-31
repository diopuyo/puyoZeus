"""src/chain_bitboard.py の近似得点 (score_approx) の回帰テスト。

#24 打ち合い計測器 Step2 (2026-07-29) で追加した
simulate_batch_with_approx_score / simulate_single_with_approx_score の検証。

検証方針:
    1. 全消去グループが size==4 (連結ボーナス0) のケースでは、近似が
       calculate_chain_score (既存・厳密) と完全一致することを確認する
       (連結ボーナスのみ近似しているため、この条件下では誤差ゼロのはず)。
    2. size>=5 の連結を含むケースでは、近似が厳密値以下 (過小評価) に
       なることを確認する (docstring で明記した近似の限界)。
    3. simulate_batch (chain_count/total_erased/total_ojama) との整合性
       (approx版でもこれらの値が既存 simulate_batch と一致すること)。
"""
from __future__ import annotations

import numpy as np

from src.board import Board, COLOR_RED, COLOR_BLUE, COLOR_GREEN, COLOR_OJAMA
from src.chain import ChainSimulator
from src.chain_bitboard import (
    batch_from_boards,
    simulate_batch,
    simulate_batch_with_approx_score,
    simulate_single_with_approx_score,
)
from src.scoring import calculate_chain_score


def _sim() -> ChainSimulator:
    return ChainSimulator()


def test_single_four_group_exact_match() -> None:
    """4連結1グループのみ (連結ボーナス0) は近似=厳密で完全一致する。"""
    board = Board()
    for row in range(4):
        board.set(row, 0, COLOR_RED)
    expected = calculate_chain_score(_sim().simulate(board)).total_score
    got = simulate_single_with_approx_score(board)
    assert got.score_approx == expected


def test_two_step_chain_all_four_groups_exact_match() -> None:
    """2連鎖・各ステップ4連結1グループのみのケースで完全一致する。"""
    board = Board()
    board.set(8, 0, COLOR_BLUE)
    for row in range(9, 13):
        board.set(row, 0, COLOR_RED)
    for row in range(10, 13):
        board.set(row, 1, COLOR_BLUE)
    result = _sim().simulate(board)
    assert result.chain_count == 2, "テスト構築ミス: 2連鎖になっていない"
    expected = calculate_chain_score(result).total_score
    got = simulate_single_with_approx_score(board)
    assert got.score_approx == expected


def test_multi_color_same_step_four_groups_exact_match() -> None:
    """同一ステップで2色同時消去 (各4連結) は色数ボーナスも含めて厳密一致する。"""
    board = Board()
    for row in range(4):
        board.set(row, 0, COLOR_RED)
    for row in range(4):
        board.set(row, 5, COLOR_BLUE)
    result = _sim().simulate(board)
    assert result.chain_count == 1
    expected = calculate_chain_score(result).total_score
    got = simulate_single_with_approx_score(board)
    assert got.score_approx == expected


def test_five_connected_group_approx_is_underestimate() -> None:
    """5連結 (連結ボーナス+2) を含むケースでは近似が厳密値以下になる (過小評価)。"""
    board = Board()
    board.set(12, 0, COLOR_GREEN)
    board.set(12, 1, COLOR_GREEN)
    board.set(12, 2, COLOR_GREEN)
    board.set(11, 2, COLOR_GREEN)
    board.set(11, 1, COLOR_GREEN)
    result = _sim().simulate(board)
    assert result.chain_count == 1
    expected = calculate_chain_score(result).total_score
    got = simulate_single_with_approx_score(board)
    assert got.score_approx <= expected
    assert got.score_approx < expected, "5連結ケースで差が出ない=テスト構築ミスの疑い"


def test_approx_batch_matches_plain_batch_on_aggregate_fields() -> None:
    """近似得点版でも chain_count/total_erased/total_ojama は simulate_batch と完全一致する。"""
    boards: list[Board] = []
    b1 = Board()
    for row in range(4):
        b1.set(row, 0, COLOR_RED)
    boards.append(b1)

    b2 = Board()
    b2.set(9, 1, COLOR_BLUE)
    b2.set(9, 2, COLOR_BLUE)
    b2.set(9, 3, COLOR_BLUE)
    b2.set(8, 2, COLOR_BLUE)
    b2.set(10, 1, COLOR_OJAMA)
    b2.set(10, 2, COLOR_OJAMA)
    b2.set(10, 3, COLOR_OJAMA)
    boards.append(b2)

    boards.append(Board())  # 空盤面 (連鎖なし)

    planes = batch_from_boards(boards)
    plain = simulate_batch(planes)
    approx = simulate_batch_with_approx_score(planes)
    for p, a in zip(plain, approx):
        assert p.chain_count == a.chain_count
        assert p.total_erased == a.total_erased
        assert p.total_ojama == a.total_ojama
        assert np.array_equal(p.final_planes[COLOR_RED], a.final_planes[COLOR_RED])


def test_empty_board_zero_score() -> None:
    """空盤面 (連鎖なし) の近似得点は0。"""
    got = simulate_single_with_approx_score(Board())
    assert got.score_approx == 0
    assert got.chain_count == 0
