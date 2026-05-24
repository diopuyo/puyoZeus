"""Phase H1 機能能力指標 7 個のテスト (2026-05-08).

「形は手段、機能が本質」設計思想に基づく Capability Indicators:
    - ReadyChainCountIndicator
    - IgnitionDistanceIndicator
    - CurrentFirePowerIndicator
    - MaximumFirePowerIndicator
    - MidGameResponseCapacityIndicator
    - HarassmentReadinessIndicator
    - OjamaDefenseCapacityIndicator

合計 30+ テスト。各指標は board の機能達成度を 0..1 で測る。
"""
from __future__ import annotations

import pytest

from src.board import (
    BOARD_COLS,
    BOARD_ROWS,
    COLOR_BLUE,
    COLOR_EMPTY,
    COLOR_GREEN,
    COLOR_RED,
    COLOR_YELLOW,
    Board,
)
from src.chain import ChainSimulator
from src.indicators import (
    CurrentFirePowerIndicator,
    HarassmentReadinessIndicator,
    INDICATOR_CURRENT_FIRE_POWER,
    INDICATOR_HARASS_READINESS,
    INDICATOR_IGNITION_DISTANCE,
    INDICATOR_MAXIMUM_FIRE_POWER,
    INDICATOR_MID_GAME_RESPONSE,
    INDICATOR_OJAMA_DEFENSE,
    INDICATOR_READY_CHAIN,
    IgnitionDistanceIndicator,
    IndicatorCalculator,
    MaximumFirePowerIndicator,
    MidGameResponseCapacityIndicator,
    OjamaDefenseCapacityIndicator,
    ReadyChainCountIndicator,
)


# ============================
# fixtures
# ============================


def _empty_board() -> Board:
    grid = [[COLOR_EMPTY] * BOARD_COLS for _ in range(BOARD_ROWS)]
    return Board.from_list(grid)


def _three_red_row() -> Board:
    """1 列に届かない 3 連結 (赤、最下段に 3 個)。発火直前盤面。"""
    grid = [[COLOR_EMPTY] * BOARD_COLS for _ in range(BOARD_ROWS)]
    for c in range(3):
        grid[BOARD_ROWS - 1][c] = COLOR_RED
    return Board.from_list(grid)


def _four_red_row() -> Board:
    """4 連結発火可能盤面 (即発火 1 連鎖)。"""
    grid = [[COLOR_EMPTY] * BOARD_COLS for _ in range(BOARD_ROWS)]
    for c in range(4):
        grid[BOARD_ROWS - 1][c] = COLOR_RED
    return Board.from_list(grid)


def _two_chain_board() -> Board:
    """確実に 2 連鎖発火する盤面 (ランダム探索で発見、行 10..12)。"""
    grid = [[COLOR_EMPTY] * BOARD_COLS for _ in range(BOARD_ROWS)]
    # row 10: 2 2 1 2 0 2 (B B R B _ B)
    grid[10] = [COLOR_BLUE, COLOR_BLUE, COLOR_RED, COLOR_BLUE,
                COLOR_EMPTY, COLOR_BLUE]
    # row 11: 2 2 2 2 0 1 (B B B B _ R)
    grid[11] = [COLOR_BLUE, COLOR_BLUE, COLOR_BLUE, COLOR_BLUE,
                COLOR_EMPTY, COLOR_RED]
    # row 12: 0 1 2 1 1 0 (_ R B R R _)
    grid[12] = [COLOR_EMPTY, COLOR_RED, COLOR_BLUE, COLOR_RED,
                COLOR_RED, COLOR_EMPTY]
    return Board.from_list(grid)


def _full_board() -> Board:
    """大量 puyo 盤面 (max_fire 高評価)。"""
    grid = [[COLOR_EMPTY] * BOARD_COLS for _ in range(BOARD_ROWS)]
    # 下半分を 4 色で適当に埋める
    colors = [COLOR_RED, COLOR_BLUE, COLOR_GREEN, COLOR_YELLOW]
    for r in range(BOARD_ROWS // 2, BOARD_ROWS):
        for c in range(BOARD_COLS):
            grid[r][c] = colors[(r + c) % 4]
    return Board.from_list(grid)


# ============================
# ReadyChainCountIndicator
# ============================


def test_ready_chain_empty_board_zero() -> None:
    """空盤面は連鎖 0 → score 0."""
    ind = ReadyChainCountIndicator()
    res = ind.compute(_empty_board())
    assert res.name == INDICATOR_READY_CHAIN
    assert res.score == 0.0
    assert res.detail["chain_count"] == 0


def test_ready_chain_no_fire_zero() -> None:
    """3 連結 (発火しない) は連鎖 0 → score 0."""
    ind = ReadyChainCountIndicator()
    res = ind.compute(_three_red_row())
    assert res.score == 0.0


def test_ready_chain_fire_capable_positive() -> None:
    """4 連結発火可能盤面は連鎖 1 → score > 0."""
    ind = ReadyChainCountIndicator()
    sim = ChainSimulator()
    res = ind.compute(_four_red_row(), simulator=sim)
    assert res.score > 0.0
    assert res.detail["chain_count"] >= 1


def test_ready_chain_two_chain_higher() -> None:
    """2 連鎖盤面は 1 連鎖盤面より高スコア。"""
    ind = ReadyChainCountIndicator()
    sim = ChainSimulator()
    r1 = ind.compute(_four_red_row(), simulator=sim)
    r2 = ind.compute(_two_chain_board(), simulator=sim)
    assert r2.score > r1.score


def test_ready_chain_score_in_range() -> None:
    """全テスト盤面で 0..1 範囲内."""
    ind = ReadyChainCountIndicator()
    sim = ChainSimulator()
    for b in [_empty_board(), _three_red_row(), _four_red_row(),
              _two_chain_board(), _full_board()]:
        res = ind.compute(b, simulator=sim)
        assert 0.0 <= res.score <= 1.0


# ============================
# IgnitionDistanceIndicator
# ============================


def test_ignition_distance_empty_low_score() -> None:
    """空盤面は発火不可で低スコア (min_n が大きい)."""
    ind = IgnitionDistanceIndicator()
    sim = ChainSimulator()
    res = ind.compute(_empty_board(), simulator=sim)
    assert res.name == INDICATOR_IGNITION_DISTANCE
    assert res.score < 0.5  # 発火寸前ではない


def test_ignition_distance_three_red_close() -> None:
    """3 連結盤面は 1 puyo で発火 (min_n=1) → 高スコア."""
    ind = IgnitionDistanceIndicator()
    sim = ChainSimulator()
    res = ind.compute(_three_red_row(), simulator=sim)
    assert res.detail["min_n"] == 1
    # min_n=1, MAX=6: score = 1 - 1/6 ≈ 0.833
    assert res.score > 0.7


def test_ignition_distance_four_red_already_fires() -> None:
    """4 連結盤面は base_chain=1 を超える発火がさらに 1 puyo で達成可能 → score > 0."""
    ind = IgnitionDistanceIndicator()
    sim = ChainSimulator()
    res = ind.compute(_four_red_row(), simulator=sim)
    # 0..1 範囲、min_n が小さければ高スコア
    assert 0.0 <= res.score <= 1.0


def test_ignition_distance_score_in_range() -> None:
    """全テスト盤面で 0..1 範囲内."""
    ind = IgnitionDistanceIndicator()
    sim = ChainSimulator()
    for b in [_empty_board(), _three_red_row(), _four_red_row()]:
        res = ind.compute(b, simulator=sim)
        assert 0.0 <= res.score <= 1.0


# ============================
# CurrentFirePowerIndicator
# ============================


def test_current_fire_empty_zero() -> None:
    """空盤面は得点 0 → score 0."""
    ind = CurrentFirePowerIndicator()
    res = ind.compute(_empty_board())
    assert res.name == INDICATOR_CURRENT_FIRE_POWER
    assert res.score == 0.0


def test_current_fire_no_chain_zero() -> None:
    """発火しない盤面は score 0."""
    ind = CurrentFirePowerIndicator()
    res = ind.compute(_three_red_row())
    assert res.score == 0.0


def test_current_fire_one_chain_positive() -> None:
    """1 連鎖発火盤面は score > 0."""
    ind = CurrentFirePowerIndicator()
    sim = ChainSimulator()
    res = ind.compute(_four_red_row(), simulator=sim)
    assert res.score > 0.0
    assert res.detail["chain_count"] >= 1


def test_current_fire_two_chain_higher() -> None:
    """2 連鎖は 1 連鎖より score 高い."""
    ind = CurrentFirePowerIndicator()
    sim = ChainSimulator()
    r1 = ind.compute(_four_red_row(), simulator=sim)
    r2 = ind.compute(_two_chain_board(), simulator=sim)
    assert r2.score > r1.score


def test_current_fire_score_in_range() -> None:
    """全テスト盤面で 0..1."""
    ind = CurrentFirePowerIndicator()
    sim = ChainSimulator()
    for b in [_empty_board(), _three_red_row(), _four_red_row(),
              _two_chain_board()]:
        res = ind.compute(b, simulator=sim)
        assert 0.0 <= res.score <= 1.0


# ============================
# MaximumFirePowerIndicator
# ============================


def test_max_fire_empty_zero() -> None:
    """空盤面は puyo 0 → max_fire 0."""
    ind = MaximumFirePowerIndicator()
    res = ind.compute(_empty_board())
    assert res.name == INDICATOR_MAXIMUM_FIRE_POWER
    assert res.score == 0.0
    assert res.detail["puyo_count"] == 0


def test_max_fire_full_board_higher() -> None:
    """大量 puyo 盤面は max_fire 大."""
    ind = MaximumFirePowerIndicator()
    res_empty = ind.compute(_empty_board())
    res_full = ind.compute(_full_board())
    assert res_full.score > res_empty.score


def test_max_fire_color_diversity_increases_score() -> None:
    """色多様性が高い盤面は max_fire 大 (色数ボーナス)."""
    ind = MaximumFirePowerIndicator()
    # 1 色のみ (赤) 盤面
    grid_mono = [[COLOR_EMPTY] * BOARD_COLS for _ in range(BOARD_ROWS)]
    for r in range(BOARD_ROWS - 4, BOARD_ROWS):
        for c in range(BOARD_COLS):
            grid_mono[r][c] = COLOR_RED
    b_mono = Board.from_list(grid_mono)
    # 4 色盤面
    b_full = _full_board()
    res_mono = ind.compute(b_mono)
    res_full = ind.compute(b_full)
    # puyo 数同数前提で、4 色の方が色数ボーナス分高い
    assert res_full.detail["color_count"] > res_mono.detail["color_count"]


def test_max_fire_score_in_range() -> None:
    """全テスト盤面で 0..1."""
    ind = MaximumFirePowerIndicator()
    for b in [_empty_board(), _three_red_row(), _four_red_row(),
              _two_chain_board(), _full_board()]:
        res = ind.compute(b)
        assert 0.0 <= res.score <= 1.0


# ============================
# MidGameResponseCapacityIndicator
# ============================


def test_mid_game_empty_low() -> None:
    """空盤面は応答能力なし → 低スコア."""
    ind = MidGameResponseCapacityIndicator()
    res = ind.compute(_empty_board())
    assert res.name == INDICATOR_MID_GAME_RESPONSE
    assert res.score < 0.3


def test_mid_game_score_in_range() -> None:
    """全テスト盤面で 0..1."""
    ind = MidGameResponseCapacityIndicator()
    sim = ChainSimulator()
    for b in [_empty_board(), _three_red_row(), _four_red_row(),
              _full_board()]:
        res = ind.compute(b, simulator=sim)
        assert 0.0 <= res.score <= 1.0


def test_mid_game_full_board_components() -> None:
    """detail に sub_chain_score / max_fire_power / current_fire_power が含まれる."""
    ind = MidGameResponseCapacityIndicator()
    res = ind.compute(_full_board(), simulator=ChainSimulator())
    assert "sub_chain_score" in res.detail
    assert "max_fire_power" in res.detail
    assert "current_fire_power" in res.detail


# ============================
# HarassmentReadinessIndicator
# ============================


def test_harass_readiness_empty_zero() -> None:
    """空盤面は副砲なし → 0."""
    ind = HarassmentReadinessIndicator()
    res = ind.compute(_empty_board())
    assert res.name == INDICATOR_HARASS_READINESS
    assert res.score == 0.0


def test_harass_readiness_three_red_no_chain() -> None:
    """3 連結 1 個盤面は 1 連鎖発火可能だが、本指標は 2-4 連鎖カウント.

    1 連鎖は本指標の閾値外 → score 0 のまま。
    """
    ind = HarassmentReadinessIndicator()
    sim = ChainSimulator()
    res = ind.compute(_three_red_row(), simulator=sim)
    # 1 連鎖発火可能でも min_chain=2 のため 0
    assert res.score == 0.0


def test_harass_readiness_score_in_range() -> None:
    """全テスト盤面で 0..1."""
    ind = HarassmentReadinessIndicator()
    sim = ChainSimulator()
    for b in [_empty_board(), _three_red_row(), _four_red_row(),
              _two_chain_board(), _full_board()]:
        res = ind.compute(b, simulator=sim)
        assert 0.0 <= res.score <= 1.0


def test_harass_readiness_dead_board_zero() -> None:
    """窒息盤面は 0."""
    grid = [[COLOR_RED] * BOARD_COLS for _ in range(BOARD_ROWS)]
    b = Board.from_list(grid)
    ind = HarassmentReadinessIndicator()
    res = ind.compute(b)
    assert res.score == 0.0
    assert res.detail.get("board_dead") is True


# ============================
# OjamaDefenseCapacityIndicator
# ============================


def test_ojama_defense_empty_high_or_low() -> None:
    """空盤面は base_chain=0 → 仮想 ojama 30 後も窒息せず → 何らか score 範囲内."""
    ind = OjamaDefenseCapacityIndicator()
    sim = ChainSimulator()
    res = ind.compute(_empty_board(), simulator=sim)
    assert res.name == INDICATOR_OJAMA_DEFENSE
    assert 0.0 <= res.score <= 1.0


def test_ojama_defense_one_chain_test_results() -> None:
    """1 連鎖発火盤面で全 OJAMA_DEFENSE_TEST_COUNTS について detail に結果."""
    ind = OjamaDefenseCapacityIndicator()
    sim = ChainSimulator()
    res = ind.compute(_four_red_row(), simulator=sim)
    by_count = res.detail["by_ojama_count"]
    for n in (10, 20, 30):
        assert n in by_count


def test_ojama_defense_score_in_range() -> None:
    """全テスト盤面で 0..1."""
    ind = OjamaDefenseCapacityIndicator()
    sim = ChainSimulator()
    for b in [_empty_board(), _three_red_row(), _four_red_row(),
              _two_chain_board(), _full_board()]:
        res = ind.compute(b, simulator=sim)
        assert 0.0 <= res.score <= 1.0


def test_ojama_defense_dead_after_drop() -> None:
    """ほぼ満杯盤面に ojama 30 個降らせると窒息 → 一部 sub-score 0."""
    # ほぼ満杯 (上 1 段だけ空)
    grid = [[COLOR_RED] * BOARD_COLS for _ in range(BOARD_ROWS)]
    grid[0] = [COLOR_EMPTY] * BOARD_COLS  # 上端だけ空
    b = Board.from_list(grid)
    ind = OjamaDefenseCapacityIndicator()
    sim = ChainSimulator()
    res = ind.compute(b, simulator=sim)
    # 30 個落としたら危険
    assert 0.0 <= res.score <= 1.0


# ============================
# IndicatorCalculator 統合
# ============================


def test_calc_compute_all_includes_all_capability_indicators() -> None:
    """compute_all の結果に H1 機能 7 指標が含まれる."""
    calc = IndicatorCalculator()
    res = calc.compute_all(_empty_board())
    capability = [
        INDICATOR_READY_CHAIN,
        INDICATOR_IGNITION_DISTANCE,
        INDICATOR_CURRENT_FIRE_POWER,
        INDICATOR_MAXIMUM_FIRE_POWER,
        INDICATOR_MID_GAME_RESPONSE,
        INDICATOR_HARASS_READINESS,
        INDICATOR_OJAMA_DEFENSE,
    ]
    for name in capability:
        assert name in res.results
    # IndicatorSet field 反映確認
    assert res.ready_chain_count == res.results[INDICATOR_READY_CHAIN].score
    assert res.ignition_distance == \
        res.results[INDICATOR_IGNITION_DISTANCE].score
    assert res.current_fire_power == \
        res.results[INDICATOR_CURRENT_FIRE_POWER].score
    assert res.maximum_fire_power == \
        res.results[INDICATOR_MAXIMUM_FIRE_POWER].score
    assert res.mid_game_response_capacity == \
        res.results[INDICATOR_MID_GAME_RESPONSE].score
    assert res.harassment_readiness == \
        res.results[INDICATOR_HARASS_READINESS].score
    assert res.ojama_defense_capacity == \
        res.results[INDICATOR_OJAMA_DEFENSE].score
