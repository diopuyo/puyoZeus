"""Phase G (C-1) 確率版 indicator のテスト.

- BaseIndicator.compute_probabilistic デフォルト実装は MLE 委譲
- MainChainMaturity / DeathRisk / Harassment / Extension の override
- IndicatorCalculator.compute_all_probabilistic
"""
from __future__ import annotations

import pytest

from src.board import (
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
    BaseIndicator,
    DeathRiskIndicator,
    EXTENSION_PROB_SAMPLES,
    ExtensionPotentialIndicator,
    FieldEfficiencyIndicator,
    HarassmentResistanceIndicator,
    INDICATOR_DEATH_RISK,
    INDICATOR_EXTENSION,
    INDICATOR_HARASSMENT,
    INDICATOR_MAIN_CHAIN,
    IndicatorCalculator,
    IndicatorSet,
    MainChainMaturityIndicator,
)
from src.probabilistic_board import ProbabilisticBoard


def _build_4chain_board() -> Board:
    """4 個の赤を下段に配置 (1 連鎖盤面)."""
    b = Board()
    b.set(12, 0, COLOR_RED)
    b.set(12, 1, COLOR_RED)
    b.set(11, 0, COLOR_RED)
    b.set(11, 1, COLOR_RED)
    return b


def _build_dense_top_board() -> Board:
    """致命列 (col=2) 上方を埋める盤面 (DeathRisk 大)."""
    b = Board()
    for row in range(2, BOARD_ROWS):
        b.set(row, 2, COLOR_YELLOW)
    return b


# ============================
# BaseIndicator デフォルト動作
# ============================


def test_base_default_delegates_to_mle() -> None:
    """確定盤面なら compute_probabilistic == compute (FieldEfficiency で確認).

    FieldEfficiencyIndicator は override していないので
    BaseIndicator.compute_probabilistic デフォルト経路を辿る。
    """
    board = _build_4chain_board()
    pb = ProbabilisticBoard.from_board(board)
    ind = FieldEfficiencyIndicator()
    sim = ChainSimulator()
    a = ind.compute(board, simulator=sim)
    b = ind.compute_probabilistic(pb, simulator=sim)
    assert a.score == b.score


# ============================
# MainChainMaturity
# ============================


def test_main_chain_certain_matches() -> None:
    """確定盤面で確率版 MainChain ≈ 通常版."""
    board = _build_4chain_board()
    pb = ProbabilisticBoard.from_board(board)
    ind = MainChainMaturityIndicator()
    a = ind.compute(board)
    b = ind.compute_probabilistic(pb, n_samples=4)
    # 確定盤面ではサンプル全て同じ → 平均=確定値
    assert a.score == b.score
    assert b.detail["probabilistic"] is True
    assert b.detail["n_samples"] == 4


def test_main_chain_quantum_distributes() -> None:
    """量子セル含むと chain_count が変動し std > 0 が起こりうる."""
    pb = ProbabilisticBoard()
    pb.set_certain(12, 0, COLOR_RED)
    pb.set_certain(12, 1, COLOR_RED)
    pb.set_certain(11, 0, COLOR_RED)
    pb.set_distribution(11, 1, {COLOR_RED: 0.5, COLOR_BLUE: 0.5})
    ind = MainChainMaturityIndicator()
    res = ind.compute_probabilistic(pb, n_samples=20)
    assert res.detail["std_chain_count"] >= 0.0
    # サンプル平均は 0..1 のどこか (4 個揃えば 1 連鎖、揃わなければ 0)
    assert 0.0 <= res.detail["mean_chain_count"] <= 1.0


# ============================
# DeathRisk
# ============================


def test_death_risk_certain_matches() -> None:
    """確定盤面で確率版 ≈ 通常版."""
    board = _build_dense_top_board()
    pb = ProbabilisticBoard.from_board(board)
    ind = DeathRiskIndicator()
    a = ind.compute(board)
    b = ind.compute_probabilistic(pb)
    # 値が極めて近い (確率版は連続値)
    assert abs(a.score - b.score) < 1e-6
    assert b.detail["probabilistic"] is True


def test_death_risk_quantum_uses_expected_height() -> None:
    """量子セルがあると期待高さ (連続値) が反映される."""
    pb = ProbabilisticBoard()
    # col=2 の上方を半分の確率で puyo
    for row in range(5, BOARD_ROWS):
        pb.set_distribution(
            row, 2, {COLOR_EMPTY: 0.5, COLOR_RED: 0.5},
        )
    ind = DeathRiskIndicator()
    res = ind.compute_probabilistic(pb)
    # 期待高さ ≈ 8 * 0.5 = 4.0 (col=2)
    expected_h = res.detail["col_expected_heights"][2]
    assert 3.5 < expected_h < 4.5


# ============================
# Harassment
# ============================


def test_harassment_certain_matches_close() -> None:
    """確定盤面なら確率版 ≈ 通常版 (サンプル全て同じ)."""
    board = _build_4chain_board()
    pb = ProbabilisticBoard.from_board(board)
    ind = HarassmentResistanceIndicator()
    a = ind.compute(board)
    b = ind.compute_probabilistic(pb, n_samples=3)
    assert abs(a.score - b.score) < 1e-9


def test_harassment_per_sample_detail() -> None:
    """detail に per-sample sub-scores が含まれる."""
    pb = ProbabilisticBoard.from_board(_build_4chain_board())
    ind = HarassmentResistanceIndicator()
    res = ind.compute_probabilistic(pb, n_samples=4, incoming_ojama=12)
    assert "per_sample_scores" in res.detail
    assert len(res.detail["per_sample_scores"]) == 4
    assert res.detail["incoming_ojama"] == 12
    assert res.detail["probabilistic"] is True


# ============================
# Extension
# ============================


def test_extension_certain_matches_close() -> None:
    """確定盤面で確率版 ≈ 通常版 (誤差小)."""
    board = _build_4chain_board()
    pb = ProbabilisticBoard.from_board(board)
    ind = ExtensionPotentialIndicator()
    a = ind.compute(board)
    b = ind.compute_probabilistic(pb, n_samples=EXTENSION_PROB_SAMPLES)
    # 確定盤面なので一致するはず
    assert abs(a.score - b.score) < 1e-9


def test_extension_default_n_samples() -> None:
    """n_samples 省略時に EXTENSION_PROB_SAMPLES が使われる."""
    pb = ProbabilisticBoard.from_board(_build_4chain_board())
    ind = ExtensionPotentialIndicator()
    res = ind.compute_probabilistic(pb)
    assert res.detail["n_samples"] == EXTENSION_PROB_SAMPLES


# ============================
# IndicatorCalculator
# ============================


def test_calculator_compute_all_probabilistic_returns_set() -> None:
    """compute_all_probabilistic は IndicatorSet を返す."""
    board = _build_4chain_board()
    pb = ProbabilisticBoard.from_board(board)
    calc = IndicatorCalculator()
    result = calc.compute_all_probabilistic(pb, n_samples=3)
    assert isinstance(result, IndicatorSet)
    # 主要 indicator が結果辞書にある
    for name in (
        INDICATOR_MAIN_CHAIN, INDICATOR_DEATH_RISK,
        INDICATOR_HARASSMENT, INDICATOR_EXTENSION,
    ):
        assert name in result.results


def test_calculator_compute_all_probabilistic_certain_close_to_normal() -> None:
    """確定盤面で確率版 IndicatorSet ≈ 通常版 (主要 4 指標)."""
    board = _build_4chain_board()
    pb = ProbabilisticBoard.from_board(board)
    calc = IndicatorCalculator()
    normal = calc.compute_all(board)
    prob = calc.compute_all_probabilistic(pb, n_samples=3)
    for name in (
        INDICATOR_MAIN_CHAIN, INDICATOR_DEATH_RISK,
        INDICATOR_HARASSMENT, INDICATOR_EXTENSION,
    ):
        delta = abs(normal.score_of(name) - prob.score_of(name))
        # 確率版は浮動小数誤差程度に近い
        assert delta < 1e-6, f"{name} delta={delta}"


def test_has_prob_override_detection() -> None:
    """_has_prob_override が override 済の indicator を識別する."""
    calc = IndicatorCalculator()
    assert calc._has_prob_override(MainChainMaturityIndicator())
    assert calc._has_prob_override(DeathRiskIndicator())
    assert calc._has_prob_override(HarassmentResistanceIndicator())
    assert calc._has_prob_override(ExtensionPotentialIndicator())
    # FieldEfficiencyIndicator は未 override (デフォルト経路)
    assert not calc._has_prob_override(FieldEfficiencyIndicator())


def test_calculator_handles_quantum_board() -> None:
    """量子セル含む盤面でも正常終了する (NaN/例外が出ない)."""
    pb = ProbabilisticBoard()
    pb.set_certain(12, 0, COLOR_RED)
    pb.set_distribution(11, 0, {COLOR_RED: 0.5, COLOR_BLUE: 0.5})
    pb.set_distribution(10, 0, {COLOR_GREEN: 0.4, COLOR_YELLOW: 0.6})
    calc = IndicatorCalculator()
    result = calc.compute_all_probabilistic(pb, n_samples=4)
    # 全 score が [0, 1] 範囲内
    for name, ind_result in result.results.items():
        assert 0.0 <= ind_result.score <= 1.0, f"{name} out of range"
