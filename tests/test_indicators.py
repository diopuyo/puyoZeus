"""
indicators.py のテスト

8指標 + IndicatorCalculator の動作を検証する。
"""

from __future__ import annotations

import pytest

from src.board import (
    BOARD_COLS,
    BOARD_ROWS,
    COLOR_BLUE,
    COLOR_EMPTY,
    COLOR_GREEN,
    COLOR_OJAMA,
    COLOR_RED,
    COLOR_YELLOW,
    Board,
)
from src.chain import ChainSimulator
from src.old.indicators import (
    ALL_INDICATOR_NAMES,
    INDICATOR_DEATH_RISK,
    INDICATOR_EXTENSION,
    INDICATOR_FIELD_EFF,
    INDICATOR_HARASSMENT,
    INDICATOR_MAIN_CHAIN,
    INDICATOR_OFFSET,
    INDICATOR_SECOND,
    INDICATOR_SUB_CHAIN,
    MAX_EXPECTED_CHAIN,
    BaseIndicator,
    DeathRiskIndicator,
    ExtensionPotentialIndicator,
    FieldEfficiencyIndicator,
    HarassmentResistanceIndicator,
    IndicatorCalculator,
    IndicatorResult,
    IndicatorSet,
    MainChainMaturityIndicator,
    OffsetPowerIndicator,
    SecondChainPotentialIndicator,
    SubChainQualityIndicator,
)


# ============================
# テスト用ヘルパー
# ============================


def empty_grid() -> list[list[int]]:
    """13×6 の全空グリッド。"""
    return [[COLOR_EMPTY] * BOARD_COLS for _ in range(BOARD_ROWS)]


def board_from_grid(grid: list[list[int]]) -> Board:
    return Board.from_list(grid)


def make_empty_board() -> Board:
    return board_from_grid(empty_grid())


def make_4_chain_board() -> Board:
    """
    4連鎖する盤面を生成する (サンドイッチ階段4連鎖)。

    構造:
        col 0 (rows 5-12): Y Y B B G G R R
        col 1-3 (rows 9-12): Y B G R
    連鎖順序: R → G → B → Y (各ステップで5ぷよ消去)
    """
    grid = empty_grid()
    col0_seq = [
        COLOR_YELLOW, COLOR_YELLOW,
        COLOR_BLUE, COLOR_BLUE,
        COLOR_GREEN, COLOR_GREEN,
        COLOR_RED, COLOR_RED,
    ]
    for i, color in enumerate(col0_seq):
        grid[5 + i][0] = color

    other_seq = [COLOR_YELLOW, COLOR_BLUE, COLOR_GREEN, COLOR_RED]
    for col in (1, 2, 3):
        for i, color in enumerate(other_seq):
            grid[9 + i][col] = color
    return board_from_grid(grid)


def make_single_erase_board() -> Board:
    """1連鎖だけする盤面 (4赤が消える)。"""
    grid = empty_grid()
    grid[12][0] = COLOR_RED
    grid[12][1] = COLOR_RED
    grid[12][2] = COLOR_RED
    grid[12][3] = COLOR_RED
    return board_from_grid(grid)


def make_no_chain_board() -> Board:
    """連鎖しない盤面 (同色3個未満)。"""
    grid = empty_grid()
    grid[12][0] = COLOR_RED
    grid[12][1] = COLOR_BLUE
    grid[12][2] = COLOR_GREEN
    grid[12][3] = COLOR_YELLOW
    return board_from_grid(grid)


def make_dead_board() -> Board:
    """窒息中の盤面 (3列目最上段にぷよ)。"""
    grid = empty_grid()
    grid[0][2] = COLOR_RED
    return board_from_grid(grid)


# ============================
# IndicatorResult / IndicatorSet
# ============================


class TestIndicatorResult:
    def test_basic_construction(self):
        r = IndicatorResult(name="test", score=0.5, raw_value=1.0)
        assert r.name == "test"
        assert r.score == 0.5
        assert r.raw_value == 1.0
        assert r.detail == {}

    def test_detail_mutable_default(self):
        r1 = IndicatorResult(name="a", score=0.1, raw_value=0.0)
        r2 = IndicatorResult(name="b", score=0.2, raw_value=0.0)
        r1.detail["x"] = 1
        assert "x" not in r2.detail


class TestIndicatorSet:
    def test_get_and_score_of(self):
        r = IndicatorResult(name=INDICATOR_MAIN_CHAIN, score=0.3, raw_value=4.0)
        s = IndicatorSet(results={INDICATOR_MAIN_CHAIN: r})
        assert s.get(INDICATOR_MAIN_CHAIN) is r
        assert s.score_of(INDICATOR_MAIN_CHAIN) == 0.3

    def test_to_dict_structure(self):
        r = IndicatorResult(
            name=INDICATOR_MAIN_CHAIN,
            score=0.3,
            raw_value=4.0,
            detail={"k": "v"},
        )
        s = IndicatorSet(results={INDICATOR_MAIN_CHAIN: r})
        d = s.to_dict()
        assert d[INDICATOR_MAIN_CHAIN]["score"] == 0.3
        assert d[INDICATOR_MAIN_CHAIN]["raw_value"] == 4.0
        assert d[INDICATOR_MAIN_CHAIN]["detail"]["k"] == "v"


# ============================
# BaseIndicator
# ============================


class TestBaseIndicator:
    def test_clamp_in_range(self):
        assert BaseIndicator._clamp(0.5) == 0.5

    def test_clamp_below_zero(self):
        assert BaseIndicator._clamp(-0.1) == 0.0

    def test_clamp_above_one(self):
        assert BaseIndicator._clamp(1.5) == 1.0

    def test_cannot_instantiate_abstract(self):
        with pytest.raises(TypeError):
            BaseIndicator()  # type: ignore[abstract]


# ============================
# FieldEfficiencyIndicator
# ============================


class TestFieldEfficiencyIndicator:
    def test_name(self):
        assert FieldEfficiencyIndicator().name == INDICATOR_FIELD_EFF

    def test_empty_board_zero_score(self):
        result = FieldEfficiencyIndicator().compute(make_empty_board())
        assert result.score == 0.0
        assert result.detail["normal_puyos"] == 0

    def test_full_participation(self):
        # 4赤のみ=全部消える
        result = FieldEfficiencyIndicator().compute(make_single_erase_board())
        assert result.score == 1.0
        assert result.detail["participating"] == 4

    def test_no_participation(self):
        # 連鎖しない盤面=参加0
        result = FieldEfficiencyIndicator().compute(make_no_chain_board())
        assert result.score == 0.0
        assert result.detail["participating"] == 0

    def test_ojama_excluded_from_denominator(self):
        grid = empty_grid()
        grid[12][0] = COLOR_OJAMA
        grid[12][1] = COLOR_OJAMA
        board = board_from_grid(grid)
        result = FieldEfficiencyIndicator().compute(board)
        assert result.detail["ojama_puyos"] == 2
        assert result.detail["normal_puyos"] == 0


# ============================
# DeathRiskIndicator
# ============================


class TestDeathRiskIndicator:
    def test_name(self):
        assert DeathRiskIndicator().name == INDICATOR_DEATH_RISK

    def test_empty_board_zero_risk(self):
        result = DeathRiskIndicator().compute(make_empty_board())
        assert result.score == 0.0

    def test_dead_board_max_risk(self):
        result = DeathRiskIndicator().compute(make_dead_board())
        assert result.score == 1.0
        assert result.detail["is_dead"] is True

    def test_death_col_weighted_heaviest(self):
        # 致命列のみ高い盤面 vs 端列のみ高い盤面で比較
        g1 = empty_grid()
        for r in range(6, 13):
            g1[r][2] = COLOR_RED  # 致命列に7個
        g2 = empty_grid()
        for r in range(6, 13):
            g2[r][5] = COLOR_RED  # 端列に7個
        score1 = DeathRiskIndicator().compute(board_from_grid(g1)).score
        score2 = DeathRiskIndicator().compute(board_from_grid(g2)).score
        assert score1 > score2

    def test_raw_value_is_weighted_average(self):
        result = DeathRiskIndicator().compute(make_empty_board())
        assert result.raw_value == 0.0


# ============================
# MainChainMaturityIndicator
# ============================


class TestMainChainMaturityIndicator:
    def test_name(self):
        assert MainChainMaturityIndicator().name == INDICATOR_MAIN_CHAIN

    def test_empty_board_zero(self):
        result = MainChainMaturityIndicator().compute(make_empty_board())
        assert result.score == 0.0
        assert result.detail["chain_count"] == 0

    def test_single_erase(self):
        result = MainChainMaturityIndicator().compute(make_single_erase_board())
        assert result.detail["chain_count"] == 1
        assert result.score == pytest.approx(1 / MAX_EXPECTED_CHAIN)

    def test_4_chain(self):
        result = MainChainMaturityIndicator().compute(make_4_chain_board())
        assert result.detail["chain_count"] == 4
        assert result.score == pytest.approx(4 / MAX_EXPECTED_CHAIN)

    def test_score_clamped_to_1(self):
        # 超長連鎖は1.0で頭打ち (ここではシンプルチェックのみ)
        result = MainChainMaturityIndicator().compute(make_4_chain_board())
        assert result.score <= 1.0


# ============================
# OffsetPowerIndicator
# ============================


class TestOffsetPowerIndicator:
    def test_name(self):
        assert OffsetPowerIndicator().name == INDICATOR_OFFSET

    def test_empty_board_zero(self):
        result = OffsetPowerIndicator().compute(make_empty_board())
        assert result.score == 0.0
        assert result.detail["estimated_ojama"] == 0

    def test_single_erase_positive(self):
        result = OffsetPowerIndicator().compute(make_single_erase_board())
        assert result.score > 0.0

    def test_longer_chain_produces_more_offset(self):
        s1 = OffsetPowerIndicator().compute(make_single_erase_board())
        s4 = OffsetPowerIndicator().compute(make_4_chain_board())
        assert s4.raw_value > s1.raw_value

    def test_score_in_range(self):
        result = OffsetPowerIndicator().compute(make_4_chain_board())
        assert 0.0 <= result.score <= 1.0


# ============================
# HarassmentResistanceIndicator
# ============================


class TestHarassmentResistanceIndicator:
    def test_name(self):
        assert HarassmentResistanceIndicator().name == INDICATOR_HARASSMENT

    def test_empty_board_survives(self):
        # 空盤面はおじゃま落下しても窒息しない
        result = HarassmentResistanceIndicator().compute(make_empty_board())
        assert result.score >= 0.0
        assert "survival_by_count" in result.detail

    def test_detail_has_all_steps(self):
        result = HarassmentResistanceIndicator().compute(make_4_chain_board())
        detail = result.detail["survival_by_count"]
        # 10,15,20,25,30 の5段階
        assert set(detail.keys()) == {10, 15, 20, 25, 30}

    def test_nearly_dead_board_low_resistance(self):
        # 致命列をほぼ埋めた盤面はおじゃまで窒息しやすい
        grid = empty_grid()
        for r in range(1, 13):
            grid[r][2] = COLOR_RED
        board = board_from_grid(grid)
        result = HarassmentResistanceIndicator().compute(board)
        assert result.score < 1.0


# ============================
# ExtensionPotentialIndicator
# ============================


class TestExtensionPotentialIndicator:
    def test_name(self):
        assert ExtensionPotentialIndicator().name == INDICATOR_EXTENSION

    def test_empty_board_no_improvement(self):
        # 空盤面は色無し→ improvement=0、empty_reserve=1.0
        # score = 0.7 * 0 + 0.3 * 1.0 = 0.3
        result = ExtensionPotentialIndicator().compute(make_empty_board())
        assert result.raw_value == 0.0  # 伸ばし候補なし
        assert result.detail["empty_reserve"] == pytest.approx(1.0)
        assert 0.25 < result.score < 0.35

    def test_near_chain_board_high_improvement(self):
        """あと1個で連鎖する盤面は improvement_ratio が高い。"""
        # 赤3個 + 1個で連鎖 → 赤を1個加えればchain=1
        grid = empty_grid()
        grid[12][0] = COLOR_RED
        grid[12][1] = COLOR_RED
        grid[12][2] = COLOR_RED
        result = ExtensionPotentialIndicator().compute(board_from_grid(grid))
        assert result.raw_value > 0.0

    def test_score_in_range(self):
        result = ExtensionPotentialIndicator().compute(make_4_chain_board())
        assert 0.0 <= result.score <= 1.0

    def test_detail_has_search_metrics(self):
        result = ExtensionPotentialIndicator().compute(make_empty_board())
        assert "improvement_ratio" in result.detail
        assert "empty_reserve" in result.detail
        assert "base_chain" in result.detail


# ============================
# SubChainQualityIndicator
# ============================


class TestSubChainQualityIndicator:
    def test_name(self):
        assert SubChainQualityIndicator().name == INDICATOR_SUB_CHAIN

    def test_empty_board_zero(self):
        result = SubChainQualityIndicator().compute(make_empty_board())
        assert result.score == 0.0

    def test_detects_remaining_groups(self):
        # 連鎖しない盤面=全部残る=グループなし(バラバラ)
        result = SubChainQualityIndicator().compute(make_no_chain_board())
        # 全て単独なのでsize>=2のグループは0
        assert result.detail["candidate_count"] == 0

    def test_remaining_pair_counted(self):
        # 赤3個 + 青2個 = 連鎖しない、青ペアが副砲候補
        grid = empty_grid()
        grid[12][0] = COLOR_RED
        grid[12][1] = COLOR_RED
        grid[12][2] = COLOR_RED  # 3個だけなので消えない
        grid[12][4] = COLOR_BLUE
        grid[12][5] = COLOR_BLUE
        result = SubChainQualityIndicator().compute(board_from_grid(grid))
        assert result.detail["candidate_count"] >= 2


# ============================
# SecondChainPotentialIndicator
# ============================


class TestSecondChainPotentialIndicator:
    def test_name(self):
        assert SecondChainPotentialIndicator().name == INDICATOR_SECOND

    def test_empty_board_small_viability_bonus(self):
        """空盤面は構築余地ありの固定値。"""
        result = SecondChainPotentialIndicator().compute(make_empty_board())
        assert 0.0 < result.score < 0.2
        assert result.detail["remaining_empty"] is True

    def test_after_full_chain_empty_bonus(self):
        """4連鎖後は final_board が空 → empty_bonus が付く。"""
        result = SecondChainPotentialIndicator().compute(make_4_chain_board())
        assert result.detail["remaining_empty"] is True

    def test_detects_viable_placement(self):
        # 赤3個のみ → 赤を1つ加えれば size=4 で連鎖発生
        grid = empty_grid()
        grid[12][0] = COLOR_RED
        grid[12][1] = COLOR_RED
        grid[12][2] = COLOR_RED
        board = board_from_grid(grid)
        result = SecondChainPotentialIndicator().compute(board)
        assert result.detail["viable_placements"] > 0
        assert result.detail["best_chain"] >= 1


# ============================
# IndicatorCalculator
# ============================


class TestIndicatorCalculator:
    def test_default_has_all_8_indicators(self):
        calc = IndicatorCalculator()
        names = calc.indicator_names()
        assert set(names) == set(ALL_INDICATOR_NAMES)
        assert len(names) == 8

    def test_compute_all_returns_indicator_set(self):
        calc = IndicatorCalculator()
        result = calc.compute_all(make_4_chain_board())
        assert isinstance(result, IndicatorSet)

    def test_compute_all_covers_all_indicators(self):
        calc = IndicatorCalculator()
        result = calc.compute_all(make_4_chain_board())
        for name in ALL_INDICATOR_NAMES:
            assert name in result.results
            assert 0.0 <= result.score_of(name) <= 1.0

    def test_custom_indicators_used(self):
        calc = IndicatorCalculator(
            indicators=[FieldEfficiencyIndicator(), DeathRiskIndicator()],
        )
        assert set(calc.indicator_names()) == {
            INDICATOR_FIELD_EFF,
            INDICATOR_DEATH_RISK,
        }

    def test_shared_simulator(self):
        sim = ChainSimulator()
        calc = IndicatorCalculator(simulator=sim)
        # 2回呼んでも動くこと
        calc.compute_all(make_empty_board())
        calc.compute_all(make_4_chain_board())

    def test_to_dict_roundtrip(self):
        calc = IndicatorCalculator()
        result = calc.compute_all(make_single_erase_board())
        d = result.to_dict()
        # 8 メイン指標 + 拡張4指標 (shape/touching/tail/color_variance) = 12 件
        assert len(d) >= 8
        for name in ALL_INDICATOR_NAMES:
            assert "score" in d[name]
            assert "raw_value" in d[name]


# ============================
# IncomingOjamaPressureIndicator テスト
# ============================


class TestIncomingOjamaPressureIndicator:
    """予告お邪魔受け圧 (incoming_ojama_pressure) のテスト。"""

    def test_zero_incoming_zero_score(self):
        """incoming_ojama=0 はスコア 0.0 を返す。"""
        from src.old.indicators import (
            INDICATOR_INCOMING_OJAMA,
            IncomingOjamaPressureIndicator,
        )
        ind = IncomingOjamaPressureIndicator()
        result = ind.compute(make_empty_board(), incoming_ojama=0)
        assert result.score == 0.0
        assert result.raw_value == 0.0
        assert result.name == INDICATOR_INCOMING_OJAMA

    def test_full_offset_clamps_to_one(self):
        """incoming_ojama=72 (=MAX_OJAMA_OFFSET) でスコア 1.0、超過時もクランプ。"""
        from src.old.indicators import (
            MAX_OJAMA_OFFSET,
            IncomingOjamaPressureIndicator,
        )
        ind = IncomingOjamaPressureIndicator()
        r1 = ind.compute(make_empty_board(), incoming_ojama=MAX_OJAMA_OFFSET)
        assert r1.score == pytest.approx(1.0)
        r2 = ind.compute(make_empty_board(), incoming_ojama=200)
        assert r2.score == 1.0   # クランプ

    def test_negative_input_treated_as_zero(self):
        """負値の incoming_ojama は 0 として扱う。"""
        from src.old.indicators import IncomingOjamaPressureIndicator
        ind = IncomingOjamaPressureIndicator()
        result = ind.compute(make_empty_board(), incoming_ojama=-10)
        assert result.score == 0.0

    def test_partial_score_proportional(self):
        """incoming_ojama=36 (=72/2) はスコア 0.5。"""
        from src.old.indicators import IncomingOjamaPressureIndicator
        ind = IncomingOjamaPressureIndicator()
        result = ind.compute(make_empty_board(), incoming_ojama=36)
        assert result.score == pytest.approx(0.5, abs=0.01)

    def test_calculator_includes_incoming_ojama(self):
        """compute_all() が incoming_ojama 引数を受け取り、結果に反映する。"""
        from src.old.indicators import INDICATOR_INCOMING_OJAMA
        calc = IndicatorCalculator()
        result = calc.compute_all(
            make_empty_board(), incoming_ojama=72,
        )
        assert INDICATOR_INCOMING_OJAMA in result.results
        assert result.incoming_ojama_pressure == pytest.approx(1.0)
        # 既存のシグネチャ (省略時) でも動く
        result2 = calc.compute_all(make_empty_board())
        assert result2.incoming_ojama_pressure == 0.0
