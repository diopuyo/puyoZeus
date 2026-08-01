"""Step5 (2026-08-01): estimate_expected_net_damage の回帰テスト。

軽量なダミー/手作り盤面のみを使用し、実動画・実npzの重い処理は行わない
(tests/test_exchange_effectiveness_step2.py と同じ方針)。
"""
from __future__ import annotations

from src.board import Board, COLOR_RED, COLOR_BLUE
from src.indicators_v2 import ojama_damage
from scripts.measure_exchange_dynamics import OppCoverageStatus
from scripts.measure_exchange_effectiveness import estimate_expected_net_damage


def _empty_attacker_board() -> Board:
    """空の攻撃側盤面 (おじゃま受け入れ余裕が大きい)。"""
    return Board()


def _empty_opp_board() -> Board:
    """空の相手側盤面 (反撃力ゼロに近い)。"""
    return Board()


def test_opp_chaining_forces_zero_counter_and_full_damage_passthrough() -> None:
    """相手が OPP_CHAINING (連鎖中=応手不能) なら期待反撃量0固定で

    net_expected = attacker_ojama_sent がそのまま ojama_damage に渡る。
    """
    attacker_board = _empty_attacker_board()
    damage = estimate_expected_net_damage(
        attacker_ojama_sent=20.0,
        opp_board=_empty_opp_board(),
        opp_coverage_status=OppCoverageStatus.OPP_CHAINING,
        attacker_chain_count=5,
        attacker_board_after_fire=attacker_board,
    )
    expected = ojama_damage(attacker_board, ojama_count=20.0).score
    assert damage == expected


def test_net_negative_clamped_to_zero_damage() -> None:
    """相手の期待反撃量が攻撃側の送付量を上回る (net_expected負) 場合は

    0 にクランプされ、ojama_damage(board, 0) と同じ (最も軽い) ダメージに
    なる (負のダメージという非現実的な値を返さない)。
    """
    attacker_board = _empty_attacker_board()
    # 相手盤面に厚く積み、OBSERVED として期待反撃量を計算させる
    opp_board = Board()
    opp_board.set(12, 0, COLOR_RED)
    opp_board.set(12, 1, COLOR_BLUE)
    damage = estimate_expected_net_damage(
        attacker_ojama_sent=0.0,  # 攻撃側は何も送っていない
        opp_board=opp_board,
        opp_coverage_status=OppCoverageStatus.OBSERVED,
        attacker_chain_count=1,
        attacker_board_after_fire=attacker_board,
    )
    expected_floor = ojama_damage(attacker_board, ojama_count=0.0).score
    assert damage == expected_floor


def test_observed_status_computes_nontrivial_expected_counter() -> None:
    """OBSERVED かつ有効な相手盤面なら期待反撃量が計算され、

    OPP_CHAINING (常に反撃0) の場合よりダメージが小さくなりうる
    (相手の反撃分が正味お邪魔から差し引かれるため)。
    """
    attacker_board = _empty_attacker_board()
    opp_board = Board()
    opp_board.set(12, 0, COLOR_RED)
    opp_board.set(12, 1, COLOR_BLUE)

    damage_observed = estimate_expected_net_damage(
        attacker_ojama_sent=20.0,
        opp_board=opp_board,
        opp_coverage_status=OppCoverageStatus.OBSERVED,
        attacker_chain_count=5,
        attacker_board_after_fire=attacker_board,
    )
    damage_chaining = estimate_expected_net_damage(
        attacker_ojama_sent=20.0,
        opp_board=opp_board,
        opp_coverage_status=OppCoverageStatus.OPP_CHAINING,
        attacker_chain_count=5,
        attacker_board_after_fire=attacker_board,
    )
    # 相手が反撃可能 (OBSERVED) な分、正味ダメージは OPP_CHAINING 以下になる。
    assert damage_observed <= damage_chaining


def test_return_value_is_clamped_0_to_1() -> None:
    """返り値は ojama_damage の score 同様 0〜1 の範囲に収まる。"""
    attacker_board = _empty_attacker_board()
    opp_board = _empty_opp_board()
    damage = estimate_expected_net_damage(
        attacker_ojama_sent=1000.0,  # 極端な大値
        opp_board=opp_board,
        opp_coverage_status=OppCoverageStatus.OPP_CHAINING,
        attacker_chain_count=19,
        attacker_board_after_fire=attacker_board,
    )
    assert 0.0 <= damage <= 1.0


def test_fast_and_precise_mode_both_runnable() -> None:
    """mode="fast" / "precise" のどちらでも例外なく動作する

    (expected_fire_power に fast 版が無い旨は docstring 参照、現状は
    挙動同一だが interface としては両方受理する)。
    """
    attacker_board = _empty_attacker_board()
    opp_board = Board()
    opp_board.set(12, 0, COLOR_RED)
    opp_board.set(12, 1, COLOR_BLUE)
    for mode in ("precise", "fast"):
        damage = estimate_expected_net_damage(
            attacker_ojama_sent=10.0,
            opp_board=opp_board,
            opp_coverage_status=OppCoverageStatus.OBSERVED,
            attacker_chain_count=3,
            attacker_board_after_fire=attacker_board,
            mode=mode,
        )
        assert isinstance(damage, float)
