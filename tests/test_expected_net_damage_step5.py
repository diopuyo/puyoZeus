"""Step5 (2026-08-01): estimate_expected_net_damage の回帰テスト。

軽量なダミー/手作り盤面のみを使用し、実動画・実npzの重い処理は行わない
(tests/test_exchange_effectiveness_step2.py と同じ方針)。
"""
from __future__ import annotations

from src.board import Board, COLOR_RED, COLOR_BLUE
from src.indicators_v2 import OJAMA_DAMAGE_FLOOR, ojama_damage
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
    2026-08-02修正: net_expected は相手側に着弾するため、受け側=相手の
    盤面 (opp_board) で評価する (旧実装は attacker_board_after_fire を
    使っておりバグだった、main精査で実バグ確定)。
    """
    attacker_board = _empty_attacker_board()
    opp_board = _empty_opp_board()
    damage = estimate_expected_net_damage(
        attacker_ojama_sent=20.0,
        opp_board=opp_board,
        opp_coverage_status=OppCoverageStatus.OPP_CHAINING,
        attacker_chain_count=5,
        attacker_board_after_fire=attacker_board,
    )
    expected = ojama_damage(opp_board, ojama_count=20.0).score
    assert damage == expected


def test_net_negative_clamped_to_zero_damage() -> None:
    """相手の期待反撃量が攻撃側の送付量を上回る (net_expected負) 場合は

    0 にクランプされ、ojama_damage(opp_board, 0) と同じ (最も軽い) ダメージに
    なる (負のダメージという非現実的な値を返さない)。2026-08-02修正:
    評価基準は受け側=相手の盤面 (opp_board) に統一。
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
    expected_floor = ojama_damage(opp_board, ojama_count=0.0).score
    assert damage == expected_floor


def test_fuller_opp_board_yields_larger_damage_for_same_net_expected() -> None:
    """相手盤面が埋まっているほど、同じ net_expected でもダメージが大きくなる

    (user伝授ドメインルール reference_ojama_damage_nonlinear_2026-07-29:
    威力は受け側の残り容量に依存する非線形性、の直接確認。2026-08-02修正の
    直接検証: 旧実装 [attacker_board_after_fire基準] だと opp_board を
    どれだけ変えても damage は不変 [常に空のattacker_boardで評価] になり
    このテストは失敗するはずだった)。

    OPP_CHAINING で期待反撃量を0固定にし net_expected=attacker_ojama_sent を
    完全に固定した上で、opp_board の埋まり具合だけを変えて比較する。
    窒息判定列 (DEATH_COL=2) を窒息一歩手前 (row=1 は空のまま残す) まで
    交互色 (同色4連結を作らず _takapt_best_drop を撹乱しない) で積み、
    ほぼ空の盤面と比較する。
    """
    attacker_board = _empty_attacker_board()
    opp_board_sparse = Board()  # 空 (余裕段数が最大)
    opp_board_full = Board()
    colors = (COLOR_RED, COLOR_BLUE)
    for i, row in enumerate(range(2, 13)):  # row=1 (窒息行) は空のまま残す
        opp_board_full.set(row, 2, colors[i % 2])

    attacker_ojama_sent = 48.0  # 8段相当、opp_board_full の残り余裕を大きく食い込ませる
    damage_sparse = estimate_expected_net_damage(
        attacker_ojama_sent=attacker_ojama_sent,
        opp_board=opp_board_sparse,
        opp_coverage_status=OppCoverageStatus.OPP_CHAINING,
        attacker_chain_count=5,
        attacker_board_after_fire=attacker_board,
    )
    damage_full = estimate_expected_net_damage(
        attacker_ojama_sent=attacker_ojama_sent,
        opp_board=opp_board_full,
        opp_coverage_status=OppCoverageStatus.OPP_CHAINING,
        attacker_chain_count=5,
        attacker_board_after_fire=attacker_board,
    )
    assert damage_full > damage_sparse
    assert damage_sparse <= OJAMA_DAMAGE_FLOOR + 1e-9  # 空盤面は「受けても無害」帯のまま
    assert damage_full >= 0.9  # 窒息一歩手前まで積んだ盤面はほぼ最大ダメージ


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
