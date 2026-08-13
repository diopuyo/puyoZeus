"""src/scoring.py のテスト。公式値との突き合わせを含む。"""
from __future__ import annotations

import pytest

from src.board import (
    BOARD_COLS,
    BOARD_ROWS,
    COLOR_BLUE,
    COLOR_EMPTY,
    COLOR_GREEN,
    COLOR_PURPLE,
    COLOR_RED,
    COLOR_YELLOW,
    Board,
)
from src.chain import ChainSimulator
from src.scoring import (
    ALL_CLEAR_BONUS,
    CHAIN_POWER_INCREMENT,
    CHAIN_POWER_TABLE,
    MARGIN_TIME_DECAY_FACTOR,
    MARGIN_TIME_DECAY_INTERVAL_SEC,
    MARGIN_TIME_MAX_DECAYS,
    MARGIN_TIME_START_SEC,
    MAX_BONUS_MULTIPLIER,
    OJAMA_RATE_MIN,
    OJAMA_RATE_STANDARD,
    SCORE_CONSISTENCY_RATIO_MAX,
    SCORE_CONSISTENCY_RATIO_MIN,
    calculate_chain_score,
    calculate_step_score,
    chain_power,
    color_bonus,
    compute_effective_rate,
    connection_bonus,
    is_score_consistent,
    score_consistency_ratio,
    score_to_ojama,
)
from src.chain import ChainStep, PuyoGroup


def _empty_grid() -> list[list[int]]:
    return [[COLOR_EMPTY] * BOARD_COLS for _ in range(BOARD_ROWS)]


# ============================
# 公式ボーナス値
# ============================


def test_chain_power_official_values() -> None:
    """公式連鎖ボーナス (19連鎖まで)。"""
    assert chain_power(1) == 0
    assert chain_power(2) == 8
    assert chain_power(3) == 16
    assert chain_power(4) == 32
    assert chain_power(5) == 64
    assert chain_power(6) == 96
    assert chain_power(7) == 128
    assert chain_power(8) == 160
    assert chain_power(9) == 192
    assert chain_power(10) == 224
    assert chain_power(15) == 384
    assert chain_power(19) == 512


def test_chain_power_beyond_table_linear_extension() -> None:
    """19連鎖超は +32 ずつ線形延長。"""
    assert chain_power(20) == 512 + CHAIN_POWER_INCREMENT
    assert chain_power(25) == 512 + 6 * CHAIN_POWER_INCREMENT


def test_connection_bonus_official_values() -> None:
    assert connection_bonus(3) == 0    # 3 連結は消えない
    assert connection_bonus(4) == 0
    assert connection_bonus(5) == 2
    assert connection_bonus(6) == 3
    assert connection_bonus(7) == 4
    assert connection_bonus(8) == 5
    assert connection_bonus(9) == 6
    assert connection_bonus(10) == 7
    assert connection_bonus(11) == 10
    assert connection_bonus(20) == 10  # 頭打ち


def test_color_bonus_official_values() -> None:
    assert color_bonus(1) == 0
    assert color_bonus(2) == 3
    assert color_bonus(3) == 6
    assert color_bonus(4) == 12
    assert color_bonus(5) == 24
    assert color_bonus(6) == 24        # 5色で頭打ち


# ============================
# 1 連鎖シンプルケース
# ============================


def test_simple_1chain_4connect_single_color() -> None:
    """
    4 連結 1 色のみの 1 連鎖:
        erased=4, chain_bonus=0, conn_bonus=0, color=0 → max(1, 0)=1
        score = 4 * 10 * 1 = 40
    """
    grid = _empty_grid()
    grid[9][0] = COLOR_RED
    grid[10][0] = COLOR_RED
    grid[11][0] = COLOR_RED
    grid[12][0] = COLOR_RED
    board = Board.from_list(grid)
    result = ChainSimulator().simulate(board)
    assert result.chain_count == 1

    scored = calculate_chain_score(result)
    assert scored.total_score == 40
    assert scored.steps[0].erased_count == 4
    assert scored.steps[0].bonus_multiplier == 1


def test_minimum_chain_score_meets_chain_total_min_score_gate() -> None:
    """W7 根治③ (2026-08-13, docs/KNOWN_WEAKNESSES.md) の判断根拠となる
    不変条件: 実在する連鎖 (chain_count>=1) の最小得点は、消去グループの
    最小サイズ (4連結) × 最小倍率 (max(1,0)=1) で決まり、必ず
    CHAIN_TOTAL_MIN_SCORE (ojama_accounting.py の連鎖ノイズゲート閾値) 以上
    になる。この不変性により
    「score_estimated=True (根治①で simulate 検証済み充填) なら常に既存の
    total_score>=CHAIN_TOTAL_MIN_SCORE ゲートを素通しで満たす」ため、
    ResolvedExchangeTracker のゲート拡張 (根治③の当初案) は不要と判断し
    見送った (scripts/visualize_advantage_overlay.py の ResolvedExchangeTracker
    参照)。本テストが破れたら根治③の簡素化前提が崩れ、ゲート拡張の
    再実装が必要になる。"""
    from src.ojama_accounting import CHAIN_TOTAL_MIN_SCORE

    grid = _empty_grid()
    grid[9][0] = COLOR_RED
    grid[10][0] = COLOR_RED
    grid[11][0] = COLOR_RED
    grid[12][0] = COLOR_RED
    board = Board.from_list(grid)
    result = ChainSimulator().simulate(board)
    assert result.chain_count >= 1
    scored = calculate_chain_score(result)
    assert scored.total_score == CHAIN_TOTAL_MIN_SCORE == 40


def test_2chain_same_color_only() -> None:
    """
    2連鎖シンプルケース。同色のみで 4+4 連結。
        step1: erased=4, bonus=max(1, 0+0+0)=1, score=40
        step2: erased=4, bonus=max(1, 8+0+0)=8, score=320
        total=360
    """
    grid = _empty_grid()
    # 上段に赤 4 (r8)、下段に赤 4 (r12) を接続させず縦に配置
    # 発火後重力で赤が赤の上に落ちて 2 連鎖目が発火するよう調整
    # 簡易には「消える赤 4」と「落ちたら消える赤 4」を用意
    # column 0 に下から: red, red, red, red
    for r in range(9, 13):
        grid[r][0] = COLOR_RED
    # column 1 に下から: blue x4 （落ちると赤 4 になる条件にするため接続しない色でスペーサ）
    # 簡単化: column 0 に赤 4、その上段から column 1 に別色、消えても 2 連鎖しない
    # 別手法: ブロックパターンで 2 連鎖化
    grid = _empty_grid()
    # column 0: red red red red (消去)、上に blue blue (落下)
    grid[11][0] = COLOR_RED
    grid[12][0] = COLOR_RED
    grid[10][0] = COLOR_RED
    grid[9][0] = COLOR_RED
    grid[8][0] = COLOR_BLUE
    grid[7][0] = COLOR_BLUE
    # column 1: blue blue blue (接続待ち、1段目が空欄)
    grid[12][1] = COLOR_BLUE
    grid[11][1] = COLOR_BLUE
    grid[10][1] = COLOR_BLUE
    board = Board.from_list(grid)
    result = ChainSimulator().simulate(board)
    assert result.chain_count == 2
    # step1: 赤 4 消し → score 40 (chain=1, conn=0, color=0)
    # step2: 青 5 連結 (col0 の blue x2 が落ちて col1 の blue x3 と接続)
    #        erased=5, chain=8, conn=2, color=0 → bonus=10, score=5*10*10=500
    scored = calculate_chain_score(result)
    assert scored.steps[0].score == 40
    assert scored.steps[1].erased_count == 5
    assert scored.steps[1].chain_bonus == 8
    assert scored.steps[1].connection_bonus_total == 2
    assert scored.steps[1].score == 500
    assert scored.total_score == 540


def test_multi_color_simultaneous() -> None:
    """1 ステップで 2 色同時消し、色数ボーナス発生。"""
    grid = _empty_grid()
    # 赤 4 連結 (column 0) + 青 4 連結 (column 1) を同時に消す
    for r in range(9, 13):
        grid[r][0] = COLOR_RED
        grid[r][1] = COLOR_BLUE
    board = Board.from_list(grid)
    result = ChainSimulator().simulate(board)
    assert result.chain_count == 1
    scored = calculate_chain_score(result)
    # erased=8, chain=0, conn=0+0=0, color=3 → bonus=max(1,3)=3
    # score = 8 * 10 * 3 = 240
    assert scored.steps[0].color_bonus == 3
    assert scored.total_score == 240


# ============================
# 全消しボーナス
# ============================


def test_chain_score_does_not_include_all_clear_bonus() -> None:
    """新仕様: calculate_chain_score は全消しボーナスを含まない素点を返す。"""
    grid = _empty_grid()
    for r in range(9, 13):
        grid[r][0] = COLOR_RED
    board = Board.from_list(grid)
    result = ChainSimulator().simulate(board)
    scored = calculate_chain_score(result)
    # 1 連鎖 4 連結 score = 40（全消しボーナス 2100 は含まない）
    assert scored.total_score == 40
    # 全消しフラグは立つ（次連鎖でボーナス適用するための信号）
    assert scored.is_all_clear is True


def test_chain_score_all_clear_flag_false_when_remnant_exists() -> None:
    """残ぷよがあれば is_all_clear=False。"""
    grid = _empty_grid()
    for r in range(9, 13):
        grid[r][0] = COLOR_RED
    grid[12][5] = COLOR_YELLOW
    board = Board.from_list(grid)
    result = ChainSimulator().simulate(board)
    scored = calculate_chain_score(result)
    assert scored.is_all_clear is False
    assert scored.total_score == 40


# ============================
# おじゃま換算
# ============================


def test_score_to_ojama_basic() -> None:
    """基本: 70 点で 1 個、余りは繰越。"""
    r = score_to_ojama(140, prev_leftover=0)
    assert r.ojama_count == 2
    assert r.leftover_score == 0
    assert r.effective_rate == OJAMA_RATE_STANDARD

    r = score_to_ojama(150, prev_leftover=0)
    assert r.ojama_count == 2
    assert r.leftover_score == 10


def test_score_to_ojama_leftover_carries() -> None:
    """繰越が次の発火に加算される。"""
    r = score_to_ojama(65, prev_leftover=10)
    # 75 / 70 = 1 余り 5
    assert r.ojama_count == 1
    assert r.leftover_score == 5


def test_score_to_ojama_rounds_down() -> None:
    """切り捨て挙動の確認。"""
    r = score_to_ojama(69, prev_leftover=0)
    assert r.ojama_count == 0
    assert r.leftover_score == 69


# ============================
# マージンタイム
# ============================


def test_margin_time_base_rate_before_start() -> None:
    assert compute_effective_rate(0.0) == OJAMA_RATE_STANDARD
    assert compute_effective_rate(95.0) == OJAMA_RATE_STANDARD
    assert compute_effective_rate(MARGIN_TIME_START_SEC) == OJAMA_RATE_STANDARD


def test_margin_time_first_decay() -> None:
    """96 秒を 1 秒でも過ぎると最初の減衰（0.75 倍）。"""
    t = MARGIN_TIME_START_SEC + 1.0
    rate = compute_effective_rate(t)
    expected = int(OJAMA_RATE_STANDARD * MARGIN_TIME_DECAY_FACTOR)
    assert rate == expected
    # 52 (= 70 * 0.75 切り捨て)
    assert rate == 52


def test_margin_time_progressive_decay() -> None:
    """16 秒間隔で段階的に減衰。"""
    # 96 + 16 = 112 秒で 2 段目 (0.75^2 = 0.5625)
    t = MARGIN_TIME_START_SEC + MARGIN_TIME_DECAY_INTERVAL_SEC + 1.0
    rate = compute_effective_rate(t)
    expected = int(OJAMA_RATE_STANDARD * MARGIN_TIME_DECAY_FACTOR ** 2)
    assert rate == expected


def test_margin_time_floor_rate() -> None:
    """十分な時間経過で下限 1 に到達（公式仕様）。"""
    rate = compute_effective_rate(10000.0)
    assert rate == OJAMA_RATE_MIN
    assert rate == 1


def test_margin_time_decays_capped_at_14() -> None:
    """公式: 減衰は最大 14 回で停止し、以降は固定値。"""
    # 14 回減衰時点と、それより遥かに後の時点が同じレート
    t14 = MARGIN_TIME_START_SEC + MARGIN_TIME_DECAY_INTERVAL_SEC * 14 + 1.0
    t_far = MARGIN_TIME_START_SEC + MARGIN_TIME_DECAY_INTERVAL_SEC * 100
    rate14 = compute_effective_rate(t14)
    rate_far = compute_effective_rate(t_far)
    assert rate14 == rate_far


def test_margin_time_decay_max_constant_value() -> None:
    """14 回減衰後の rate は 70 * 0.75^14 切り捨て = 1（クランプ後）"""
    expected_raw = int(70 * (MARGIN_TIME_DECAY_FACTOR ** MARGIN_TIME_MAX_DECAYS))
    # 70 * 0.75^14 ≈ 1.31 → int で 1
    assert expected_raw == 1
    t = MARGIN_TIME_START_SEC + MARGIN_TIME_DECAY_INTERVAL_SEC * MARGIN_TIME_MAX_DECAYS + 1.0
    assert compute_effective_rate(t) == 1


# ============================
# 999 クランプ
# ============================


def _fake_step(chain_idx: int, group_size: int, num_colors: int) -> ChainStep:
    """指定パラメータで擬似的な ChainStep を作る（PuyoGroup を最小限に）。"""
    cells = frozenset({(0, c) for c in range(group_size)})
    groups = []
    for ci in range(num_colors):
        # 各色 1 グループ、サイズはまとめて group_size に分割しないで
        # 最後の色だけ size を持たせる簡略実装
        if ci == num_colors - 1:
            g = PuyoGroup(
                color=ci + 1,
                cells=cells,
                size=group_size,
                ojama_adjacent=frozenset(),
            )
        else:
            # サイズ 4 のダミー（消去対象）
            g = PuyoGroup(
                color=ci + 1,
                cells=frozenset({(1, ci)}),
                size=4,
                ojama_adjacent=frozenset(),
            )
        groups.append(g)
    erased = sum(g.size for g in groups)
    return ChainStep(
        chain_index=chain_idx,
        erased_groups=groups,
        erased_ojama=0,
        erased_count=erased,
        board_before=Board.from_list(_empty_grid()),
        board_after=Board.from_list(_empty_grid()),
    )


def test_bonus_multiplier_clamped_at_999() -> None:
    """超大連鎖で raw_bonus が 999 を超えても 999 にクランプ。"""
    # 35 連鎖目: CP = 512 + 16*32 = 1024、 + 連結 10 + 色 24 = 1058
    step = _fake_step(chain_idx=35, group_size=11, num_colors=5)
    sc = calculate_step_score(step)
    assert sc.bonus_multiplier == MAX_BONUS_MULTIPLIER == 999


def test_bonus_multiplier_below_999_unchanged() -> None:
    """通常範囲ではクランプ非発動。"""
    # 5 連鎖目 4 連結 1 色: CP=64, conn=0, color=0 → bonus=64
    step = _fake_step(chain_idx=5, group_size=4, num_colors=1)
    sc = calculate_step_score(step)
    assert sc.bonus_multiplier == 64


# ============================
# 得点整合性チェック (2026-07-29 追加)
# ============================


def test_score_consistency_ratio_exact_match_is_1() -> None:
    """期待得点と実測が完全一致するとき比率は1.0になること。"""
    assert score_consistency_ratio(1000, 1000) == pytest.approx(1.0)


def test_score_consistency_ratio_known_bad_case_c54() -> None:
    """c54 の既知の疑義事例 (期待40点 vs 実測7598点) は大幅な比率になること
    (all_clear_carryover を考慮しても 2140 に対し 3.55 倍で不整合の範囲)。"""
    ratio = score_consistency_ratio(40, 7598)
    # 全消し仮説 (40+2100=2140) を考慮しても 7598/2140 ≈ 3.55
    assert ratio == pytest.approx(7598 / 2140, rel=1e-6)
    assert not is_score_consistent(40, 7598)


def test_score_consistency_ratio_all_clear_carryover_accepted() -> None:
    """全消し繰越を考慮すると許容範囲になるケース (期待40点+2100点=2140点付近)。"""
    ratio = score_consistency_ratio(40, 2140, allow_all_clear_carryover=True)
    assert ratio == pytest.approx(1.0, rel=1e-6)
    assert is_score_consistent(40, 2140)


def test_score_consistency_ratio_all_clear_carryover_disabled() -> None:
    """allow_all_clear_carryover=False では全消し仮説を使わないこと
    (後方互換: 既定 True、明示 False で従来の単純比較のみ)。"""
    ratio = score_consistency_ratio(40, 2140, allow_all_clear_carryover=False)
    assert ratio == pytest.approx(2140 / 40, rel=1e-6)
    assert not is_score_consistent(40, 2140, allow_all_clear_carryover=False)


def test_is_score_consistent_within_default_band() -> None:
    """許容比率 [0.5, 2.0] 内は整合と判定されること。"""
    assert is_score_consistent(100, 100)
    assert is_score_consistent(100, 50)  # ratio=0.5 (下限ちょうど)
    assert is_score_consistent(100, 200)  # ratio=2.0 (上限ちょうど)


def test_is_score_consistent_outside_default_band() -> None:
    """許容比率の外は不整合と判定されること。"""
    assert not is_score_consistent(100, 49)
    assert not is_score_consistent(100, 201)


def test_score_consistency_ratio_custom_band() -> None:
    """ratio_min/ratio_max を明示指定すると既定値の代わりに使われること。"""
    assert is_score_consistent(100, 250, ratio_min=0.1, ratio_max=3.0)
    assert not is_score_consistent(100, 250, ratio_min=0.1, ratio_max=2.0)


def test_score_consistency_ratio_zero_expected_nonzero_observed_is_inf() -> None:
    """期待得点が0以下で実測>0の場合は明確な不整合 (inf) を返すこと。"""
    ratio = score_consistency_ratio(0, 100, allow_all_clear_carryover=False)
    assert ratio == float("inf")


def test_score_consistency_default_thresholds_are_module_constants() -> None:
    """既定の許容比率が SCORE_CONSISTENCY_RATIO_MIN/MAX と一致すること。"""
    assert SCORE_CONSISTENCY_RATIO_MIN == pytest.approx(0.5)
    assert SCORE_CONSISTENCY_RATIO_MAX == pytest.approx(2.0)
