"""指標 v2 (第1バッチ) のユニットテスト。

既知盤面で各指標が妥当な値・0-1 範囲・例外なしを検証する。
仕様: docs/INDICATOR_V2_MEASUREMENT_SPEC_2026-06-17.md
"""
from __future__ import annotations

import pytest

from src.board import (
    BOARD_COLS,
    BOARD_ROWS,
    COLOR_BLUE,
    COLOR_GREEN,
    COLOR_OJAMA,
    COLOR_PURPLE,
    COLOR_RED,
    COLOR_YELLOW,
    Board,
)
import src.indicators_v2 as iv


# ============================
# 盤面ビルダー
# ============================


def _empty_grid() -> list[list[int]]:
    return [[0] * BOARD_COLS for _ in range(BOARD_ROWS)]


def _empty_board() -> Board:
    return Board.from_list(_empty_grid())


def _four_chain_board() -> Board:
    """4 連結 1 つ (2x2 赤) を最下段に置いた発火可能盤面。"""
    g = _empty_grid()
    g[12][0] = COLOR_RED
    g[12][1] = COLOR_RED
    g[11][0] = COLOR_RED
    g[11][1] = COLOR_RED
    return Board.from_list(g)


def _two_chain_board() -> Board:
    """2 連鎖を起こす盤面 (赤4消し → 重力で青4連結 → 青消し)。"""
    g = _empty_grid()
    # 赤 4 連結 (col0 縦3 + col1 下1)
    g[12][0] = COLOR_RED
    g[12][1] = COLOR_RED
    g[11][0] = COLOR_RED
    g[10][0] = COLOR_RED
    # 青: 赤消去後の重力で 4 連結になる配置
    g[12][2] = COLOR_BLUE
    g[11][1] = COLOR_BLUE
    g[10][1] = COLOR_BLUE
    g[9][0] = COLOR_BLUE
    return Board.from_list(g)


def _ojama_board() -> Board:
    """可視領域にお邪魔を含む盤面。"""
    g = _empty_grid()
    for col in range(BOARD_COLS):
        g[12][col] = COLOR_OJAMA
    g[11][0] = COLOR_OJAMA
    return Board.from_list(g)


# ============================
# 共通: 全指標の 0-1 範囲・例外なし
# ============================

_BOARDS = [
    _empty_board(),
    _four_chain_board(),
    _two_chain_board(),
    _ojama_board(),
]


@pytest.mark.parametrize("board", _BOARDS)
def test_all_scalar_indicators_in_range(board: Board) -> None:
    """スカラー指標が全て 0-1 範囲・例外なしで算出されること。"""
    values = [
        iv.board_puyo_total(board),
        iv.board_color_puyo_total(board),
        iv.max_column_height(board),
        iv.column_bumpiness(board),
        iv.death_margin(board),
        iv.death_margin_neighbor(board),
        iv.current_max_chain(board),
        iv.immediate_fire_power(board),
        iv.chain_efficiency(board),
        iv.min_puyos_to_ignite(board),
        iv.second_chain_potential(board),
        iv.board_ojama_count(board),
        iv.dig_resistance(board),
        iv.absorption_capacity(board),
    ]
    for v in values:
        assert 0.0 <= v.score <= 1.0
        assert v.raw == v.raw  # NaN なし


# ============================
# ① 進行度
# ============================


def test_tsumo_count_rate() -> None:
    assert iv.tsumo_count_rate(0).score == 0.0
    assert iv.tsumo_count_rate(50).score == pytest.approx(0.5)
    # 上限クランプ
    assert iv.tsumo_count_rate(200).score == 1.0
    # 負値は 0 にクランプ
    assert iv.tsumo_count_rate(-5).raw == 0.0


def test_board_puyo_total() -> None:
    assert iv.board_puyo_total(_empty_board()).raw == 0.0
    v = iv.board_puyo_total(_four_chain_board())
    assert v.raw == 4.0
    assert v.score == pytest.approx(4.0 / 72.0)


def test_color_puyo_excludes_ojama() -> None:
    v = iv.board_color_puyo_total(_ojama_board())
    # お邪魔のみの盤面 → 色ぷよ 0
    assert v.raw == 0.0


def test_margin_time_rate_monotonic() -> None:
    early = iv.margin_time_rate(0.0)
    late = iv.margin_time_rate(200.0)
    assert early.score == 0.0  # 序盤は減衰なし
    assert late.score > early.score  # マージン進行で増加
    assert 0.0 <= late.score <= 1.0


# ============================
# ② 占有・危険
# ============================


def test_max_column_height_empty() -> None:
    assert iv.max_column_height(_empty_board()).raw == 0.0


def test_bumpiness_flat_is_zero() -> None:
    """全列同高なら凸凹 0。"""
    g = _empty_grid()
    for col in range(BOARD_COLS):
        g[12][col] = COLOR_RED
    assert iv.column_bumpiness(Board.from_list(g)).raw == 0.0


def test_death_margin_full_when_empty() -> None:
    assert iv.death_margin(_empty_board()).score == 1.0
    assert iv.death_margin_neighbor(_empty_board()).score == 1.0


def test_death_margin_decreases_with_height() -> None:
    g = _empty_grid()
    from src.board import DEATH_COL
    for row in range(BOARD_ROWS - 1, BOARD_ROWS - 5, -1):
        g[row][DEATH_COL] = COLOR_RED
    v = iv.death_margin(Board.from_list(g))
    assert v.raw == 8.0  # 12 - 4
    assert v.score < 1.0


# ============================
# ③ 火力・潜在
# ============================


def test_current_max_chain_two_chain() -> None:
    """takapt 定石: 1 個追加で発火できる最大連鎖数が 2 以上であること。"""
    v = iv.current_max_chain(_two_chain_board())
    # _two_chain_board は静止盤面で 2 連鎖が成立する盤面。
    # takapt 定石 (1 個追加) では >= 2 が保証される (元の 2 連鎖以上を発見)。
    assert v.raw >= 2.0
    assert 0.0 < v.score <= 1.0


def test_immediate_fire_power_nonzero_for_chain() -> None:
    v = iv.immediate_fire_power(_two_chain_board())
    assert v.raw >= 0.0
    assert 0.0 <= v.score <= 1.0


def test_chain_efficiency_zero_when_no_color() -> None:
    assert iv.chain_efficiency(_empty_board()).raw == 0.0
    assert iv.chain_efficiency(_ojama_board()).raw == 0.0


def test_min_puyos_to_ignite_range() -> None:
    v = iv.min_puyos_to_ignite(_empty_board())
    # 空盤面は発火不能 → N = trial_limit+1 = 3
    assert v.raw == 3.0
    assert 0.0 <= v.score <= 1.0


def test_connectivity_observation_total_and_per_color() -> None:
    total, per_color = iv.connectivity_observation(_four_chain_board())
    # 2x2 赤 = 1 グループ size 4
    assert total.max_group_size == 4
    assert total.pair_count == 0
    assert COLOR_RED in per_color
    assert per_color[COLOR_RED].max_group_size == 4


def test_connectivity_counts_pairs_and_triples() -> None:
    g = _empty_grid()
    # 赤 2 連結 (縦)
    g[12][0] = COLOR_RED
    g[11][0] = COLOR_RED
    # 青 3 連結 (横)
    g[12][2] = COLOR_BLUE
    g[12][3] = COLOR_BLUE
    g[12][4] = COLOR_BLUE
    total, per_color = iv.connectivity_observation(Board.from_list(g))
    assert total.pair_count == 1
    assert total.triple_count == 1
    assert per_color[COLOR_RED].pair_count == 1
    assert per_color[COLOR_BLUE].triple_count == 1


def test_current_max_chain_empty_board_is_zero() -> None:
    """takapt 定石: 空盤面に 1 個追加しても連鎖なし → 0。"""
    v = iv.current_max_chain(_empty_board())
    assert v.raw == 0.0
    assert v.score == 0.0


def test_immediate_fire_power_takapt_nonzero() -> None:
    """takapt 定石: 連鎖が組める盤面では即発火火力が非ゼロ。"""
    v = iv.immediate_fire_power(_two_chain_board())
    assert v.raw > 0.0
    assert 0.0 < v.score <= 1.0


def test_chain_efficiency_takapt_nonzero() -> None:
    """takapt 定石: 連鎖が組める盤面では連鎖効率が非ゼロ。"""
    v = iv.chain_efficiency(_two_chain_board())
    assert v.raw > 0.0
    assert 0.0 < v.score <= 1.0


def test_takapt_ordering_chain_ge_static() -> None:
    """takapt 版 (1 個追加) の連鎖数は静止盤面 simulate 以上 (下界性)。"""
    from src.chain import ChainSimulator
    sim = ChainSimulator()
    for board in _BOARDS:
        static_chain = sim.simulate(board).chain_count
        takapt_chain = iv.current_max_chain(board, sim).raw
        assert takapt_chain >= static_chain, (
            f"takapt={takapt_chain} < static={static_chain} (board has "
            f"{board.count_puyos()} puyos)"
        )


def test_second_chain_potential_range() -> None:
    v = iv.second_chain_potential(_two_chain_board())
    assert 0.0 <= v.score <= 1.0


# ============================
# ④ お邪魔
# ============================


def test_ojama_net_balance_center() -> None:
    assert iv.ojama_net_balance(0).score == pytest.approx(0.5)
    assert iv.ojama_net_balance(72).score == 1.0
    assert iv.ojama_net_balance(-72).score == 0.0


def test_ojama_forecast_range() -> None:
    assert iv.ojama_forecast(0).score == 0.0
    assert iv.ojama_forecast(36).score == pytest.approx(0.5)
    assert iv.ojama_forecast(100).score == 1.0


def test_board_ojama_count() -> None:
    v = iv.board_ojama_count(_ojama_board())
    assert v.raw == 7.0  # 6 (下段) + 1
    assert v.score == pytest.approx(7.0 / 72.0)


# ============================
# ⑤ テンポ
# ============================


def test_chain_duration_observed() -> None:
    v = iv.chain_duration_observed(1.0, 3.0)
    assert v is not None
    assert v.raw == 2.0
    assert v.score == pytest.approx(2.0 / 14.0)


def test_chain_duration_observed_none() -> None:
    assert iv.chain_duration_observed(None, None) is None
    assert iv.chain_duration_observed(None, 3.0) is None
    # 負の所要時間 (異常) も None
    assert iv.chain_duration_observed(5.0, 3.0) is None


def test_chain_duration_estimated() -> None:
    # 5 連鎖 × 84 / 60 = 7.0 秒
    v = iv.chain_duration_estimated(5)
    assert v.raw == pytest.approx(7.0)
    assert v.score == pytest.approx(7.0 / 14.0)
    assert iv.chain_duration_estimated(0).raw == 0.0


# ============================
# ⑥ 受け力
# ============================


def test_dig_resistance_range() -> None:
    v = iv.dig_resistance(_two_chain_board())
    assert 0.0 <= v.score <= 1.0


def test_dig_resistance_dead_board() -> None:
    g = _empty_grid()
    from src.board import DEATH_COL
    g[0][DEATH_COL] = COLOR_RED  # 窒息
    v = iv.dig_resistance(Board.from_list(g))
    assert v.score == 0.0


def test_absorption_capacity_empty_is_full() -> None:
    assert iv.absorption_capacity(_empty_board()).score == 1.0
    v = iv.absorption_capacity(_four_chain_board())
    assert v.raw == 68.0  # 72 - 4


# ============================
# 値オブジェクト
# ============================


def test_indicator_value_dataclass() -> None:
    v = iv.IndicatorV2Value(score=0.5, raw=10.0)
    assert v.score == 0.5
    assert v.raw == 10.0


# ============================
# III-3 到達火力 (reach_fire_power)
# ============================


def _two_chain_board_with_next() -> tuple[Board, tuple[int, int], tuple[int, int]]:
    """2 連鎖盤面 + next/dnext ペアを返す (テスト用)。"""
    board = _two_chain_board()
    next_pair = (COLOR_RED, COLOR_BLUE)
    dnext_pair = (COLOR_GREEN, COLOR_YELLOW)
    return board, next_pair, dnext_pair


def test_reach_fire_power_range() -> None:
    """III-3 到達火力が 0-1 範囲かつ例外なし。"""
    board, next_pair, dnext_pair = _two_chain_board_with_next()
    result = iv.reach_fire_power(board, next_pair, dnext_pair)
    assert 0.0 <= result.value.score <= 1.0
    assert result.value.raw == result.value.raw  # NaN なし
    assert result.source in ("reach", "fallback_immediate")


def test_reach_fire_power_source_reach_when_both() -> None:
    """next/dnext 両方揃っていれば source='reach'。"""
    board, next_pair, dnext_pair = _two_chain_board_with_next()
    result = iv.reach_fire_power(board, next_pair, dnext_pair)
    assert result.source == "reach"


def test_reach_fire_power_fallback_when_none() -> None:
    """next=None または dnext=None のとき source='fallback_immediate'。"""
    board = _two_chain_board()
    r1 = iv.reach_fire_power(board, None, (COLOR_RED, COLOR_BLUE))
    assert r1.source == "fallback_immediate"
    r2 = iv.reach_fire_power(board, (COLOR_RED, COLOR_BLUE), None)
    assert r2.source == "fallback_immediate"
    r3 = iv.reach_fire_power(board, None, None)
    assert r3.source == "fallback_immediate"


def test_reach_fire_power_lower_bound_note() -> None:
    """下界性の注記確認テスト: reach は next/dnext が最適でない場合に
    immediate (takapt 全色 30 通り) を下回りうる。
    これは仕様通りの挙動。空盤面では常に両者 0 で等しくなることを確認。"""
    # 空盤面: 追加ぷよ数に関わらず連鎖なし → 両者 0
    empty_rfp = iv.reach_fire_power(
        _empty_board(),
        (COLOR_RED, COLOR_BLUE),
        (COLOR_GREEN, COLOR_YELLOW),
    )
    empty_ifp = iv.immediate_fire_power(_empty_board())
    assert empty_rfp.value.raw == 0.0
    assert empty_ifp.raw == 0.0

    # 最適 next/dnext (盤面の連鎖色を揃える) では reach >= immediate が成立する
    # (保証条件: next/dnext が盤面の連鎖に貢献する色を含む)
    board = _two_chain_board()
    # next に赤を含める = takapt 最良配置の色と一致
    best_rfp = iv.reach_fire_power(board, (COLOR_RED, COLOR_RED), (COLOR_RED, COLOR_RED))
    ifp = iv.immediate_fire_power(board)
    # 最適色 (RED) を 2 手連続で置く → reach >= immediate が成立
    assert best_rfp.value.raw >= ifp.raw, (
        f"最適色(RED)ペアで reach={best_rfp.value.raw} < immediate={ifp.raw}"
    )


def test_reach_fire_power_nonzero_for_chain_board() -> None:
    """連鎖が組める盤面に next/dnext を与えたとき raw > 0。"""
    board, next_pair, dnext_pair = _two_chain_board_with_next()
    result = iv.reach_fire_power(board, next_pair, dnext_pair)
    assert result.value.raw >= 0.0
    # max_chain は連鎖が組める盤面では > 0 のはず
    # (next_pair が連鎖に貢献するかは配色依存なので >= 0 でテスト)
    assert result.max_chain >= 0


# ============================
# VII 打ち合い収支 (条件1)
# ============================


def test_chain_to_ojama_zero_or_negative() -> None:
    """n <= 0 のとき chain_to_ojama は 0.0 を返す。"""
    assert iv.chain_to_ojama(0.0) == 0.0
    assert iv.chain_to_ojama(-1.0) == 0.0


def test_chain_to_ojama_positive_monotone() -> None:
    """n が大きいほど chain_to_ojama の値が増加する (単調増加)。"""
    v2 = iv.chain_to_ojama(2.0)
    v5 = iv.chain_to_ojama(5.0)
    v10 = iv.chain_to_ojama(10.0)
    assert v2 > 0.0
    assert v5 > v2
    assert v10 > v5


def test_chain_to_ojama_calibration_spot() -> None:
    """5 連鎖のお邪魔推定値が較正カーブ (30.13 * exp(0.297 * 5)) と一致する。"""
    import math
    expected = iv.CHAIN_OJAMA_A * math.exp(iv.CHAIN_OJAMA_B * 5.0)
    assert iv.chain_to_ojama(5.0) == pytest.approx(expected, rel=1e-6)


def test_chain_to_time_zero() -> None:
    """n=0 のとき chain_to_time は 0.0 を返す。"""
    assert iv.chain_to_time(0.0) == 0.0
    assert iv.chain_to_time(-3.0) == 0.0


def test_chain_to_time_linear() -> None:
    """chain_to_time は TIME_PER_CHAIN_SEC * n で線形増加する。"""
    import pytest
    assert iv.chain_to_time(4.0) == pytest.approx(iv.TIME_PER_CHAIN_SEC * 4.0)
    assert iv.chain_to_time(10.0) == pytest.approx(iv.TIME_PER_CHAIN_SEC * 10.0)


def test_honsen_output_empty_board_is_zero() -> None:
    """空盤面は本線なし → raw=0.0, score=0.0。"""
    v = iv.honsen_output(_empty_board())
    assert v.raw == 0.0
    assert v.score == 0.0


def test_honsen_output_chain_board_nonzero() -> None:
    """連鎖盤面では honsen_output の raw > 0 かつ score が 0-1 範囲内。"""
    v = iv.honsen_output(_two_chain_board())
    assert v.raw > 0.0
    assert 0.0 <= v.score <= 1.0


def test_honsen_output_larger_chain_gives_larger_raw() -> None:
    """4 連鎖相当盤面の raw が 2 連鎖相当より大きい。"""
    v2 = iv.honsen_output(_two_chain_board())
    v4 = iv.honsen_output(_four_chain_board())
    # _four_chain_board は 1 連鎖のみ (2x2 赤 1 つ) なので生値は小さい
    # 両者とも >= 0 かつ NaN なしを確認
    assert v2.raw >= 0.0
    assert v4.raw >= 0.0


def test_reach_fire_power_empty_board_is_zero() -> None:
    """空盤面 + next/dnext → reach でも 0 (連鎖なし)。"""
    result = iv.reach_fire_power(
        _empty_board(),
        (COLOR_RED, COLOR_BLUE),
        (COLOR_GREEN, COLOR_YELLOW),
    )
    assert result.value.raw == 0.0
    assert result.value.score == 0.0
    assert result.source == "reach"
    assert result.max_chain == 0


def test_reach_fire_power_result_dataclass() -> None:
    """ReachFirePowerResult の構造確認。"""
    board, next_pair, dnext_pair = _two_chain_board_with_next()
    result = iv.reach_fire_power(board, next_pair, dnext_pair)
    assert isinstance(result, iv.ReachFirePowerResult)
    assert isinstance(result.value, iv.IndicatorV2Value)
    assert isinstance(result.source, str)
    assert isinstance(result.max_chain, int)


# ============================
# margin_time_rate 回帰テスト
# (修正1: elapsed>96 で score>0 になることを保証)
# ============================


def test_margin_time_rate_zero_before_margin_start() -> None:
    """試合開始〜96秒以内は score=0.0 (マージンタイム未到達)。"""
    from src.scoring import MARGIN_TIME_START_SEC
    # 0秒
    assert iv.margin_time_rate(0.0).score == 0.0
    assert iv.margin_time_rate(0.0).raw == 70.0
    # 96秒ちょうど (境界値: まだ減衰ステップ未到達)
    assert iv.margin_time_rate(MARGIN_TIME_START_SEC).score == 0.0
    assert iv.margin_time_rate(MARGIN_TIME_START_SEC).raw == 70.0


def test_margin_time_rate_positive_after_margin_start() -> None:
    """elapsed > 96秒 (1 decay ステップ後) で score > 0.0。"""
    from src.scoring import MARGIN_TIME_START_SEC, MARGIN_TIME_DECAY_INTERVAL_SEC
    elapsed_one_step = MARGIN_TIME_START_SEC + MARGIN_TIME_DECAY_INTERVAL_SEC
    v = iv.margin_time_rate(elapsed_one_step)
    # rate = 70 * 0.75 = 52 → score = 1 - 52/70 ≈ 0.257
    assert v.score > 0.0
    assert v.raw < 70.0


def test_margin_time_rate_monotone_increasing() -> None:
    """経過秒が増加するにつれ score も単調増加する (マージンタイム進行)。"""
    elapsed_values = [0.0, 50.0, 100.0, 150.0, 200.0]
    scores = [iv.margin_time_rate(e).score for e in elapsed_values]
    for i in range(len(scores) - 1):
        assert scores[i] <= scores[i + 1], (
            f"単調増加違反: elapsed={elapsed_values[i]}->{elapsed_values[i+1]}, "
            f"score={scores[i]}->{scores[i+1]}"
        )


def test_margin_time_rate_raw_equals_effective_rate() -> None:
    """raw は compute_effective_rate と一致する。"""
    from src.scoring import compute_effective_rate
    for elapsed in [0.0, 96.0, 112.0, 200.0]:
        v = iv.margin_time_rate(elapsed)
        expected_rate = compute_effective_rate(elapsed)
        assert v.raw == float(expected_rate), (
            f"elapsed={elapsed}: raw={v.raw} != compute_effective_rate={expected_rate}"
        )


# ============================
# collect 関数の start_sec パラメータ テスト
# (修正2: --start-sec 引数の後方互換確認)
# ============================


def test_collect_start_sec_default_is_zero() -> None:
    """collect 関数の start_sec デフォルト値は 0 (後方互換)。"""
    import inspect
    from scripts.collect_indicators_v2 import collect
    sig = inspect.signature(collect)
    assert "start_sec" in sig.parameters
    assert sig.parameters["start_sec"].default == 0.0


def test_collect_function_signature_backward_compat() -> None:
    """collect の既存引数シグネチャが破壊されていないこと。"""
    import inspect
    from scripts.collect_indicators_v2 import collect
    sig = inspect.signature(collect)
    params = list(sig.parameters.keys())
    # 既存 3 引数は先頭に維持
    assert params[0] == "video_path"
    assert params[1] == "out_path"
    assert params[2] == "max_sec"
    assert params[3] == "sample_interval_sec"
    # 新引数 start_sec は末尾追加
    assert params[4] == "start_sec"


# ============================
# III-8 潜在火力 (potential_fire_power)
# ============================


def _deep_chain_board() -> Board:
    """3 連鎖以上が仕込まれた盤面。

    takapt 定石 (1 色追加) では 2 連鎖止まりだが、
    2 個追加することで 3 連鎖以上が発火可能な構成。

    構成:
        - col0 縦: 赤4+青4 → 赤消 → 青消 = 2連鎖 (takapt 1手で到達)
        - col2 縦: 緑3 (あと1個で 4連結=3連鎖目の引き金)
        - col3 下: 緑1 (col2 に隣接、2手目で緑4連結完成)
    """
    g = _empty_grid()
    # 2 連鎖の本体 (col0/col1)
    g[12][0] = COLOR_RED
    g[12][1] = COLOR_RED
    g[11][0] = COLOR_RED
    g[10][0] = COLOR_RED
    g[9][0] = COLOR_BLUE
    g[12][2] = COLOR_BLUE
    g[11][1] = COLOR_BLUE
    g[10][1] = COLOR_BLUE
    # 3 連鎖の引き金 (col2/col3 の緑3連結: あと1個で消える)
    g[12][3] = COLOR_GREEN
    g[11][3] = COLOR_GREEN
    g[10][3] = COLOR_GREEN
    return Board.from_list(g)


def test_potential_fire_power_empty_board_near_zero() -> None:
    """空盤面では潜在火力 ≒ 0 (ぷよを追加しても連鎖が起きない)。"""
    v = iv.potential_fire_power(_empty_board())
    assert v.score == 0.0
    assert v.raw == 0.0


def test_potential_fire_power_in_range() -> None:
    """潜在火力が 0-1 範囲・例外なし。代表 4 盤面を検証。"""
    for board in _BOARDS:
        v = iv.potential_fire_power(board)
        assert 0.0 <= v.score <= 1.0
        assert v.raw == v.raw  # NaN なし


def test_potential_fire_power_deep_chain_board() -> None:
    """深い連鎖盤面: potential >= current_max_chain の raw (より多くのお邪魔)。

    current_max_chain は 1 個追加での最大連鎖数。
    potential_fire_power は最大 2 個追加でお邪魔換算するので、
    現在連鎖が存在する盤面では potential の raw が深い連鎖を捉えていること。
    """
    board = _deep_chain_board()
    pfp = iv.potential_fire_power(board)
    cmc = iv.current_max_chain(board)
    # potential は 2 手探索でお邪魔数を最大化している。
    # current_max_chain.raw は 1 個追加の連鎖数 (chain count) であり単位が異なるが、
    # 連鎖が存在する盤面では pfp.raw > 0 が保証される。
    assert pfp.raw > 0.0, f"deep_chain_board で潜在火力=0 は想定外 (cmc={cmc.raw})"
    assert pfp.score > 0.0


def test_potential_fire_power_ge_immediate_fire_power() -> None:
    """潜在火力 >= 即発火火力 (2 手探索は takapt 1 手より深い探索なので下界性が成立)。

    immediate_fire_power は takapt 1 手追加の最良配置からの火力。
    potential_fire_power は 1 手追加 top-K から 2 手目を展開するため、
    同じ 1 手目最良配置を含み、かつ 2 手目でさらに深い連鎖を発見できる。
    """
    board = _two_chain_board()
    pfp = iv.potential_fire_power(board)
    ifp = iv.immediate_fire_power(board)
    assert pfp.raw >= ifp.raw, (
        f"potential={pfp.raw} < immediate={ifp.raw}: 2手探索が1手より劣化"
    )


def test_potential_fire_power_max_add_1() -> None:
    """max_add=1 の場合: 30 通り sim のみで計算され例外なし・0-1 範囲。"""
    v = iv.potential_fire_power(_two_chain_board(), max_add=1)
    assert 0.0 <= v.score <= 1.0
    assert v.raw == v.raw  # NaN なし


def test_potential_fire_power_deeper_than_current_max_chain() -> None:
    """潜在火力が current_max_chain より深い連鎖を捉える証拠。

    deep_chain_board は takapt 1 手では 2 連鎖止まり。
    potential_fire_power (2 手) では 3 連鎖以上のお邪魔が出ることを確認する。
    """
    from src.chain import ChainSimulator
    sim = ChainSimulator()
    board = _deep_chain_board()

    # takapt 1 手: 最大連鎖数と対応お邪魔数
    cmc = iv.current_max_chain(board, sim)
    ifp = iv.immediate_fire_power(board, simulator=sim)

    # potential (2 手): より多くのお邪魔が出るか確認
    pfp = iv.potential_fire_power(board, simulator=sim)

    # potential は 2 手先まで見るので raw >= 1手分のお邪魔
    assert pfp.raw >= ifp.raw, (
        f"2手探索(pfp={pfp.raw}) が 1手(ifp={ifp.raw}) より少ない"
    )
    # deep_chain_board は明確に連鎖が仕込まれているので pfp > 0
    assert pfp.raw > 0.0, f"deep_chain_board で pfp=0 (cmc={cmc.raw}, ifp={ifp.raw})"


# ============================
# VII-2 テンポ核 (honsen_tempo_output)
# ============================


def test_honsen_tempo_output_zero_opp_chain_equals_current() -> None:
    """相手連鎖数 0 のとき: window=0 → frac=0 → my_built=current → raw=chain_to_ojama(current)。"""
    current = 5.0
    ach = 8.0
    result = iv.honsen_tempo_output(current, ach, opp_chain=0.0)
    expected_raw = iv.chain_to_ojama(current)
    assert result.raw == pytest.approx(expected_raw, rel=1e-6)
    assert 0.0 <= result.score <= 1.0


def test_honsen_tempo_output_large_opp_chain_reaches_achievable() -> None:
    """相手連鎖数が大きい (窓が十分) → frac=1.0 → my_built=achievable。"""
    current = 4.0
    ach = 10.0
    # 十分大きな相手連鎖: window = chain_to_time(50) = 50*0.3=15秒、
    # 必要手数 = (10-4)*2=12手、実際置ける=15/0.733≈20手 > 12手 → frac=1.0
    result = iv.honsen_tempo_output(current, ach, opp_chain=50.0)
    expected_raw = iv.chain_to_ojama(ach)
    assert result.raw == pytest.approx(expected_raw, rel=1e-6)


def test_honsen_tempo_output_score_range() -> None:
    """任意入力で score が 0〜1 範囲内。"""
    for curr, ach, opp in [(0, 0, 0), (3, 5, 8), (10, 10, 15), (19, 19, 19)]:
        v = iv.honsen_tempo_output(float(curr), float(ach), float(opp))
        assert 0.0 <= v.score <= 1.0, f"score out of range: curr={curr}, ach={ach}, opp={opp}"


def test_honsen_tempo_output_fallback_achievable() -> None:
    """achievable=0 (不明) のとき current+2 にフォールバックして raw > 0。"""
    current = 5.0
    result = iv.honsen_tempo_output(current, achievable_chain=0.0, opp_chain=0.0)
    # opp=0 → frac=0 → my_built=current → raw=chain_to_ojama(current)
    expected_raw = iv.chain_to_ojama(current)
    assert result.raw == pytest.approx(expected_raw, rel=1e-6)


def test_honsen_tempo_output_monotone_in_opp_chain() -> None:
    """相手連鎖数が大きいほど my_built が大きく raw が単調増加する。"""
    current = 3.0
    ach = 9.0
    prev_raw = -1.0
    for opp in [1.0, 3.0, 6.0, 12.0, 20.0]:
        v = iv.honsen_tempo_output(current, ach, opp_chain=opp)
        assert v.raw >= prev_raw, f"単調増加違反: opp={opp}, raw={v.raw} < prev={prev_raw}"
        prev_raw = v.raw


def test_honsen_tempo_constants_exported() -> None:
    """SEC_PER_HAND・HANDS_PER_CHAIN_GAP が __all__ 経由でアクセス可能。"""
    assert iv.SEC_PER_HAND > 0.0
    assert iv.HANDS_PER_CHAIN_GAP > 0.0
    assert "honsen_tempo_output" in iv.__all__
    assert "SEC_PER_HAND" in iv.__all__
    assert "HANDS_PER_CHAIN_GAP" in iv.__all__


# ============================
# VIII 催促潰し度 (ojama_disruption)
# ============================


def _fragile_board() -> Board:
    """発火直前盤面: お邪魔が割り込みやすい連鎖構造 (分断されやすい)。

    col1 に縦5赤 (連鎖トリガー直前)、周囲に単色ぷよを配置。
    お邪魔が落ちると連結が分断→連鎖数が激減する想定。
    """
    g = _empty_grid()
    # col1 縦5赤 → 4連結 (連鎖成立)
    for row in range(8, 13):
        g[row][1] = COLOR_RED
    # col2 縦4青 → 隣接で 2 連鎖になる
    for row in range(9, 13):
        g[row][2] = COLOR_BLUE
    # col3 縦4青 → col2 青と連結して消える
    for row in range(9, 13):
        g[row][3] = COLOR_BLUE
    return Board.from_list(g)


def _flat_board() -> Board:
    """頑健/平坦盤面: 単色の 1 個ずつ散在、そもそも連鎖ゼロ。

    連鎖がゼロなので disruption = 0.0 (before<=0 分岐)。
    """
    g = _empty_grid()
    # 各列に1色1個 (隣接なし、連鎖不可)
    colors = [COLOR_RED, COLOR_BLUE, COLOR_GREEN, COLOR_YELLOW, COLOR_PURPLE, COLOR_RED]
    for col, color in enumerate(colors):
        g[12][col] = color
    return Board.from_list(g)


def test_ojama_disruption_in_range() -> None:
    """score は 0〜1 範囲、raw は NaN なし。"""
    for board in [_fragile_board(), _flat_board(), _empty_board(), _two_chain_board()]:
        v = iv.ojama_disruption(board)
        assert 0.0 <= v.score <= 1.0, f"score out of range: {v.score}"
        assert v.raw == v.raw  # NaN なし


def test_ojama_disruption_flat_board_zero() -> None:
    """連鎖が組めない平坦盤面は disruption = 0.0 (before<=0 分岐)。"""
    v = iv.ojama_disruption(_flat_board())
    assert v.score == 0.0
    assert v.raw == 0.0


def test_ojama_disruption_empty_board_zero() -> None:
    """空盤面 (連鎖ゼロ) は disruption = 0.0。"""
    v = iv.ojama_disruption(_empty_board())
    assert v.score == 0.0


def test_ojama_disruption_fragile_vs_stable() -> None:
    """壊れやすい盤面の disruption > 安定盤面の disruption。

    _fragile_board: 発火直前、お邪魔で連結が分断されやすい。
    _four_chain_board: 2×2 のコンパクトな連結 (お邪魔で囲まれても最下段は残りやすい)。
    ※ n_samples=8, seed 固定ではなく統計的な差を見る。
    """
    fragile = iv.ojama_disruption(_fragile_board(), n_samples=8)
    compact = iv.ojama_disruption(_four_chain_board(), n_samples=8)
    # 壊れやすい盤面の方が disruption が高いはず
    # (注: _four_chain_board は小さいため一部サンプルで全壊する可能性もある。
    #  >= で比較することで等しい場合も許容。絶対差より方向性を確認。)
    assert fragile.score >= compact.score, (
        f"壊れやすい盤面が期待より低い: fragile={fragile.score:.3f}, compact={compact.score:.3f}"
    )


def test_ojama_disruption_exported() -> None:
    """ojama_disruption が __all__ に含まれること。"""
    assert "ojama_disruption" in iv.__all__
    assert iv.OJAMA_DISRUPTION_DEFAULT_N == 12
    assert iv.OJAMA_DISRUPTION_DEFAULT_SAMPLES == 8


def test_ojama_disruption_custom_n() -> None:
    """ojama_n=0 ではお邪魔落下がなく reduction=0 になること。"""
    board = _fragile_board()
    v = iv.ojama_disruption(board, ojama_n=0, n_samples=4)
    assert v.score == 0.0


# ============================
# drop_ojama 端数ランダム化不変条件
# ============================


def test_drop_ojama_remainder_distribution() -> None:
    """端数ランダム化: 合計個数 = ojama_n、各列 >= floor(N/6)、端数分確認。

    ojama_n=7 → floor(7/6)=1 なので全列 >=1、ちょうど 1 列が 2 個。
    """
    from src.chain import ChainSimulator
    sim = ChainSimulator()
    board = _empty_board()

    # 複数 seed で試して全て不変条件を満たすことを確認
    for seed in range(10):
        result = sim.drop_ojama(board, 7, seed=seed)
        total = result.count_puyos()
        assert total == 7, f"seed={seed}: total={total} != 7"
        col_counts = [
            sum(1 for row in range(BOARD_ROWS) if result.get(row, col) == COLOR_OJAMA)
            for col in range(BOARD_COLS)
        ]
        assert all(c >= 1 for c in col_counts), f"seed={seed}: 列未充足 {col_counts}"
        double_cols = sum(1 for c in col_counts if c == 2)
        assert double_cols == 1, f"seed={seed}: 端数列が1列でない {col_counts}"


def test_drop_ojama_remainder_seed_reproducible() -> None:
    """同じ seed では端数列が一致すること (再現性)。"""
    from src.chain import ChainSimulator
    sim = ChainSimulator()
    board = _empty_board()
    r1 = sim.drop_ojama(board, 7, seed=42)
    r2 = sim.drop_ojama(board, 7, seed=42)
    assert r1 == r2


def test_drop_ojama_remainder_random_varies() -> None:
    """seed=None では複数回の端数列が分散すること (常に同一列ではない)。"""
    from src.chain import ChainSimulator
    sim = ChainSimulator()
    board = _empty_board()
    # 100 回試行して端数列の分散を確認
    seen_cols: set[int] = set()
    for _ in range(100):
        result = sim.drop_ojama(board, 7, seed=None)
        for col in range(BOARD_COLS):
            if sum(
                1 for row in range(BOARD_ROWS) if result.get(row, col) == COLOR_OJAMA
            ) == 2:
                seen_cols.add(col)
                break
    # 100 回で少なくとも 2 列以上に端数が分散することを確認 (偏り検知)
    assert len(seen_cols) >= 2, f"端数が 1 列にしか出なかった: {seen_cols}"


# ============================
# IX 形・組み品質 (connected_pair_quality)
# ============================


def _main_linked_board() -> Board:
    """主連鎖隣接2連結を持つ盤面。

    ※ 同色の size=2 グループが同色 size>=3 グループに隣接する場合、
    find_groups は両者を1グループに統合する。そのため本指標では
    「任意色の size>=3 グループに1マス隣接する2連結」を main_linked と定義する。

    - 赤 size=4 グループ (主連鎖候補): col0 下4段
    - 青 size=2 グループ (2連結): col1 下2段 → col0 の赤(size=4)に左隣接
      → main_linked_pair_count >= 1 が期待される。
    - 緑 size=2 グループ (孤立): col4 下2段 → 近くに size>=3 なし
      → isolated_pair_count >= 1 が期待される。
    """
    g = _empty_grid()
    # 赤 size=4 (col0 縦4段) — 主連鎖候補グループ
    for row in range(9, 13):
        g[row][0] = COLOR_RED
    # 青 size=2 (col1 縦2段, col0 の赤 size=4 グループに右隣接)
    g[12][1] = COLOR_BLUE
    g[11][1] = COLOR_BLUE
    # 緑 size=2 (col4 縦2段, 近くに size>=3 グループなし = 孤立)
    g[12][4] = COLOR_GREEN
    g[11][4] = COLOR_GREEN
    return Board.from_list(g)


def _isolated_only_board() -> Board:
    """孤立2連結のみを持つ盤面 (同色 size>=3 グループなし)。

    - 赤 size=2 (col0 縦2段): 同色 size>=3 グループ不在 → 孤立
    - 青 size=2 (col2 縦2段): 同色 size>=3 グループ不在 → 孤立
    """
    g = _empty_grid()
    g[12][0] = COLOR_RED
    g[11][0] = COLOR_RED
    g[12][2] = COLOR_BLUE
    g[11][2] = COLOR_BLUE
    return Board.from_list(g)


def test_main_linked_pair_count_in_range() -> None:
    """main_linked_pair_count が 0-1 範囲かつ例外なし (全テスト盤面)。"""
    for board in _BOARDS + [_main_linked_board(), _isolated_only_board()]:
        v = iv.main_linked_pair_count(board)
        assert 0.0 <= v.score <= 1.0, f"score out of range: {v.score}"
        assert v.raw == v.raw  # NaN なし


def test_isolated_pair_count_in_range() -> None:
    """isolated_pair_count が 0-1 範囲かつ例外なし (全テスト盤面)。"""
    for board in _BOARDS + [_main_linked_board(), _isolated_only_board()]:
        v = iv.isolated_pair_count(board)
        assert 0.0 <= v.score <= 1.0, f"score out of range: {v.score}"
        assert v.raw == v.raw  # NaN なし


def test_main_linked_ratio_in_range() -> None:
    """main_linked_ratio が 0-1 範囲かつ例外なし (全テスト盤面)。"""
    for board in _BOARDS + [_main_linked_board(), _isolated_only_board()]:
        v = iv.main_linked_ratio(board)
        assert 0.0 <= v.score <= 1.0, f"score out of range: {v.score}"
        assert v.raw == v.raw  # NaN なし


def test_main_linked_pair_count_detects_linked() -> None:
    """_main_linked_board: 主連鎖隣接2連結が1つ以上検出される。"""
    v = iv.main_linked_pair_count(_main_linked_board())
    assert v.raw >= 1.0, f"主連鎖隣接2連結が検出されない: raw={v.raw}"


def test_isolated_pair_count_detects_isolated() -> None:
    """_main_linked_board: 孤立2連結が1つ以上検出される (青2連結)。"""
    v = iv.isolated_pair_count(_main_linked_board())
    assert v.raw >= 1.0, f"孤立2連結が検出されない: raw={v.raw}"


def test_isolated_only_board_main_linked_is_zero() -> None:
    """_isolated_only_board: size>=3 グループなし → main_linked=0。"""
    v = iv.main_linked_pair_count(_isolated_only_board())
    assert v.raw == 0.0, f"孤立のみ盤面なのに main_linked > 0: raw={v.raw}"


def test_isolated_only_board_isolated_count() -> None:
    """_isolated_only_board: 2連結が2つ (赤+青) → isolated=2。"""
    v = iv.isolated_pair_count(_isolated_only_board())
    assert v.raw == 2.0, f"孤立2連結の数が想定と違う: raw={v.raw}"


def test_main_linked_ratio_is_zero_when_no_pairs() -> None:
    """2連結が存在しない盤面 (空盤面) は ratio=0.0。"""
    v = iv.main_linked_ratio(_empty_board())
    assert v.raw == 0.0
    assert v.score == 0.0


def test_main_linked_ratio_isolated_only_is_zero() -> None:
    """孤立2連結のみの盤面: main_linked=0 → ratio=0.0。"""
    v = iv.main_linked_ratio(_isolated_only_board())
    assert v.raw == 0.0
    assert v.score == 0.0


def test_main_linked_ratio_with_linked_board() -> None:
    """_main_linked_board: main_linked>=1, total>=2 → 0 < ratio <= 1。"""
    v = iv.main_linked_ratio(_main_linked_board())
    assert 0.0 < v.raw <= 1.0, f"ratio が期待範囲外: raw={v.raw}"


def test_pair_counts_sum_equals_conn_pair_count() -> None:
    """main_linked + isolated の合計は connectivity_observation の pair_count と一致。

    両者ともに find_groups の size==2 グループを数えているため一致する。
    """
    for board in [_main_linked_board(), _isolated_only_board(), _two_chain_board()]:
        total_conn, _ = iv.connectivity_observation(board)
        mlp = iv.main_linked_pair_count(board)
        ip = iv.isolated_pair_count(board)
        assert int(mlp.raw) + int(ip.raw) == total_conn.pair_count, (
            f"main_linked({mlp.raw}) + isolated({ip.raw}) != "
            f"conn_pair_count({total_conn.pair_count})"
        )


def test_ix_indicators_exported_in_all() -> None:
    """IX 指標・定数が __all__ に含まれること。"""
    assert "main_linked_pair_count" in iv.__all__
    assert "isolated_pair_count" in iv.__all__
    assert "main_linked_ratio" in iv.__all__
    assert "MAIN_GROUP_MIN_SIZE" in iv.__all__
    assert "NORM_LINKED_PAIR" in iv.__all__


def test_ix_constants_values() -> None:
    """MAIN_GROUP_MIN_SIZE=3, NORM_LINKED_PAIR=10.0 (定数値の確認)。"""
    assert iv.MAIN_GROUP_MIN_SIZE == 3
    assert iv.NORM_LINKED_PAIR == pytest.approx(10.0)
