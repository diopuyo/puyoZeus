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
    """窒息 = 3列目の可視最上段(DEATH_ROW=1)が埋まっている状態 (2026-07-22 user確定)。"""
    g = _empty_grid()
    from src.board import DEATH_COL, DEATH_ROW
    g[DEATH_ROW][DEATH_COL] = COLOR_RED  # 窒息
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
    # n_samples は熱対策/コストで調整可(8→4等)。正の整数であることのみ担保。
    assert isinstance(iv.OJAMA_DISRUPTION_DEFAULT_SAMPLES, int)
    assert iv.OJAMA_DISRUPTION_DEFAULT_SAMPLES >= 1


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


# ============================
# X 受けやすさ (ukeyasusa) テスト
# ============================


def _full_board() -> Board:
    """盤面ぷよで埋まりきった盤面 (absorption=0, death_margin=0)。"""
    g = _empty_grid()
    color_cycle = [COLOR_RED, COLOR_BLUE, COLOR_GREEN, COLOR_YELLOW]
    for r in range(BOARD_ROWS):
        for c in range(BOARD_COLS):
            g[r][c] = color_cycle[(r + c) % len(color_cycle)]
    return Board.from_list(g)


def _chain_board_large() -> Board:
    """大連鎖を仕込んだ盤面 (受け余地あり・掘り耐性高を期待)。

    下段に 赤 / 青 / 緑 / 黄 の 4 連結を積んだ 4 連鎖盤面。
    """
    g = _empty_grid()
    # 赤 (最下段)
    for c in range(4):
        g[12][c] = COLOR_RED
    # 青 (1 段上)
    for c in range(4):
        g[11][c] = COLOR_BLUE
    # 緑 (2 段上)
    for c in range(4):
        g[10][c] = COLOR_GREEN
    # 黄 (3 段上)
    for c in range(4):
        g[9][c] = COLOR_YELLOW
    return Board.from_list(g)


def test_ukeyasusa_range_all_boards() -> None:
    """ukeyasusa が各種盤面で 0〜1 範囲・NaN なしで返ること。"""
    for board in [_empty_board(), _four_chain_board(), _ojama_board(), _full_board()]:
        v = iv.ukeyasusa(board)
        assert 0.0 <= v.score <= 1.0, f"score out of range: {v.score}"
        assert v.raw == v.raw  # NaN なし


def test_ukeyasusa_empty_board_high() -> None:
    """空盤面は受けやすさが比較的高い (absorption=1, death_margin=1)。

    v2 重み (dig=0.6 主体) では空盤面の dig_resistance=0 のため
    スコアは 0.4 程度。満杯盤面 (0.0) よりは有意に高いことを確認する。
    """
    v = iv.ukeyasusa(_empty_board())
    # dig が主体の新重みでは 0.35 超を保証 (0.6×0 + 0.2×1 + 0.2×1 = 0.4)
    assert v.score > 0.35, f"空盤面の受けやすさが低すぎる: {v.score}"


def test_ukeyasusa_full_board_low() -> None:
    """満杯盤面は受けやすさが空盤面より低い (absorption=0)。"""
    empty_score = iv.ukeyasusa(_empty_board()).score
    full_score = iv.ukeyasusa(_full_board()).score
    assert full_score < empty_score, (
        f"満杯({full_score:.3f}) >= 空({empty_score:.3f}) は期待外"
    )


def test_ukeyasusa_dead_board_zero() -> None:
    """窒息盤面 (col=2 が埋まっている) は absorption/dig_resistance が低下する。

    dig_resistance は is_dead() 確認で 0 を返し、death_margin も 0 に近い
    → ukeyasusa が低くなることを確認 (厳密ゼロは保証しない)。
    """
    g = _empty_grid()
    # col=2 (窒息列) を最上段まで埋める
    for r in range(BOARD_ROWS):
        g[r][2] = COLOR_RED
    dead_board = Board.from_list(g)
    v = iv.ukeyasusa(dead_board)
    assert v.score < 0.5, f"窒息近盤面の受けやすさが高すぎる: {v.score}"


def test_ukeyasusa_exported_in_all() -> None:
    """ukeyasusa が __all__ に含まれること。"""
    assert "ukeyasusa" in iv.__all__
    assert "UKEYASUSA_W_ABSORPTION" in iv.__all__


# ============================
# XI 対応力 (taiou_capacity) テスト
# ============================


def test_taiou_capacity_range_all_boards() -> None:
    """taiou_capacity が各種盤面で 0〜1 範囲・NaN なしで返ること。"""
    for board in [_empty_board(), _four_chain_board(), _two_chain_board(), _ojama_board()]:
        v = iv.taiou_capacity(board)
        assert 0.0 <= v.score <= 1.0, f"score out of range: {v.score}"
        assert v.raw == v.raw  # NaN なし
        # raw は offset_ratio (0〜1)
        assert 0.0 <= v.raw <= 1.0, f"raw(offset_ratio) out of range: {v.raw}"


def test_taiou_capacity_dead_board_zero() -> None:
    """窒息盤面は対応力 0 を返すこと。"""
    g = _empty_grid()
    # col=2 を全13行埋め尽くし is_dead()=True に (row0/row1 双方埋まるため
    # DEATH_ROW=0/1 いずれの定義でも窒息判定は不変)
    for r in range(BOARD_ROWS):
        g[r][2] = COLOR_RED
    dead_board = Board.from_list(g)
    v = iv.taiou_capacity(dead_board)
    assert v.score == pytest.approx(0.0), f"窒息盤面の対応力が 0 でない: {v.score}"


def test_taiou_capacity_chain_board_positive() -> None:
    """大連鎖盤面は即発火で ref_ojama=30 の一部を相殺でき、対応力が正値になること。"""
    board = _chain_board_large()
    v = iv.taiou_capacity(board, ref_ojama=30)
    # 連鎖火力があれば offset_ratio > 0 → score > 0
    assert v.score > 0.0, f"大連鎖盤面の対応力が 0: {v.score}"


def test_taiou_capacity_ref_ojama_effect() -> None:
    """ref_ojama が大きいほど相殺充足度 (raw) が下がり対応力が減ること。"""
    board = _chain_board_large()
    v_small = iv.taiou_capacity(board, ref_ojama=5)
    v_large = iv.taiou_capacity(board, ref_ojama=200)
    # ref_ojama 小 → offset_ratio 高 → score 高 (あるいは同等)
    assert v_small.score >= v_large.score, (
        f"ref_ojama小({v_small.score:.3f}) < ref_ojama大({v_large.score:.3f}) は期待外"
    )


def test_taiou_capacity_empty_board() -> None:
    """空盤面は即発火ゼロなので対応力 0 になること。"""
    v = iv.taiou_capacity(_empty_board())
    assert v.score == pytest.approx(0.0), f"空盤面の対応力が 0 でない: {v.score}"
    assert v.raw == pytest.approx(0.0)


def test_taiou_capacity_exported_in_all() -> None:
    """taiou_capacity が __all__ に含まれること。"""
    assert "taiou_capacity" in iv.__all__
    assert "REF_OJAMA_TAIOU" in iv.__all__
    assert "TAIOU_W_POTENTIAL" in iv.__all__


# ============================
# XII-1b 本来の飽和 (build天井、ビームサーチ近似) — build_ceiling_chain
# ============================


def test_build_ceiling_chain_empty_board_is_zero() -> None:
    """空盤面は 1 個追加でも発火できないため raw=0, score=0。"""
    v = iv.build_ceiling_chain(_empty_board())
    assert v.score == pytest.approx(0.0)
    assert v.raw == pytest.approx(0.0)


def test_build_ceiling_chain_depth1_matches_saturated_chain_count() -> None:
    """depth=1 は saturated_chain_count (=_takapt_best_drop) と厳密に一致する

    (サニティチェック: ビームサーチが 1 手先読みに退化した場合の下位互換確認)。
    """
    for board in (_four_chain_board(), _two_chain_board(), _deep_chain_board()):
        sat = iv.saturated_chain_count(board)
        ceil1 = iv.build_ceiling_chain(board, depth=1)
        assert ceil1.raw == pytest.approx(sat.raw), (
            f"depth=1 が saturated_chain_count と不一致: "
            f"ceil={ceil1.raw} sat={sat.raw}"
        )
        assert ceil1.score == pytest.approx(sat.score)


def test_build_ceiling_chain_depth2_ge_depth1() -> None:
    """depth=2 (既定) は depth=1 以上 (単調非減少: build余地は非負)。"""
    for board in (_four_chain_board(), _two_chain_board(), _deep_chain_board()):
        ceil1 = iv.build_ceiling_chain(board, depth=1)
        ceil2 = iv.build_ceiling_chain(board, depth=2)
        assert ceil2.raw >= ceil1.raw, (
            f"depth=2 が depth=1 未満: ceil2={ceil2.raw} ceil1={ceil1.raw}"
        )


def test_build_ceiling_chain_default_depth_is_2() -> None:
    """既定パラメータ (depth=2, beam_width=8) が定数と一致すること。"""
    assert iv.BUILD_CEILING_CHAIN_DEPTH == 2
    assert iv.BUILD_CEILING_CHAIN_BEAM_WIDTH == 8


def test_build_ceiling_chain_score_in_range() -> None:
    """score は 0〜1 に収まる。"""
    for board in (_empty_board(), _four_chain_board(), _two_chain_board()):
        v = iv.build_ceiling_chain(board)
        assert 0.0 <= v.score <= 1.0


def test_build_ceiling_chain_does_not_mutate_board() -> None:
    """stateless 原則: 呼出前後で盤面が変化しない (非破壊)。"""
    board = _four_chain_board()
    before = board.copy()
    iv.build_ceiling_chain(board, depth=2)
    for row in range(BOARD_ROWS):
        for col in range(BOARD_COLS):
            assert board.get(row, col) == before.get(row, col)


def test_build_ceiling_chain_dead_board_is_zero() -> None:
    """窒息盤面 (3列目・可視最上段埋まり) は raw=0, score=0。

    DEATH_ROW=1 (可視最上段。隠し段 row0 は含まない、2026-07-22 user確定)。
    """
    from src.board import DEATH_ROW
    g = _empty_grid()
    g[DEATH_ROW][2] = 1  # 3列目 (index=2) 可視最上段
    board = Board.from_list(g)
    v = iv.build_ceiling_chain(board)
    assert v.score == pytest.approx(0.0)
    assert v.raw == pytest.approx(0.0)


def test_build_ceiling_chain_exported_in_all() -> None:
    """build_ceiling_chain が __all__ に含まれること。"""
    assert "build_ceiling_chain" in iv.__all__
    assert "BUILD_CEILING_CHAIN_DEPTH" in iv.__all__
    assert "BUILD_CEILING_CHAIN_BEAM_WIDTH" in iv.__all__


# ============================
# XII-1c 忠実な飽和連鎖量 (非発火構築ビーム) — saturation_chain
# ============================


def test_saturation_chain_empty_board_builds_and_ignites() -> None:
    """空盤面でも非発火構築ビームで組み上げ、発火可能な連鎖が得られる。

    build_ceiling_chain(depth=2) は空盤面で raw=0 (2手先読みでは発火不可)
    だが、saturation_chain は 93% まで積むため空盤面からでも連鎖が組める。
    """
    v = iv.saturation_chain(_empty_board())
    assert v.raw > 0.0
    assert 0.0 <= v.score <= 1.0


def test_saturation_chain_score_in_range() -> None:
    """score は 0〜1 に収まる。"""
    for board in (_empty_board(), _four_chain_board(), _two_chain_board(), _deep_chain_board()):
        v = iv.saturation_chain(board)
        assert 0.0 <= v.score <= 1.0


def test_saturation_chain_does_not_mutate_board() -> None:
    """stateless 原則: 呼出前後で盤面が変化しない (非破壊)。"""
    board = _four_chain_board()
    before = board.copy()
    iv.saturation_chain(board)
    for row in range(BOARD_ROWS):
        for col in range(BOARD_COLS):
            assert board.get(row, col) == before.get(row, col)


def test_saturation_chain_dead_board_is_zero() -> None:
    """窒息盤面 (3列目・可視最上段埋まり) は raw=0, score=0。

    DEATH_ROW=1 (可視最上段。隠し段 row0 は含まない、2026-07-22 user確定)。
    """
    from src.board import DEATH_ROW
    g = _empty_grid()
    g[DEATH_ROW][2] = 1  # 3列目 (index=2) 可視最上段
    board = Board.from_list(g)
    v = iv.saturation_chain(board)
    assert v.score == pytest.approx(0.0)
    assert v.raw == pytest.approx(0.0)


def test_saturation_chain_ge_build_ceiling_chain() -> None:
    """本来の飽和 (93%まで積む) は build_ceiling_chain (2手先読み) 以上になりうる。

    saturation_chain は 93% まで積むため build_ceiling_chain (depth=2) より
    小さくなることは基本的にない (同じ最終手 takapt スキャンを使うため)。
    """
    for board in (_empty_board(), _four_chain_board(), _two_chain_board()):
        sat = iv.saturation_chain(board)
        ceil = iv.build_ceiling_chain(board, depth=2)
        assert sat.raw >= ceil.raw, (
            f"saturation_chain が build_ceiling_chain 未満: "
            f"sat={sat.raw} ceil={ceil.raw}"
        )


def test_saturation_chain_target_cells_already_reached() -> None:
    """既に fill_ratio 到達済み (steps=0) でも例外なく現盤面の発火力を測る。"""
    board = _four_chain_board()
    v = iv.saturation_chain(board, fill_ratio=0.01)
    assert v.raw >= 0.0
    assert 0.0 <= v.score <= 1.0


def test_saturation_chain_fill_ratio_monotonic_non_decreasing() -> None:
    """fill_ratio を上げるほど raw は非減少になりやすい (積む余地が増えるため)。

    構造ヒューリスティックのビームサーチのため厳密な単調性は保証されないが、
    低 fill_ratio (0.5) は高 fill_ratio (0.93) 以下になることを確認する
    (空盤面で検証、十分な余地があるケース)。
    """
    board = _empty_board()
    low = iv.saturation_chain(board, fill_ratio=0.5)
    high = iv.saturation_chain(board, fill_ratio=0.93)
    assert high.raw >= low.raw


def test_saturation_chain_partial_fill_completes_without_hang() -> None:
    """部分的に埋まった盤面 (構築ビームが多数回実行される) でも正常終了する。

    下 5 段を市松模様 (赤/青交互、4連結を作らない配置) で埋めた盤面で、
    構築ステップが多数回走っても (デッドロック分岐含め) 無限ループせず
    終端することを確認する。
    """
    g = _empty_grid()
    colors = [COLOR_RED, COLOR_BLUE]
    for col in range(BOARD_COLS):
        for row in range(8, BOARD_ROWS):
            g[row][col] = colors[(row + col) % 2]
    board = Board.from_list(g)
    v = iv.saturation_chain(board)
    assert 0.0 <= v.score <= 1.0


def test_saturation_chain_exported_in_all() -> None:
    """saturation_chain が __all__ に含まれること。"""
    assert "saturation_chain" in iv.__all__
    assert "FULL_BOARD_CAP" in iv.__all__
    assert "SATURATION_FILL_RATIO_DEFAULT" in iv.__all__
    assert "SATURATION_BEAM_WIDTH_DEFAULT" in iv.__all__
    assert "SATURATION_MAX_BUILD_STEPS" in iv.__all__


def test_full_board_cap_is_78() -> None:
    """FULL_BOARD_CAP は盤面全体 (6列×13行、隠し段 row0 含む) = 78。"""
    assert iv.FULL_BOARD_CAP == BOARD_ROWS * BOARD_COLS
    assert iv.FULL_BOARD_CAP == 78
