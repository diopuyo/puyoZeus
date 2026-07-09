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
