"""exchange_virtual_board.py のテスト

発火イベントから仮想盤面ペアを再構成する純関数群 (Step2) を検証する。
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
    DEATH_COL,
    DEATH_ROW,
    Board,
)
from src.chain import ChainSimulator
from src.exchange_virtual_board import (
    VirtualBoardPair,
    reconstruct_virtual_board_pair,
)


# ============================
# テスト用ヘルパー (tests/test_chain.py と同じ方式)
# ============================


def empty_grid() -> list[list[int]]:
    """13×6 の全空グリッドを生成する。"""
    return [[COLOR_EMPTY] * BOARD_COLS for _ in range(BOARD_ROWS)]


def board_from_grid(grid: list[list[int]]) -> Board:
    """グリッドリストから Board を生成する。"""
    return Board.from_list(grid)


def place_horizontal(
    grid: list[list[int]], row: int, col_start: int, color: int, count: int,
) -> None:
    """指定行に color のぷよを count 個横に並べる (in-place)。"""
    for col in range(col_start, col_start + count):
        grid[row][col] = color


def make_4connect_board() -> Board:
    """最下段に赤4個 (1連鎖確定) を並べた盤面を返す。"""
    grid = empty_grid()
    place_horizontal(grid, BOARD_ROWS - 1, 0, COLOR_RED, 4)
    return board_from_grid(grid)


def make_no_chain_board() -> Board:
    """連鎖が起きない (4連結なし) 盤面を返す。"""
    grid = empty_grid()
    grid[BOARD_ROWS - 1][0] = COLOR_RED
    grid[BOARD_ROWS - 1][1] = COLOR_BLUE
    grid[BOARD_ROWS - 1][2] = COLOR_GREEN
    return board_from_grid(grid)


def make_dead_board() -> Board:
    """窒息判定 (Board.is_dead()) が True になる最小盤面を返す

    (DEATH_ROW, DEATH_COL の1マスだけおじゃまを置く。連鎖対象にはならない
    色 (おじゃま) を使うため simulate しても消えない)。
    """
    grid = empty_grid()
    grid[DEATH_ROW][DEATH_COL] = COLOR_OJAMA
    return board_from_grid(grid)


def make_near_death_board() -> Board:
    """窒息一歩手前 (DEATH_ROW の DEATH_COL 以外は全埋まり) の盤面を返す。

    DEATH_ROW より下 (row=DEATH_ROW+1〜BOARD_ROWS-1) は全列おじゃまで満杯、
    DEATH_ROW 自体は DEATH_COL 以外の列を埋める。この状態でおじゃまを
    ちょうど BOARD_COLS 個 (=6個、floor(6/6)=1・端数0で列選択の乱数に
    依存しない) 落とすと、DEATH_COL の落下先は必ず DEATH_ROW になり
    決定論的に窒息する (他列の落下先は隠し段 row0)。
    """
    grid = empty_grid()
    for row in range(DEATH_ROW + 1, BOARD_ROWS):
        for col in range(BOARD_COLS):
            grid[row][col] = COLOR_OJAMA
    for col in range(BOARD_COLS):
        if col != DEATH_COL:
            grid[DEATH_ROW][col] = COLOR_OJAMA
    return board_from_grid(grid)


# ============================
# reconstruct_virtual_board_pair
# ============================


def test_attacker_board_matches_direct_simulate():
    """attacker_board_after は ChainSimulator.simulate の final_board と一致する。"""
    before = make_4connect_board()
    opp = Board()
    result = reconstruct_virtual_board_pair(before, opp, net_ojama_after_pred=0.0)
    expected = ChainSimulator().simulate(before).final_board
    assert result.attacker_board_after == expected


def test_no_chain_attacker_after_equals_before():
    """連鎖が起きない盤面では attacker_board_after == before (消去なし)。"""
    before = make_no_chain_board()
    opp = Board()
    result = reconstruct_virtual_board_pair(before, opp, net_ojama_after_pred=0.0)
    assert result.attacker_board_after == before
    assert result.chain_result.chain_count == 0


def test_zero_net_ojama_no_change_to_opponent():
    """net_ojama_after_pred=0 なら相手盤面は変化しない。"""
    before = make_4connect_board()
    opp = make_no_chain_board()
    result = reconstruct_virtual_board_pair(before, opp, net_ojama_after_pred=0.0)
    assert result.opponent_board_after == opp
    assert result.ojama_to_opponent == 0
    assert result.ojama_to_attacker == 0


def test_positive_net_ojama_lands_on_opponent():
    """net_ojama_after_pred が正なら相手側に着弾し、攻撃側には降らない。"""
    before = make_4connect_board()
    opp = Board()
    result = reconstruct_virtual_board_pair(before, opp, net_ojama_after_pred=6.0)
    assert result.ojama_to_opponent == 6
    assert result.ojama_to_attacker == 0
    n_ojama = sum(
        1 for r in range(BOARD_ROWS) for c in range(BOARD_COLS)
        if result.opponent_board_after.get(r, c) == COLOR_OJAMA
    )
    assert n_ojama == 6


def test_negative_net_ojama_lands_on_attacker():
    """net_ojama_after_pred が負なら攻撃側 (連鎖消化後盤面) に着弾する。"""
    before = make_4connect_board()
    opp = Board()
    result = reconstruct_virtual_board_pair(before, opp, net_ojama_after_pred=-5.0)
    assert result.ojama_to_attacker == 5
    assert result.ojama_to_opponent == 0
    assert result.opponent_board_after == opp  # 相手盤面は無変化


def test_ojama_count_rounding():
    """net_ojama_after_pred は四捨五入して整数個着弾させる。"""
    before = make_4connect_board()
    opp = Board()
    result_low = reconstruct_virtual_board_pair(before, opp, net_ojama_after_pred=3.4)
    result_high = reconstruct_virtual_board_pair(before, opp, net_ojama_after_pred=3.6)
    assert result_low.ojama_to_opponent == 3
    assert result_high.ojama_to_opponent == 4


def test_nan_raises_value_error():
    """NaN は silent fallback せず ValueError を送出する。"""
    before = make_4connect_board()
    opp = Board()
    with pytest.raises(ValueError):
        reconstruct_virtual_board_pair(before, opp, net_ojama_after_pred=float("nan"))


def test_deterministic_seed_reproducible():
    """同一入力に対して端数列配置を含め完全に同一の結果が再現される。"""
    before = make_4connect_board()
    opp = Board()
    r1 = reconstruct_virtual_board_pair(before, opp, net_ojama_after_pred=7.0)
    r2 = reconstruct_virtual_board_pair(before, opp, net_ojama_after_pred=7.0)
    assert r1.opponent_board_after == r2.opponent_board_after


def test_attacker_dead_flag_true():
    """攻撃側が連鎖消化後も窒息状態なら attacker_dead=True。"""
    before = make_dead_board()
    assert before.is_dead() is True  # 事前条件確認
    opp = Board()
    result = reconstruct_virtual_board_pair(before, opp, net_ojama_after_pred=0.0)
    assert result.chain_result.chain_count == 0  # おじゃま単体は連鎖対象外
    assert result.attacker_dead is True


def test_opponent_dead_flag_after_large_ojama_drop():
    """相手盤面が窒息一歩手前の状態でおじゃま6個 (floor(6/6)=1・端数0で

    列選択の乱数に依存せず決定論的) を受けると opponent_dead=True になる。
    """
    before = make_no_chain_board()
    opp = make_near_death_board()
    assert opp.is_dead() is False  # 事前条件確認: まだ窒息していない
    result = reconstruct_virtual_board_pair(before, opp, net_ojama_after_pred=6.0)
    assert result.opponent_dead is True


def test_large_net_ojama_capped_at_max_drop_per_turn():
    """2026-08-03 指摘 欠陥E-1: 448個等の大型予測値でも実配置は30個 (1ターン上限) まで。"""
    from src.exchange_virtual_board import OJAMA_MAX_DROP_PER_TURN

    before = make_no_chain_board()
    opp = Board()  # 空盤面 (十分な空きあり)
    result = reconstruct_virtual_board_pair(before, opp, net_ojama_after_pred=448.0)
    assert result.ojama_to_opponent == OJAMA_MAX_DROP_PER_TURN
    placed = sum(1 for r in range(BOARD_ROWS) for c in range(BOARD_COLS)
                if result.opponent_board_after.get(r, c) == COLOR_OJAMA)
    assert placed == OJAMA_MAX_DROP_PER_TURN


def test_large_net_ojama_does_not_force_instant_death_on_spacious_board():
    """main実測 match_02 の再現: 448個予測でも空きが十分あれば即死判定にならない
    (旧実装は全量投下で単独盤面容量超過→不当に opponent_dead=True になっていた)。
    """
    before = make_no_chain_board()
    opp = Board()  # 空盤面、30個程度なら窒息しない
    result = reconstruct_virtual_board_pair(before, opp, net_ojama_after_pred=448.0)
    assert result.opponent_dead is False


def test_capped_and_exact_max_drop_produce_identical_result():
    """448個予測と30個予測(=上限そのもの)の結果が一致する (上限適用の一貫性)。"""
    from src.exchange_virtual_board import OJAMA_MAX_DROP_PER_TURN

    before = make_no_chain_board()
    opp = Board()
    result_large = reconstruct_virtual_board_pair(before, opp, net_ojama_after_pred=448.0)
    result_exact = reconstruct_virtual_board_pair(
        before, opp, net_ojama_after_pred=float(OJAMA_MAX_DROP_PER_TURN))
    assert result_large.opponent_board_after == result_exact.opponent_board_after


def test_small_net_ojama_below_cap_unaffected():
    """30個以下の予測は従来通り全量配置される (後方互換、既存挙動を破壊しない)。"""
    before = make_no_chain_board()
    opp = Board()
    result = reconstruct_virtual_board_pair(before, opp, net_ojama_after_pred=20.0)
    assert result.ojama_to_opponent == 20


def test_custom_simulator_reused():
    """呼び出し側が渡した ChainSimulator インスタンスでも同じ結果になる。"""
    before = make_4connect_board()
    opp = Board()
    shared_sim = ChainSimulator()
    result = reconstruct_virtual_board_pair(
        before, opp, net_ojama_after_pred=0.0, simulator=shared_sim,
    )
    expected = shared_sim.simulate(before).final_board
    assert result.attacker_board_after == expected


def test_returns_virtual_board_pair_instance():
    """戻り値の型が VirtualBoardPair であること。"""
    before = make_no_chain_board()
    opp = Board()
    result = reconstruct_virtual_board_pair(before, opp, net_ojama_after_pred=0.0)
    assert isinstance(result, VirtualBoardPair)
