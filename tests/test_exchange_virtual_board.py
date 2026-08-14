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


# ============================
# resolve_mutual_exchange (2026-08-13、デモレビュー #9 対処)
# ============================


def test_cancel_mutual_pending_own_cancel_then_surplus_crosses():
    """自分の生成量で自分の予告を相殺→余剰が相手に交差する。"""
    from src.exchange_virtual_board import _cancel_mutual_pending

    # p1 生成10、p1への予告6 (自分の生成で自分への予告を相殺→残り0・余剰4はp2へ)。
    # p2 生成3、p2への予告5 (自分の生成で自分への予告を相殺→残り2・余剰0)。
    final_p1, final_p2 = _cancel_mutual_pending(
        gen_p1_ojama=10, gen_p2_ojama=3, pending_p1=6, pending_p2=5)
    assert final_p1 == 0       # p1の相殺残り0(own1) + p2発の余剰0(surplus2)
    assert final_p2 == 6       # p2の相殺残り2(own2) + p1発の余剰4(surplus1)


def test_cancel_mutual_pending_no_pending_passes_gen_through():
    """予告が0なら、生成量がそのまま相手への着弾量になる (通常の一方向と同じ)。"""
    from src.exchange_virtual_board import _cancel_mutual_pending

    final_p1, final_p2 = _cancel_mutual_pending(
        gen_p1_ojama=10, gen_p2_ojama=0, pending_p1=0, pending_p2=0)
    assert final_p1 == 0
    assert final_p2 == 10


def test_resolve_mutual_exchange_simulates_both_sides_independently():
    """両側とも自分の連鎖を消化した final_board が使われる。"""
    from src.exchange_virtual_board import resolve_mutual_exchange

    before1 = make_4connect_board()
    before2 = make_no_chain_board()
    result = resolve_mutual_exchange(
        before1, before2, gen_p1_ojama=0, gen_p2_ojama=0,
        pending_p1=0, pending_p2=0,
    )
    assert result.board_p1_after == ChainSimulator().simulate(before1).final_board
    assert result.board_p2_after == before2  # 連鎖なし=消化前と同一


def test_resolve_mutual_exchange_mutual_fire_lands_on_both():
    """両者とも生成量>0・予告0 の場合、双方に相手発のおじゃまが着弾する。"""
    from src.exchange_virtual_board import resolve_mutual_exchange

    before1 = make_no_chain_board()
    before2 = make_no_chain_board()
    result = resolve_mutual_exchange(
        before1, before2, gen_p1_ojama=6, gen_p2_ojama=12,
        pending_p1=0, pending_p2=0,
    )
    assert result.dropped_to_p1 == 12  # p2発の12個がp1へ
    assert result.dropped_to_p2 == 6   # p1発の6個がp2へ
    assert result.leftover_p1 == 0
    assert result.leftover_p2 == 0


def test_resolve_mutual_exchange_caps_at_max_drop_per_turn():
    """欠陥E-1と同じ1ターン上限が両者同時発火でも適用される。"""
    from src.exchange_virtual_board import OJAMA_MAX_DROP_PER_TURN, resolve_mutual_exchange

    before1 = make_no_chain_board()
    before2 = make_no_chain_board()
    result = resolve_mutual_exchange(
        before1, before2, gen_p1_ojama=0, gen_p2_ojama=448,
        pending_p1=0, pending_p2=0,
    )
    assert result.dropped_to_p1 == OJAMA_MAX_DROP_PER_TURN
    assert result.leftover_p1 == 448 - OJAMA_MAX_DROP_PER_TURN


def test_resolve_mutual_exchange_negative_gen_raises_value_error():
    """負の生成量は silent fallback せず ValueError (fail-silent 回避)。"""
    from src.exchange_virtual_board import resolve_mutual_exchange

    before1 = make_no_chain_board()
    before2 = make_no_chain_board()
    with pytest.raises(ValueError):
        resolve_mutual_exchange(
            before1, before2, gen_p1_ojama=-1, gen_p2_ojama=0,
            pending_p1=0, pending_p2=0,
        )


def test_resolve_mutual_exchange_dead_flags_and_determinism():
    """窒息判定 + 同一入力の再現性 (端数列配置含む)。"""
    from src.exchange_virtual_board import resolve_mutual_exchange

    dead = make_dead_board()
    opp = make_no_chain_board()
    result = resolve_mutual_exchange(
        dead, opp, gen_p1_ojama=0, gen_p2_ojama=0, pending_p1=0, pending_p2=0)
    assert result.p1_dead is True
    assert result.p2_dead is False
    r1 = resolve_mutual_exchange(
        opp, opp, gen_p1_ojama=6, gen_p2_ojama=9, pending_p1=0, pending_p2=0)
    r2 = resolve_mutual_exchange(
        opp, opp, gen_p1_ojama=6, gen_p2_ojama=9, pending_p1=0, pending_p2=0)
    assert r1.board_p1_after == r2.board_p1_after
    assert r1.board_p2_after == r2.board_p2_after


# ============================
# board_p1_pre_landing / board_p2_pre_landing (2026-08-14 指摘12 意味論バグ対処)
# ============================
# 応手可否判定は「相殺後の余剰おじゃまがまだ配置されていない盤面」で行うべき
# (おじゃまは連鎖完了後・受け側ツモ設置時まで降らない、
# memory reference_ojama_landing_gated_by_placement)。これを検証する。


def test_pre_landing_excludes_incoming_ojama_no_own_chain():
    """自分は連鎖なし・相手から着弾のみのケース: pre_landing は着弾前
    (=消化前 before_board と同一)、after は着弾後 (おじゃまが増えている)。"""
    from src.exchange_virtual_board import resolve_mutual_exchange

    before1 = make_no_chain_board()
    before2 = make_no_chain_board()
    result = resolve_mutual_exchange(
        before1, before2, gen_p1_ojama=0, gen_p2_ojama=12,
        pending_p1=0, pending_p2=0,
    )
    assert result.dropped_to_p1 == 12
    assert result.board_p1_pre_landing == before1  # 着弾前=連鎖前と同一(連鎖なしのため)
    assert result.board_p1_pre_landing != result.board_p1_after  # 着弾後は変化している
    assert result.board_p1_after.count_puyos() - result.board_p1_pre_landing.count_puyos() == 12


def test_pre_landing_equals_own_chain_result_when_chain_and_incoming_both_occur():
    """自分は連鎖あり・相手からも着弾があるケース: pre_landing は「自分の連鎖は
    消化済みだが着弾前」= ChainSimulator の final_board そのもの (着弾で
    上書きされる前)、after だけがさらに着弾分を含む。"""
    from src.exchange_virtual_board import resolve_mutual_exchange

    before1 = make_4connect_board()
    before2 = make_no_chain_board()
    result = resolve_mutual_exchange(
        before1, before2, gen_p1_ojama=0, gen_p2_ojama=12,
        pending_p1=0, pending_p2=0,
    )
    own_chain_final = ChainSimulator().simulate(before1).final_board
    assert result.board_p1_pre_landing == own_chain_final
    assert result.board_p1_pre_landing != result.board_p1_after
    assert result.dropped_to_p1 == 12


def test_pre_landing_unaffected_by_leftover_capped_beyond_max_drop():
    """1ターン上限 (欠陥E-1) を超える着弾でも、pre_landing は上限適用前の
    着弾ゼロ状態のまま (leftover の有無に関わらず pre_landing の意味論は
    「まだ何も配置していない」で固定)。"""
    from src.exchange_virtual_board import OJAMA_MAX_DROP_PER_TURN, resolve_mutual_exchange

    before1 = make_no_chain_board()
    before2 = make_no_chain_board()
    result = resolve_mutual_exchange(
        before1, before2, gen_p1_ojama=0, gen_p2_ojama=448,
        pending_p1=0, pending_p2=0,
    )
    assert result.board_p1_pre_landing == before1
    assert result.dropped_to_p1 == OJAMA_MAX_DROP_PER_TURN
    assert result.leftover_p1 == 448 - OJAMA_MAX_DROP_PER_TURN


def test_pre_landing_no_incoming_equals_after_board():
    """着弾が0個の side は pre_landing と after が完全に同一 (差分が無い
    ケースでの回帰確認、後方互換の基準線)。"""
    from src.exchange_virtual_board import resolve_mutual_exchange

    before1 = make_4connect_board()
    before2 = make_no_chain_board()
    result = resolve_mutual_exchange(
        before1, before2, gen_p1_ojama=0, gen_p2_ojama=0,
        pending_p1=0, pending_p2=0,
    )
    assert result.dropped_to_p1 == 0
    assert result.board_p1_pre_landing == result.board_p1_after


def test_pre_landing_deterministic_across_repeated_calls():
    """同一入力は pre_landing も再現性がある (乱数着弾より前の値なので当然
    決定論だが、明示的に固定する)。"""
    from src.exchange_virtual_board import resolve_mutual_exchange

    opp = make_no_chain_board()
    r1 = resolve_mutual_exchange(
        opp, opp, gen_p1_ojama=6, gen_p2_ojama=9, pending_p1=0, pending_p2=0)
    r2 = resolve_mutual_exchange(
        opp, opp, gen_p1_ojama=6, gen_p2_ojama=9, pending_p1=0, pending_p2=0)
    assert r1.board_p1_pre_landing == r2.board_p1_pre_landing
    assert r1.board_p2_pre_landing == r2.board_p2_pre_landing
