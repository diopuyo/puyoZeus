"""隠し段は設置可能だが、可視段へ落ちるまで連結しない。"""
from __future__ import annotations

import numpy as np
import pytest

from src import chain_bitboard, puyo_core_bridge as native
from src import indicators_v2 as iv
from src.b3_search import HMAX_TOTAL_HANDS, _SIM, ideal_fire_search
from src.board import Board


def hidden_board(kind: str) -> Board:
    """支持をおじゃまで埋め、浮遊による曖昧さを除いた盤面。"""
    g = np.full((13, 6), 9, dtype=np.int8)
    if kind == "horizontal":
        g[0, :4] = 1
    elif kind == "vertical":
        g[:4, 0] = 1
    elif kind == "bridge":
        g[0, :3] = 1
        g[1, (0, 2)] = 1
    elif kind == "visible":
        g[:5, 0] = 1
    else:
        g[0, :4] = 1
        g[1, :4] = 2
    return Board.from_list(g.tolist())


@pytest.mark.parametrize("kind", ["horizontal", "vertical", "bridge", "fall", "visible"])
@pytest.mark.parametrize("engine", ["b3", "native", "bitboard"])
def test_hidden_connection_rule(kind: str, engine: str) -> None:
    board = hidden_board(kind)
    if engine == "b3":
        result = _SIM.simulate(board)
    elif engine == "native":
        result = native.simulate_chain(board, exclude_hidden_row_from_pop=True)
    else:
        result = chain_bitboard.simulate_single(board, exclude_hidden_row_from_pop=True)
    count = {"fall": 2, "visible": 1}.get(kind, 0)
    assert result.chain_count == count
    assert result.total_erased == count * 4
    if kind == "visible" and engine != "bitboard":
        # 可視4個+隠し1個を5連結と数えて連結ボーナスを加算してはいけない。
        assert result.exact_score == 40
        assert np.count_nonzero(result.final_board._grid == 1) == 1


@pytest.mark.parametrize("rotation", [0, 1, 2, 3])
def test_pair_can_use_hidden_row(rotation: int) -> None:
    g = np.zeros((13, 6), dtype=np.int8)
    g[2:, :2] = 9
    if rotation in (1, 3):
        g[1, :2] = 9
    board = Board.from_list(g.tolist())
    placed = iv._place_pair_to_board(board, (1, 2), 0, rotation)
    assert placed is not None and placed._grid[0, 0] != 0
    assert not placed.is_dead()
    ref = next(b for c, r, b in native.enumerate_placements(board, (1, 2)) if c == 0 and r == rotation)
    np.testing.assert_array_equal(placed._grid, ref._grid)
    assert iv._drop_one_color(placed, 0, 3) is None


def test_single_can_use_hidden_row_without_connecting() -> None:
    g = np.zeros((13, 6), dtype=np.int8)
    g[1:, 0] = 9
    g[1:4, 0] = 1
    placed = iv._drop_one_color(Board.from_list(g.tolist()), 0, 1)
    assert placed is not None and placed._grid[0, 0] == 1
    assert _SIM.simulate(placed).chain_count == 0


@pytest.mark.parametrize("hmax", [1, 20])
def test_full_is_terminal_even_at_exact_cap(hmax: int) -> None:
    # 中央の可視最上段は空ける。他列最後の1マスを埋めると合法手が無くなる。
    g = np.full((13, 6), 9, dtype=np.int8)
    g[:2, 2] = 0
    g[0, :2] = 0
    res = ideal_fire_search(g, (), (1,), hmax=hmax)
    assert res.done_hands == 1 and res.full
    assert res.empty_cells == 2
    assert res.at(100)[2] is False
    assert res.diagnostics(100)['full'] == 1
    assert np.isnan(res.diagnostics(100)['trunc_empty'])


def test_already_full_and_known_pair_cannot_fit() -> None:
    g = np.full((13, 6), 9, dtype=np.int8)
    g[:2, 2] = 0
    g[0, 0] = 0
    res = ideal_fire_search(g, (1, 2), (1, 2))
    assert res.done_hands == 0 and res.full and res.beam_died
    assert not res.at(21)[2]


def test_cap20_and_empty_cells_include_hidden_row() -> None:
    assert HMAX_TOTAL_HANDS == 20
    g = np.zeros((13, 6), dtype=np.int8)
    res = ideal_fire_search(g, (), (1, 2), hmax=1)
    assert not res.full and res.empty_cells == 76
    assert res.diagnostics(2)['trunc_empty'] == 76
    assert res.diagnostics(2)['trunc_empty_nf'] == pytest.approx(76 / 78)
    assert np.isnan(res.diagnostics(1)['trunc_empty'])
    assert ideal_fire_search(g, (), (1,), hmax=100).hmax == 20


@pytest.mark.parametrize("hmax", [1, 2, 7, 12])
def test_search_prefix_is_independent_of_cap(hmax: int) -> None:
    g = np.zeros((13, 6), dtype=np.int8)
    g[10:, 0] = [1, 2, 1]
    g[11:, 1] = [1, 2]
    a = ideal_fire_search(g, (1, 2, 3, 4), (1, 2, 3, 4), hmax=hmax)
    b = ideal_fire_search(g, (1, 2, 3, 4), (1, 2, 3, 4))
    assert a.scores == b.scores[:hmax]
    assert a.chains == b.chains[:hmax]


def test_diagnostics_side_swap_and_append_only() -> None:
    from src.b3_column_spec import COLUMN_NAMES, SPEC_BY_NAME, Kind, swap_sides
    stems = ('search_trunc_empty', 'search_trunc_empty_nf', 'search_full', 'mk_search_trunc',
             'ta_trunc_empty', 'ta_trunc_empty_nf', 'ta_full', 'mk_ta_trunc')
    names = [f'{stem}_{s}' for stem in stems for s in ('p1', 'p2')]
    end = COLUMN_NAMES.index('search_version')
    assert COLUMN_NAMES[end-len(names):end] == names
    row = {name: float(i) for i, name in enumerate(names)}
    swapped = swap_sides(row)
    for name in names:
        spec = SPEC_BY_NAME[name]
        assert spec.kind is Kind.SIDE_ABS and spec.role == 'audit'
        assert swapped[name] == row[spec.counterpart]
    assert swap_sides(swapped) == row


def test_lookahead_counts_updates_and_excludes_full() -> None:
    from scripts.b3_time_gate_support import lookahead_rates, search_status
    base = {f'{stem}_{s}': value for s in ('p1', 'p2') for stem, value in (
        ('search_done_h', 20), ('ta_trunc', 0), ('ta_req_h', 8), ('ta_done_h', 8),
        ('chain_n_steps', 0), ('ep_chain7plus', 0))}
    rows = [dict(base), dict(base), dict(base), dict(base)]
    rows[1].update(ta_req_h_p1=21, ta_done_h_p1=20, ta_trunc_p1=1, ta_trunc_empty_p1=6)
    rows[2].update(ta_req_h_p1=21, ta_done_h_p1=20, ta_trunc_p1=1, ta_trunc_empty_p1=5)
    rows[3].update(search_done_h_p2=2, search_full_p2=1, search_trunc_empty_p2=60)
    result = lookahead_rates([search_status(row) for row in rows])
    assert result['unreached'] == dict(n=4, failures=3, rate=0.75)
    assert result['spacious_truncation'] == dict(n=4, failures=1, rate=0.25)
    assert lookahead_rates([])['spacious_truncation']['rate'] is None
    assert lookahead_rates([{'unreached': True}])['spacious_truncation']['rate'] is None


def test_met_request_is_not_spacious_truncation() -> None:
    from scripts.b3_time_gate_support import spacious_truncation
    assert not spacious_truncation(dict(ta_req_h_p1=20, ta_done_h_p1=20, ta_trunc_empty_p1=10))


@pytest.mark.parametrize("full", [False, True])
def test_time_axis_propagates_search_diagnostics(full: bool) -> None:
    from src.b3_state import Episode, SideState
    from src.b3_timeaxis import time_axis_features
    from src.b3_timing import TimingTable
    a, b = SideState('1P'), SideState('2P')
    a.episode = Episode(0.0, 0.0, pred_l=100, pred_score=40)
    b.board = np.full((13, 6), 9 if full else 0, dtype=np.int8)
    b.board[:2, 2] = 0
    row = time_axis_features(a, b, 0.0, 0.0, (1, 2), TimingTable(), 0.0)
    assert row['ta_req_h'] > HMAX_TOTAL_HANDS
    assert row['ta_full'] == float(full)
    assert row['ta_trunc'] == float(not full)
    assert row['mk_ta_trunc'] == (8 if full else 0)
    if full:
        assert row['ta_done_h'] == 0
        assert np.isnan(row['ta_trunc_empty'])
    else:
        assert row['ta_done_h'] == HMAX_TOTAL_HANDS
        assert row['ta_trunc_empty'] >= 6
        assert row['ta_trunc_empty_nf'] == pytest.approx(row['ta_trunc_empty'] / 78)
    a.episode = None
    blank = time_axis_features(a, b, 0.0, 0.0, (1, 2), TimingTable(), 0.0)
    assert all(np.isnan(blank[k]) for k in ('ta_full', 'ta_trunc_empty', 'ta_trunc_empty_nf'))
