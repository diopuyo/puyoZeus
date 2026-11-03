"""高速化の境界と独立した従来物理との等価性を検証する。"""
from collections import Counter
from itertools import combinations

import numpy as np
import pytest

from src.board import Board, BOARD_ROWS, BOARD_COLS, HIDDEN_ROWS
from src.chain import ChainSimulator, MIN_ERASE_COUNT
from src.post_counter_geometry import BoundSimulator, COLORS, geometry, usable_colors, all_landings_dead, place_pair
from src.post_counter_geometry import has_four, component
from src.indicators_v2 import _place_pair_to_board

SEED = 3502
SAMPLES = 60


def boards(settled: bool = True) -> list[Board]:
    """窒息・隠し段・埋没色・空盤面を含む固定乱数の盤面群。"""
    rng = np.random.default_rng(SEED)
    result = [Board()]
    for _ in range(SAMPLES):
        board = Board()
        for col in range(BOARD_COLS):
            height = int(rng.integers(BOARD_ROWS + 1))
            if height:
                board._grid[-height:, col] = rng.choice((*COLORS, 9), height)
        if not settled:
            board._grid[rng.random(board._grid.shape) < 0.25] = 0
        result.append(board)
    return result


def oracle_usable(board: Board, supply: Counter) -> int:
    """空きマスを色で埋めた連結を従来BFSで検査する独立な参照計算。"""
    free = board._grid == 0
    for col in range(BOARD_COLS):
        occupied = np.flatnonzero(board._grid[:, col])
        if occupied.size:
            free[occupied[0]:, col] = False
    free[:HIDDEN_ROWS] = False
    simulator = ChainSimulator(exclude_hidden_row_from_pop=True)
    for color in COLORS:
        filled = board.copy()
        filled._grid[free] = color
        for group in simulator.find_groups(filled):
            if group.color != color:
                continue
            existing = sum(board._grid[r, c] == color for r, c in group.cells)
            vacant = sum(free[r, c] for r, c in group.cells)
            if existing and existing + min(vacant, supply[color]) >= MIN_ERASE_COUNT:
                return int(np.isin(board._grid, COLORS).sum())
    return 0


@pytest.mark.parametrize('settled', (True, False))
@pytest.mark.parametrize('supply', (Counter(), Counter({1: 2}), Counter({1: 1, 3: 1}),
                                    Counter({c: 2 for c in COLORS})))
def test_reachability_matches_independent_flood_fill(settled: bool, supply: Counter) -> None:
    for board in boards(settled):
        assert usable_colors(board, supply) == oracle_usable(board, supply)


@pytest.mark.parametrize('amount', range(31))
def test_all_landing_remainders_match_legacy(amount: int) -> None:
    fast, old = BoundSimulator(exclude_hidden_row_from_pop=True), ChainSimulator()
    for board in boards():
        before = board._grid.copy()
        for cols in combinations(range(BOARD_COLS), amount % BOARD_COLS):
            actual = fast.drop_ojama_with_remainder_columns(board, amount, cols)
            expected = old.drop_ojama_with_remainder_columns(board, amount, cols)
            np.testing.assert_array_equal(actual._grid, expected._grid)
        assert all_landings_dead(board, amount) == all(old.drop_ojama_with_remainder_columns(
            board, amount, cols).is_dead() for cols in combinations(range(BOARD_COLS), amount % BOARD_COLS))
        np.testing.assert_array_equal(board._grid, before)


def test_quiet_and_firing_simulations_match_legacy() -> None:
    fast = BoundSimulator(exclude_hidden_row_from_pop=True)
    old = ChainSimulator(exclude_hidden_row_from_pop=True)
    for board in boards():
        actual, expected = fast.simulate_reply(board), old.simulate(board)
        assert (actual.chain_count, actual.total_erased, actual.total_ojama) == (
            expected.chain_count, expected.total_erased, expected.total_ojama)
        np.testing.assert_array_equal(actual.final_board._grid, expected.final_board._grid)
        actual.final_board._grid[:] = 0
        np.testing.assert_array_equal(fast.simulate_reply(board).final_board._grid, expected.final_board._grid)


def test_geometry_cache_does_not_reuse_mutated_board() -> None:
    board = Board()
    empty = geometry(board._grid.tobytes())
    board._grid[-1] = 1
    assert geometry(board._grid.tobytes()) != empty
    assert geometry.cache_info().maxsize is not None


@pytest.mark.parametrize('pair', ((1, 1), (1, 2), (4, 5)))
def test_all_pair_placements_match_legacy(pair: tuple[int, int]) -> None:
    for board in boards():
        for rotation in range(4):
            for col in range(BOARD_COLS if rotation % 2 == 0 else BOARD_COLS-1):
                actual = place_pair(board, pair, col, rotation)
                expected = _place_pair_to_board(board, pair, col, rotation)
                assert (actual is None) == (expected is None)
                if actual is not None:
                    np.testing.assert_array_equal(actual._grid, expected._grid)


def test_four_detection_exhaustive_local_patterns() -> None:
    # 4×4の全65,536配置で、端と折返しを含む次数短絡を連結探索と比較する。
    width = MIN_ERASE_COUNT
    cells = [1 << (row * BOARD_COLS + col) for row in range(width) for col in range(width)]
    for pattern in range(1 << len(cells)):
        bits = sum(bit for index, bit in enumerate(cells) if pattern & (1 << index))
        remaining, expected = bits, False
        while remaining:
            group = component(remaining & -remaining, bits)
            expected |= group.bit_count() >= MIN_ERASE_COUNT
            remaining &= ~group
        assert has_four(bits) == expected


@pytest.mark.parametrize('field', ('board', 'queue', 'incoming', 'hands', 'elapsed', 'palette'))
def test_proof_cache_keys_all_inputs(monkeypatch: pytest.MonkeyPatch, field: str) -> None:
    import src.exchange_post_counter_bound as module
    from src.scoring import compute_effective_rate
    engine, calls = module.PostCounterDeathBound(), []
    def proof(*args: object) -> dict:
        calls.append(args)
        return dict(dead=False, nodes=len(calls))
    monkeypatch.setattr(module, 'prove_post_counter', proof)
    args = dict(boards=[Board()], queue=(1, 2, 3, 4), incoming=171, hands=1,
                elapsed=0., palette=(1, 2, 3, 4))
    expected = engine.proofs_for(**args)
    assert engine.proofs_for(**args) == expected and len(calls) == 1
    if field == 'board':
        args['boards'][0]._grid[-1, 0] = 1
    elif field == 'queue':
        args[field] = (2, 1, 3, 4)
    elif field == 'palette':
        args[field] = (1, 2, 3, 5)
    elif field == 'elapsed':
        args[field] = 1000.
        assert compute_effective_rate(args[field]) != compute_effective_rate(0.)
    else:
        args[field] += 1
    engine.proofs_for(**args)
    assert len(calls) == 2


def test_proof_cache_is_bounded_and_cleared_at_game_boundary(monkeypatch: pytest.MonkeyPatch) -> None:
    from types import SimpleNamespace as NS
    import src.exchange_post_counter_bound as module
    from src.board_state_machine import BoardState
    monkeypatch.setattr(module, 'CACHE_LIMIT', 2)
    monkeypatch.setattr(module, 'prove_post_counter', lambda *args: dict(dead=False))
    engine = module.PostCounterDeathBound()
    for incoming in (170, 171, 172):
        engine.proofs_for([Board()], (1, 2), incoming, 1, 0., (1, 2, 3, 4))
    assert len(engine.board_cache) == 2
    side = NS(state=BoardState.CHAIN, confirmed_board=None, next_pair=None, dnext_pair=None)
    engine.observe(NS(p1=side, p2=side), 2)
    assert not engine.board_cache
