"""1 手展開の高速化 (src/exchange_fast_expand.py) が従来経路と全出力で一致することを確かめる。"""
from __future__ import annotations

import random

import numpy as np
import pytest

from src.board import BOARD_COLS, BOARD_ROWS, COLOR_OJAMA, Board
from src.chain import ChainSimulator
from src.exchange_event_multilanding import prove_multilanding
from src.exchange_fast_expand import FastExpander, local_pop
from src.indicators_v2 import (_enumerate_placements, _near_future_free_expand,
                               _near_future_known_expand, near_future_fire_power)
from src.scoring import calculate_chain_score

SEEDS = range(3)
BOARDS_PER_SEED = 25
COLORS = (1, 2, 3, 4)
PAIRS = ((1, 2), (3, 3), (4, 1))
FILL_HEIGHTS = (2, 4, 6, 8, 10, 11, 12)
OJAMA_PROBABILITY = 0.1


def random_stack(rng: random.Random) -> Board:
    """重力済みの積み上げ盤面 (連結が残りうる = 静止でない基準もそのまま含む)。"""
    board = Board()
    grid = np.zeros((BOARD_ROWS, BOARD_COLS), dtype=board._grid.dtype)
    for col in range(BOARD_COLS):
        for k in range(rng.choice(FILL_HEIGHTS)):
            grid[BOARD_ROWS - 1 - k, col] = COLOR_OJAMA if rng.random() < OJAMA_PROBABILITY else rng.choice(COLORS)
    board._grid = grid
    return board


def stable_of(sim: ChainSimulator, board: Board) -> Board:
    """連鎖を解決した静止盤面。"""
    return sim.simulate(board).final_board


def boards(sim: ChainSimulator, seed: int, stable: bool) -> list[Board]:
    rng = random.Random(seed)
    made = [random_stack(rng) for _ in range(BOARDS_PER_SEED)]
    return [stable_of(sim, b) for b in made] if stable else made


@pytest.mark.parametrize('ghost', (False, True))
@pytest.mark.parametrize('stable', (False, True))
def test_placements_identical(ghost: bool, stable: bool) -> None:
    reference = ChainSimulator(exclude_hidden_row_from_pop=ghost)
    fast = FastExpander(ChainSimulator(exclude_hidden_row_from_pop=ghost, fast_groups=True))
    compared = with_chain = 0
    for seed in SEEDS:
        for base in boards(reference, seed, stable):
            for pair in PAIRS:
                expected = _enumerate_placements(base, pair, reference)
                actual = fast.placements(base, pair)
                assert [c for c, _ in expected] == [c for c, _, _ in actual]
                for (_, placed), (_, got, result) in zip(expected, actual):
                    assert np.array_equal(placed._grid, got._grid)
                    ref = reference.simulate(placed)
                    final = got if result is None else result.final_board
                    assert np.array_equal(ref.final_board._grid, final._grid)
                    score = 0 if result is None else calculate_chain_score(result).total_score
                    assert score == calculate_chain_score(ref).total_score
                    compared += 1
                    with_chain += ref.chain_count > 0
    assert compared > 500 and with_chain > 10


@pytest.mark.parametrize('resolve', (False, True))
@pytest.mark.parametrize('exact', (False, True))
def test_expansions_identical_to_reference(resolve: bool, exact: bool) -> None:
    sim = ChainSimulator(exclude_hidden_row_from_pop=True)
    fast_sim = ChainSimulator(exclude_hidden_row_from_pop=True, fast_groups=True)
    for seed in SEEDS:
        frontier = [(0.0, b) for b in boards(sim, seed, stable=True)[:8]]
        for pair in PAIRS:
            ref = _near_future_known_expand(frontier, pair, sim, resolve_before_death=resolve,
                                            use_exact_score=exact)
            got = _near_future_known_expand(frontier, pair, fast_sim, resolve_before_death=resolve,
                                            use_exact_score=exact, fast_expand=True)
            assert [(s, c, b._grid.tobytes()) for s, b, c in ref] == [(s, c, b._grid.tobytes()) for s, b, c in got]
        ref = _near_future_free_expand(frontier, COLORS, sim, resolve_before_death=resolve, use_exact_score=exact)
        got = _near_future_free_expand(frontier, COLORS, fast_sim, resolve_before_death=resolve,
                                       use_exact_score=exact, fast_expand=True)
        assert [(s, c, b._grid.tobytes()) for s, b, c in ref] == [(s, c, b._grid.tobytes()) for s, b, c in got]


def test_near_future_fire_power_identical() -> None:
    sim = ChainSimulator(exclude_hidden_row_from_pop=True)
    fast_sim = ChainSimulator(exclude_hidden_row_from_pop=True, fast_groups=True)
    for seed in SEEDS:
        for base in boards(sim, seed, stable=True)[:6]:
            if base.is_dead():
                continue
            kwargs = dict(elapsed_sec=30.0, k_levels=(0, 1, 2), resolve_before_death=True, beam_width=8)
            a = near_future_fire_power(base, (1, 2), (3, 4), simulator=sim, **kwargs)
            b = near_future_fire_power(base, (1, 2), (3, 4), simulator=fast_sim, fast_expand=True, **kwargs)
            assert {k: v.raw for k, v in a.values.items()} == {k: v.raw for k, v in b.values.items()}
            assert a.chain_refs == b.chain_refs


def test_local_pop_matches_full_scan() -> None:
    sim = ChainSimulator(exclude_hidden_row_from_pop=True, fast_groups=True)
    rng = random.Random(5)
    for _ in range(300):
        base = stable_of(sim, random_stack(rng))
        col, color = rng.randrange(BOARD_COLS), rng.choice(COLORS)
        heights = [int(base.height_of(c)) for c in range(BOARD_COLS)]
        if heights[col] >= BOARD_ROWS:
            continue
        cell = (BOARD_ROWS - 1 - heights[col], col)
        child = base.copy()
        child._grid[cell] = color
        assert local_pop(child._grid, [cell], 1) == bool(sim.find_erasable_groups(child))


def stub_optimistic(board: Board, queue: np.ndarray, hands: int, elapsed: float) -> float:
    return 0.0


def test_prove_multilanding_fast_path_identical(monkeypatch: pytest.MonkeyPatch) -> None:
    import src.exchange_event_multilanding as multilanding
    rng = random.Random(21)
    reasons: set[str] = set()
    for _ in range(30):
        board = stable_of(ChainSimulator(exclude_hidden_row_from_pop=True), random_stack(rng))
        if board.is_dead():
            continue
        for incoming, hands, limit in ((40, 1, 800), (120, 2, 800), (200, 2, 3000)):
            queue = (1, 2, 3, 4)
            monkeypatch.setattr(multilanding, 'EXACT_FAST_GROUPS', False)  # 基準は全て従来経路
            ref = prove_multilanding(board, queue, incoming, hands, 0.0, stub_optimistic, 0, limit,
                                     precheck=False, fast_expand=False)
            monkeypatch.setattr(multilanding, 'EXACT_FAST_GROUPS', True)
            got = prove_multilanding(board, queue, incoming, hands, 0.0, stub_optimistic, 0, limit,
                                     precheck=True, fast_expand=True)
            assert got == ref
            reasons.add(ref['reason'])
    assert len(reasons) >= 2


def test_drop_ojama_fast_identical() -> None:
    """端数列を全通り試し、従来の着地と盤面が一致する (空き穴のある柱・満杯の柱を含む)。"""
    from itertools import combinations
    from src.exchange_fast_expand import drop_ojama_fast
    sim = ChainSimulator(exclude_hidden_row_from_pop=True)
    rng = random.Random(9)
    compared = 0
    for seed in range(20):
        base = random_stack(random.Random(seed))
        if seed % 2:
            base._grid[BOARD_ROWS - 3, rng.randrange(BOARD_COLS)] = 0  # 柱の途中に穴
        for dropped in (0, 5, 6, 13, 30):
            for columns in combinations(range(BOARD_COLS), dropped % BOARD_COLS):
                expected = sim.drop_ojama_with_remainder_columns(base, dropped, columns)
                assert np.array_equal(drop_ojama_fast(base, dropped, columns)._grid, expected._grid)
                compared += 1
    assert compared >= 300
    with pytest.raises(ValueError):
        drop_ojama_fast(Board(), 7, (1, 1))
