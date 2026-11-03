"""連結確認の短縮が消去・探索・死亡証明を変えないことを検証する。"""
from __future__ import annotations

import random

import numpy as np
import pytest

from src.board import BOARD_COLS, BOARD_ROWS, Board
from src.chain import ChainSimulator
from src import exchange_fast_expand as expand
from src.exchange_event_multilanding import prove_multilanding
from src.phase_j.live_connectivity import exact_connectivity, is_stable, local_pop, erasable_groups

SHAPE = (BOARD_ROWS, BOARD_COLS)
COLORS = (0, 1, 2, 3, 4, 5, 9, 10)
SAMPLES = 80


@pytest.mark.parametrize('hidden', (0, 1))
@pytest.mark.parametrize('seed', range(8))
def test_predicates_match_reference(hidden: int, seed: int) -> None:
    """浮遊・UNKNOWN・おじゃま・隠し段を含む盤面を基準実装と照合する。"""
    rng = np.random.default_rng(seed)
    simulator = ChainSimulator(exclude_hidden_row_from_pop=bool(hidden), fast_groups=True)
    for _ in range(SAMPLES):
        board = Board()
        board._grid = rng.choice(COLORS, size=SHAPE).astype(np.uint8)
        assert is_stable(board._grid, hidden) == (not simulator.find_erasable_groups(board))
        assert erasable_groups(board._grid, hidden) == simulator._erasable_fast(board)
        cells = [(int(rng.integers(BOARD_ROWS)), int(rng.integers(BOARD_COLS))) for _ in range(2)]
        assert local_pop(board._grid, cells, hidden) == expand.local_pop(board._grid, cells, hidden)


@pytest.mark.parametrize('hidden', (0, 1))
@pytest.mark.parametrize('dtype', ('uint8', 'int8', 'int64'))
def test_cache_tracks_mutation_and_ghost_rule(hidden: int, dtype: str) -> None:
    """盤面のin-place更新・dtype・隠し段設定を跨いで誤再利用しない。"""
    simulator = ChainSimulator(exclude_hidden_row_from_pop=bool(hidden))
    expander, board = expand.FastExpander(simulator), Board()
    board._grid = board._grid.astype(dtype)
    original = expand.FastExpander.is_stable
    with exact_connectivity() as counts:
        assert expander.is_stable(board)
        assert expander.is_stable(board)
        board._grid[0, :4] = 1
        assert expander.is_stable(board) == bool(hidden)
        board._grid[-1, :4] = 2
        assert not expander.is_stable(board)
        assert counts == dict(hits=1 + hidden, misses=3 - hidden)
    assert expand.FastExpander.is_stable is original


@pytest.mark.parametrize('seed', range(8))
def test_expansion_and_proof_identical(seed: int) -> None:
    """候補の順序・得点・最終盤面と、ノード数を含む死亡証明を照合する。"""
    rng = random.Random(seed)
    board = Board()
    for col in range(BOARD_COLS):
        for row in range(rng.randrange(3, 9), BOARD_ROWS):
            board._grid[row, col] = rng.choice((1, 2, 3, 4, 9))
    sim = ChainSimulator(exclude_hidden_row_from_pop=True, fast_groups=True)
    expander = expand.FastExpander(sim)
    before = expander.known([(0., board)], (1, 2), True, False)
    proof = prove_multilanding(board, (1, 2, 3, 4), 75, 1, 0., lambda *args: 0.)
    with exact_connectivity():
        after = expander.known([(0., board)], (1, 2), True, False)
        actual = prove_multilanding(board, (1, 2, 3, 4), 75, 1, 0., lambda *args: 0.)
    assert [(score, b._grid.tobytes(), n) for score, b, n in before] == [
        (score, b._grid.tobytes(), n) for score, b, n in after]
    assert actual == proof


def test_unknown_simulator_keeps_original() -> None:
    """差し替えシミュレータの独自規則は上書きしない。"""
    class Custom(ChainSimulator):
        def find_erasable_groups(self, board: Board) -> list:
            return [object()]
    with exact_connectivity():
        assert not expand.FastExpander(Custom()).is_stable(Board())


def test_scope_restored_on_error() -> None:
    original = expand.local_pop
    with pytest.raises(RuntimeError), exact_connectivity():
        raise RuntimeError('終了時復元の検査')
    assert expand.local_pop is original


def test_default_off_and_evaluation_scope(monkeypatch: pytest.MonkeyPatch) -> None:
    """既定OFF、ONでも認識用スコープには適用しない。"""
    from src.phase_j.live_cache import bounded_chain_caches
    from src.production_config import LIVE_EXACT_CONNECTIVITY_DEFAULT
    assert LIVE_EXACT_CONNECTIVITY_DEFAULT is False
    original = expand.local_pop
    monkeypatch.delenv('PUYO_LIVE_EXACT_CONNECTIVITY', raising=False)
    with bounded_chain_caches(future_search=True):
        assert expand.local_pop is original
    monkeypatch.setenv('PUYO_LIVE_EXACT_CONNECTIVITY', '1')
    with bounded_chain_caches():
        assert expand.local_pop is original
    with bounded_chain_caches(future_search=True):
        assert expand.local_pop is local_pop
    assert expand.local_pop is original


@pytest.mark.parametrize('hidden', (0, 1))
def test_adjacent_ojama_in_hidden_row(hidden: int) -> None:
    board = Board()
    board._grid[0, :4] = 9
    board._grid[1, :4] = 1
    sim = ChainSimulator(exclude_hidden_row_from_pop=bool(hidden), fast_groups=True)
    assert erasable_groups(board._grid, hidden) == sim._erasable_fast(board)
    with exact_connectivity():
        result = sim.simulate(board)
    assert result.total_erased == 4 and result.total_ojama == 4
