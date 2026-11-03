"""単盤面Rust経路の全段・UNKNOWN壁・隠し段・入力不変を照合する。"""
from __future__ import annotations

import numpy as np
import pytest

from src.board import BOARD_COLS, BOARD_ROWS, Board
from src.chain import ChainResult, ChainSimulator
from src.phase_j.live_native import LIBRARY, exact_native
from src.exchange_fast_expand import FastExpander
from src import exchange_fast_expand as expand
from src.exchange_prefire_candidates import completion
from src.scoring import calculate_chain_score

pytestmark = pytest.mark.skipif(not LIBRARY.exists(), reason='単盤面Rust DLL未ビルド')
SEED = 20261013
SAMPLES = 5000
SHAPE = (BOARD_ROWS, BOARD_COLS)
PROOF_HEIGHT = 5


def signature(result: ChainResult) -> tuple:
    """返り値の全フィールドを比較可能な値へ変換する。"""
    steps = tuple((s.chain_index, s.erased_groups, s.erased_count, s.erased_ojama,
                   s.board_before._grid.dtype.str, s.board_before._grid.tobytes(),
                   s.board_after._grid.dtype.str, s.board_after._grid.tobytes()) for s in result.steps)
    return (steps, result.chain_count, result.total_erased, result.total_ojama,
            result.participating_cells, result.final_board._grid.dtype.str,
            result.final_board._grid.tobytes(), calculate_chain_score(result))


@pytest.mark.parametrize('hidden', [False, True])
@pytest.mark.parametrize('dtype', [np.int8, np.uint8])
@pytest.mark.parametrize('fast_groups', [False, True])
def test_random_full_results(hidden: bool, dtype: type, fast_groups: bool) -> None:
    """穴・隠し段・おじゃま・UNKNOWNを含む4万盤面で全段を照合する。"""
    rng = np.random.default_rng(SEED)
    sim = ChainSimulator(cache_enabled=False, exclude_hidden_row_from_pop=hidden, fast_groups=fast_groups)
    for index in range(SAMPLES):
        palette = [0, 1, 2, 3, 4, 5, 9, 10] if index % 2 else [0, 1, 2, 9, 10]
        board = Board.from_list(rng.choice(palette, SHAPE).tolist())
        board._grid = board._grid.astype(dtype)
        saved = board._grid.tobytes()
        expected = signature(sim.simulate(board))
        stable = FastExpander(sim).is_stable(board)
        expected_completion = completion(board, sim)
        cells = [(int(rng.integers(BOARD_ROWS)), int(rng.integers(BOARD_COLS))) for _ in range(2)]
        expected_local = expand.local_pop(board._grid, cells, int(hidden))
        with exact_native():
            assert signature(sim.simulate(board)) == expected, index
            assert FastExpander(sim).is_stable(board) == stable, index
            assert completion(board, sim) == expected_completion, index
            assert expand.local_pop(board._grid, cells, int(hidden)) == expected_local, index
        assert board._grid.tobytes() == saved


def test_context_restores_methods() -> None:
    """有効範囲を抜けた後に通常実装へ戻る。"""
    original = ChainSimulator._simulate_uncached
    with exact_native():
        assert ChainSimulator._simulate_uncached is not original
    assert ChainSimulator._simulate_uncached is original


def test_default_off_and_scope(monkeypatch: pytest.MonkeyPatch) -> None:
    """通常既定OFF、認識スコープではON指定も無効にする。"""
    from src.phase_j.live_cache import bounded_chain_caches
    from src.production_config import LIVE_EXACT_NATIVE_DEFAULT
    assert LIVE_EXACT_NATIVE_DEFAULT is False
    original = ChainSimulator._simulate_uncached
    monkeypatch.delenv('PUYO_LIVE_EXACT_NATIVE', raising=False)
    with bounded_chain_caches(future_search=True):
        assert ChainSimulator._simulate_uncached is original
    monkeypatch.setenv('PUYO_LIVE_EXACT_NATIVE', '1')
    with bounded_chain_caches():
        assert ChainSimulator._simulate_uncached is original
    with bounded_chain_caches(future_search=True):
        assert ChainSimulator._simulate_uncached is not original
    assert ChainSimulator._simulate_uncached is original


@pytest.mark.parametrize('seed', range(8))
def test_expansion_proof_and_order(seed: int) -> None:
    """候補順・全ノード数・死亡証明をnative前後で照合する。"""
    from src.exchange_event_multilanding import prove_multilanding
    rng = np.random.default_rng(seed)
    board = Board()
    board._grid[-PROOF_HEIGHT:] = rng.choice([1, 2, 3, 4, 9], (PROOF_HEIGHT, BOARD_COLS))
    sim = ChainSimulator(cache_enabled=False, exclude_hidden_row_from_pop=True, fast_groups=True)
    def outcomes() -> list:
        return [(n, placed._grid.tobytes(), None if result is None else signature(result))
                for n, placed, result in FastExpander(sim).placements(board, (1, 2))]
    def optimistic(*args: object) -> float:
        return 0.
    expected = outcomes()
    proof = prove_multilanding(board, (1, 2, 3, 4), 75, 1, 0., optimistic)
    with exact_native():
        assert outcomes() == expected
        assert prove_multilanding(board, (1, 2, 3, 4), 75, 1, 0., optimistic) == proof
