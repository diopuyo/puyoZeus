"""段間の高速化が候補・不可視段・おじゃまの物理を変えないことを確認する。"""
from types import SimpleNamespace
import numpy as np
import pytest
from src.board import Board
from src import light_likelihood as light

SEEDS = 200

def reference(board: np.ndarray) -> np.ndarray:
    value = Board.from_list(board.tolist())
    groups = light.SIMULATOR.find_erasable_groups(value)
    light.SIMULATOR._erase_fast(value, groups)
    light.SIMULATOR._gravity_fast(value)
    return value._grid.copy()

@pytest.mark.parametrize('seed', range(SEEDS))
def test_native_step_matches_python(seed: int) -> None:
    rng = np.random.default_rng(seed)
    board = rng.choice((0, 1, 2, 3, 4, 5, 9), size=(13, 6)).astype(np.int8)
    if seed % 2:
        board[9:, 0] = 1
        board[0, 0] = 9
        board[1:5, 1] = 2
    before = board.copy()
    assert np.array_equal(light.advance(board), reference(board))
    assert np.array_equal(board, before)

def test_unknown_and_no_native_keep_reference(monkeypatch: pytest.MonkeyPatch) -> None:
    board = np.zeros((13, 6), dtype=np.int8)
    board[1, 0], board[5, 0], board[10:, 1] = 10, 2, 3
    assert np.array_equal(light.advance(board), reference(board))
    monkeypatch.setattr(light.native, 'NATIVE_AVAILABLE', False)
    board[1, 0] = 1
    assert np.array_equal(light.advance(board), reference(board))

def test_converged_candidates_share_work_without_aliases(monkeypatch: pytest.MonkeyPatch) -> None:
    board = np.zeros((13, 6), dtype=np.int8)
    calls = []
    original = light.advance
    def counted(value: np.ndarray) -> np.ndarray:
        calls.append(value.tobytes())
        return original(value)
    monkeypatch.setattr(light, 'advance', counted)
    result = light.advance_matrix(np.stack([board, board]), {})
    assert len(calls) == 1
    result[0, 12, 0] = 1
    assert not result[1].any() and not board.any()

@pytest.mark.parametrize('limit, expected', [(4, 0), (5, 5)])
def test_work_limit_precedes_expansion_and_deduplicates_origins(
        monkeypatch: pytest.MonkeyPatch, limit: int, expected: int) -> None:
    from src import light_refind as refind
    board = np.zeros((13, 6), dtype=np.int8)
    spec = (board, [(12, 0)], (board.tobytes(), (0,)))
    monkeypatch.setattr(refind, 'after_specs', lambda *args: iter([spec]))
    monkeypatch.setattr(refind, 'MAX_ORIGIN_COMBINATIONS', limit)
    value = refind.PhysicalRefind(lambda grid: ([], grid.copy()))
    value.base, value.base_versions = board, [board, board.copy()]
    value.palette = (0, 1, 2, 3, 4)
    value.collect_candidates(set(), 40, {})
    assert value.work_items == 5 and len(value.candidates) == expected
    assert value.work_limited == (expected == 0)

def test_limited_updates_use_base_forecast_and_next_chain_recovers(
        monkeypatch: pytest.MonkeyPatch) -> None:
    from src import light_refind as refind
    from src.fire_board import Forecast
    from src.light_hidden_posterior import scenarios
    monkeypatch.setenv('PUYO_LIGHT_ROBUST_CELLS', '1')
    board = np.zeros((13, 6), dtype=np.int8)
    value = refind.PhysicalRefind(lambda grid: ([], grid.copy()))
    value.base, value.chain_id, value.work_limited = board, 1., True
    value.colors.update((1, 2, 3, 4))
    forecast = Forecast(1., (40,), (40, 320), 'refind', None, 1.)
    owner = SimpleNamespace(_light_physics=value)
    for stamp in (1.1, 1.2):
        value.observe(stamp, {}, forecast, True)
        assert refind.physical_forecast(owner, forecast) is forecast
        assert scenarios([owner, SimpleNamespace()]) == []
    assert value.work_limit_hits == 2
    spec = (board, [], (board.tobytes(), ()))
    monkeypatch.setattr(refind, 'after_specs', lambda *args: iter([spec]))
    value.start(2., 40, True)
    assert not value.work_limited and len(value.candidates) == 1
    assert value.work_limit_hits == 2

@pytest.mark.parametrize('module', ('src.fire_board', 'src.phase_j._live_ex_frozen.src.fire_board'))
def test_over_limit_disables_hidden_guard_for_source_and_frozen(
        monkeypatch: pytest.MonkeyPatch, module: str) -> None:
    from importlib import import_module
    from src import light_refind as refind
    monkeypatch.setenv('PUYO_LIGHT_ROBUST_CELLS', '1')
    board = np.zeros((13, 6), dtype=np.int8)
    owner = import_module(module).ChainForecast()
    owner._new_chain(1., 40, uncertain_hidden=True)
    value = refind.PhysicalRefind(owner.trace)
    value.chain_id, value.work_limited = owner.fc.chain_id, True
    owner._light_physics = value
    refind.update_physical_refind(owner, 1.1, board, {})
    assert not owner._light_hidden_fire
    assert owner.current() is owner.fc
