"""DLL欠損と不正入力はPythonへ戻し、評価子の起動を維持する。"""
from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from src.board import BOARD_ROWS, BOARD_COLS, Board
from src.chain import ChainSimulator
from src.phase_j import live_native as native
from src.phase_j import live_connectivity as connectivity
from src.phase_j.live_cache import bounded_chain_caches
from tests.test_live_native import signature

SHAPE = (BOARD_ROWS, BOARD_COLS)


@pytest.mark.parametrize('broken', [False, True])
def test_missing_library_keeps_worker_alive(
        monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
        caplog: pytest.LogCaptureFixture, broken: bool) -> None:
    """実際のworker入口を通し、欠損・壊れたDLLでも繰返し評価できる。"""
    from src.phase_j import live_eval_worker as worker
    from src.phase_j import live_cpu, live_lifetime
    path = tmp_path / 'invalid.dll'
    if broken:
        path.write_bytes(b'not a DLL')
    monkeypatch.setattr(native, 'LIBRARY', path)
    monkeypatch.setattr(native, '_load_warning_emitted', False)
    monkeypatch.setenv('PUYO_LIVE_EXACT_NATIVE', '1')
    monkeypatch.setattr(live_cpu, 'apply_runtime', lambda *a: {})
    monkeypatch.setattr(live_lifetime, 'protect_parent', lambda *a: None)
    monkeypatch.setattr(worker, 'make_overlay', lambda *a: None)
    board = Board()
    board._grid[-1, :4] = 1
    sim = ChainSimulator(cache_enabled=False)
    expected = signature(sim.simulate(board))
    results: list[tuple] = []

    def serve(*args: object) -> None:
        results.append(signature(sim.simulate(board)))

    monkeypatch.setattr(worker, 'serve', serve)
    with caplog.at_level(logging.WARNING, logger=native.__name__):
        for _ in range(2):
            worker.worker(SimpleNamespace(send=lambda value: None), (), 0)
    assert results == [expected, expected]
    assert len([r for r in caplog.records if r.name == native.__name__]) == 1


@pytest.mark.parametrize('grid', [
    np.full(SHAPE, -1, dtype=np.int8),
    np.full(SHAPE, 256, dtype=np.int16),
    np.full(SHAPE, 11, dtype=np.uint8),
    np.full(SHAPE, 1.5), np.full(SHAPE, np.nan),
    np.zeros((BOARD_ROWS - 1, BOARD_COLS), dtype=np.int8),
])
def test_invalid_grid_uses_python(monkeypatch: pytest.MonkeyPatch, grid: np.ndarray) -> None:
    """不正盤面を丸めずに元のPython処理へ渡し、FFIを呼ばない。"""
    library = Mock()
    fallback = Mock(return_value=object())
    stable = Mock(return_value=True)
    monkeypatch.setattr(native, 'load_library', lambda: library)
    monkeypatch.setattr(ChainSimulator, '_simulate_uncached', fallback)
    monkeypatch.setattr(connectivity, 'is_stable', stable)
    board = Board()
    board._grid = grid
    sim = ChainSimulator(cache_enabled=False)
    with native.exact_native():
        assert sim._simulate_uncached(board) is fallback.return_value
        assert connectivity.is_stable(grid, 0)
    fallback.assert_called_once_with(sim, board)
    stable.assert_called_once_with(grid, 0)
    assert not library.mock_calls
    with pytest.raises(ValueError):
        native.packed(grid)


@pytest.mark.parametrize('hidden', [-1, BOARD_ROWS + 1, 1.5])
def test_invalid_hidden_uses_python(monkeypatch: pytest.MonkeyPatch, hidden: int) -> None:
    """隠し段の不正値をsize_tへ変換しない。"""
    library, fallback = Mock(), Mock(return_value=True)
    monkeypatch.setattr(native, 'load_library', lambda: library)
    monkeypatch.setattr(connectivity, 'is_stable', fallback)
    grid = np.zeros(SHAPE, dtype=np.int8)
    with native.exact_native():
        assert connectivity.is_stable(grid, hidden)
    fallback.assert_called_once_with(grid, hidden)
    assert not library.mock_calls


def test_body_error_is_not_swallowed(monkeypatch: pytest.MonkeyPatch) -> None:
    """開始後のOSErrorをDLL故障と取り違えて再実行しない。"""
    monkeypatch.setattr(native, 'load_library', Mock(return_value=Mock()))
    with pytest.raises(OSError, match='body'):
        with native.exact_native():
            raise OSError('body')


def test_negative_values_keep_python_result(monkeypatch: pytest.MonkeyPatch) -> None:
    """int8負値をuint8へ巻き戻さず、元のPythonと全段一致する。"""
    library = Mock()
    monkeypatch.setattr(native, 'load_library', lambda: library)
    board = Board()
    board._grid = board._grid.astype(np.int8)
    board._grid[-1, :4] = -1
    sim = ChainSimulator(cache_enabled=False)
    expected = signature(sim.simulate(board))
    with native.exact_native():
        assert signature(sim.simulate(board)) == expected
    assert not library.mock_calls


def test_missing_symbol_and_recovery(monkeypatch: pytest.MonkeyPatch,
                                     caplog: pytest.LogCaptureFixture) -> None:
    """ABI不足も復帰し、後からロードできた場合はnativeを再開する。"""
    loader = Mock(side_effect=[AttributeError('live_stable'), Mock()])
    monkeypatch.setattr(native, 'load_library', loader)
    monkeypatch.setattr(native, '_load_warning_emitted', False)
    original = ChainSimulator._simulate_uncached
    with native.exact_native():
        assert ChainSimulator._simulate_uncached is original
    with native.exact_native():
        assert ChainSimulator._simulate_uncached is not original
    assert ChainSimulator._simulate_uncached is original
    assert len(caplog.records) == 1


@pytest.mark.skipif(not native.LIBRARY.exists(), reason='DLL未ビルド')
@pytest.mark.parametrize('hidden', [False, True])
@pytest.mark.parametrize('dtype', [np.int8, np.uint8])
def test_valid_results_match_k(monkeypatch: pytest.MonkeyPatch, hidden: bool, dtype: type) -> None:
    """kと同じDLL直接呼出しとPython全段結果にバイト単位で一致する。"""
    library = native.load_library()
    rng = np.random.default_rng(20261013)
    monkeypatch.setenv('PUYO_LIVE_EXACT_NATIVE', '1')
    sim = ChainSimulator(cache_enabled=False, exclude_hidden_row_from_pop=hidden)
    for _ in range(16):
        board = Board()
        board._grid = rng.choice([0, 1, 2, 3, 4, 5, 9, 10], SHAPE).astype(dtype)
        expected = signature(sim.simulate(board))
        previous = signature(native.simulate(library, sim, board))
        with bounded_chain_caches(future_search=True):
            assert signature(sim.simulate(board)) == previous == expected
