"""ChainSimulator.simulate_probabilistic / ProbabilisticChainResult のテスト.

Phase G (C-1): 確率版シミュレーション基盤の単体テスト。
- 確定盤面 (from_board) で simulate_probabilistic ≈ simulate に一致するか
- 量子セルを含む盤面で chain_count に分布が出るか
- mle_final_board が代表盤面として機能するか
"""
from __future__ import annotations

import numpy as np
import pytest

from src.board import (
    BOARD_COLS,
    BOARD_ROWS,
    COLOR_BLUE,
    COLOR_EMPTY,
    COLOR_GREEN,
    COLOR_RED,
    COLOR_YELLOW,
    Board,
)
from src.chain import (
    PROBABILISTIC_DEFAULT_SAMPLES,
    PROBABILISTIC_SCORE_ERASE_DIVISOR,
    ChainSimulator,
    ProbabilisticChainResult,
)
from src.probabilistic_board import ProbabilisticBoard


def _build_2chain_board() -> Board:
    """確実に 2 連鎖以上発火する盤面を返す."""
    b = Board()
    # 下段 4 個赤
    b.set(12, 0, COLOR_RED)
    b.set(12, 1, COLOR_RED)
    b.set(11, 0, COLOR_RED)
    b.set(11, 1, COLOR_RED)
    # その上に乗せる 4 個青 (赤消し後に落下して連鎖)
    b.set(10, 0, COLOR_BLUE)
    b.set(10, 1, COLOR_BLUE)
    b.set(9, 0, COLOR_BLUE)
    b.set(9, 1, COLOR_BLUE)
    return b


def test_simulate_probabilistic_certain_matches_normal() -> None:
    """確定盤面なら simulate_probabilistic ≈ simulate."""
    board = _build_2chain_board()
    sim = ChainSimulator()
    normal = sim.simulate(board)
    pb = ProbabilisticBoard.from_board(board)
    prob = sim.simulate_probabilistic(pb, n_samples=4, seed=0)
    assert prob.mean_chain_count == float(normal.chain_count)
    assert prob.mean_erased_puyos == float(normal.total_erased)
    assert prob.n_samples == 4


def test_simulate_probabilistic_default_samples() -> None:
    """n_samples 省略時に PROBABILISTIC_DEFAULT_SAMPLES が使われる."""
    pb = ProbabilisticBoard.from_board(Board())
    sim = ChainSimulator()
    prob = sim.simulate_probabilistic(pb)
    assert prob.n_samples == PROBABILISTIC_DEFAULT_SAMPLES


def test_simulate_probabilistic_invalid_n_samples() -> None:
    """n_samples <= 0 で ValueError."""
    pb = ProbabilisticBoard.from_board(Board())
    sim = ChainSimulator()
    with pytest.raises(ValueError):
        sim.simulate_probabilistic(pb, n_samples=0)


def test_simulate_probabilistic_type_check() -> None:
    """ProbabilisticBoard でない引数で TypeError."""
    sim = ChainSimulator()
    with pytest.raises(TypeError):
        sim.simulate_probabilistic(Board(), n_samples=2)  # type: ignore[arg-type]


def test_simulate_probabilistic_quantum_cells_distribute() -> None:
    """量子セルを含むと chain_count に分布が出る (std > 0 が起こりうる)."""
    pb = ProbabilisticBoard()
    # 下段 1 列を確実に赤、別列を 50% 赤 / 50% 青で揺らす
    pb.set_certain(12, 0, COLOR_RED)
    pb.set_certain(12, 1, COLOR_RED)
    pb.set_certain(11, 0, COLOR_RED)
    pb.set_distribution(
        11, 1, {COLOR_RED: 0.5, COLOR_BLUE: 0.5},
    )
    sim = ChainSimulator()
    prob = sim.simulate_probabilistic(pb, n_samples=20, seed=42)
    chain_counts = [s.chain_count for s in prob.samples]
    # 0 と >0 の両方が出るはず
    assert any(c == 0 for c in chain_counts)
    assert any(c >= 1 for c in chain_counts)
    assert prob.std_chain_count > 0.0


def test_probabilistic_chain_result_dataclass() -> None:
    """ProbabilisticChainResult は dataclass で各属性アクセス可能."""
    sim = ChainSimulator()
    pb = ProbabilisticBoard.from_board(_build_2chain_board())
    prob = sim.simulate_probabilistic(pb, n_samples=3, seed=1)
    assert isinstance(prob, ProbabilisticChainResult)
    assert isinstance(prob.mean_chain_count, float)
    assert isinstance(prob.mean_erased_puyos, float)
    assert isinstance(prob.mean_score, float)
    assert isinstance(prob.samples, list)
    assert isinstance(prob.mle_final_board, Board)
    # 簡易代理スコア = chain + erased / 10 が想定通り
    expected = (
        prob.mean_chain_count
        + prob.mean_erased_puyos / PROBABILISTIC_SCORE_ERASE_DIVISOR
    )
    assert abs(prob.mean_score - expected) < 1e-9


def test_simulate_probabilistic_seed_reproducibility() -> None:
    """同じ seed なら結果が再現する."""
    pb = ProbabilisticBoard()
    pb.set_distribution(12, 0, {COLOR_RED: 0.3, COLOR_BLUE: 0.7})
    sim = ChainSimulator()
    a = sim.simulate_probabilistic(pb, n_samples=5, seed=123)
    b = sim.simulate_probabilistic(pb, n_samples=5, seed=123)
    assert a.mean_chain_count == b.mean_chain_count
    assert a.mean_erased_puyos == b.mean_erased_puyos


def test_mle_final_board_shape() -> None:
    """mle_final_board の形状確認."""
    sim = ChainSimulator()
    pb = ProbabilisticBoard.from_board(_build_2chain_board())
    prob = sim.simulate_probabilistic(pb, n_samples=2, seed=0)
    assert prob.mle_final_board._grid.shape == (BOARD_ROWS, BOARD_COLS)


def test_to_max_likelihood_board_consistency() -> None:
    """確定盤面の to_max_likelihood_board は元盤面と一致."""
    board = _build_2chain_board()
    pb = ProbabilisticBoard.from_board(board)
    mle = pb.to_max_likelihood_board()
    for r in range(BOARD_ROWS):
        for c in range(BOARD_COLS):
            assert mle.get(r, c) == board.get(r, c)


def test_sample_board_certain_deterministic() -> None:
    """全セル確定の ProbabilisticBoard で sample_board は一意."""
    board = _build_2chain_board()
    pb = ProbabilisticBoard.from_board(board)
    rng = np.random.default_rng(0)
    sampled = pb.sample_board(rng=rng)
    for r in range(BOARD_ROWS):
        for c in range(BOARD_COLS):
            assert sampled.get(r, c) == board.get(r, c)


def test_sample_board_distribution_picks_with_weights() -> None:
    """distribution セルは確率に応じて選ばれる (高確率色が多い)."""
    pb = ProbabilisticBoard()
    pb.set_distribution(12, 0, {COLOR_RED: 0.9, COLOR_BLUE: 0.1})
    rng = np.random.default_rng(0)
    n_red = 0
    n_total = 200
    for _ in range(n_total):
        sampled = pb.sample_board(rng=rng)
        if sampled.get(12, 0) == COLOR_RED:
            n_red += 1
    # 90% に近い (誤差 ±10%)
    assert n_red / n_total > 0.75
