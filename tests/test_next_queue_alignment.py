"""NEXT 意味ずれ補正 (学習行の補正・提供側整列・既定OFF) のテスト (2026-10-01)。"""
from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from src.board import Board
from src.board_state_machine import BoardState
from src.exchange_event_evaluator import shared_model_directory
from src.exchange_event_overlay import ExchangeEventOverlay
from src.next_queue_alignment import (
    KIND_CHAIN, KIND_MULTI, KIND_OJAMA, KIND_PLACE, KIND_SAME, METHOD_INVALID, METHOD_KEEP,
    METHOD_SHIFT, causal_row_queue, correct_sequence, displayed_slots, pair_key,
    transition_kind, truth_row_queue, truth_sequence,
)
from src.next_queue_serving import FrozenQueue, QueueAlignment

ROWS, COLS = 13, 6


def grid(cells: dict[tuple[int, int], int]) -> np.ndarray:
    """指定セルだけ埋めた盤面。"""
    g = np.zeros((ROWS, COLS), dtype=np.int8)
    for (r, c), v in cells.items():
        g[r, c] = v
    return g


def test_transition_kinds() -> None:
    a = grid({(12, 0): 1})
    assert transition_kind(a, grid({(12, 0): 1, (12, 1): 2, (11, 1): 3})) == (KIND_PLACE, (2, 3))
    assert transition_kind(a, grid({}))[0] == KIND_CHAIN
    assert transition_kind(a, grid({(12, 0): 1, (12, 1): 9}))[0] == KIND_OJAMA
    assert transition_kind(a, grid({(12, 0): 1, (12, 1): 2, (11, 1): 3, (12, 2): 4}))[0] == KIND_MULTI
    assert transition_kind(a, a.copy())[0] == KIND_SAME


def test_causal_rule_restores_in_hand_pair_after_chain() -> None:
    prev = np.array([1, 2, 3, 4], np.int8)      # 連鎖中: next=P_k(1,2), dnext=P_k+1(3,4)
    slid = np.array([3, 4, 1, 1], np.int8)      # 連鎖後: 繰り上がり済み
    out, method = causal_row_queue(KIND_CHAIN, prev, slid)
    assert method == METHOD_SHIFT and out.tolist() == [1, 2, 3, 4]
    out, method = causal_row_queue(KIND_PLACE, prev, slid)   # 通常設置の後はそのまま
    assert method == METHOD_KEEP and out.tolist() == slid.tolist()
    out, method = causal_row_queue(KIND_CHAIN, prev, prev)   # 繰り上がっていない
    assert method == METHOD_KEEP and out.tolist() == prev.tolist()
    out, method = causal_row_queue(KIND_CHAIN, prev, np.array([0, 4, 1, 1], np.int8))
    assert method == METHOD_INVALID


def random_sequence(rng: np.random.Generator, n: int) -> tuple[np.ndarray, np.ndarray]:
    """設置・消去が混ざる盤面列と、ずれ・誤読が混ざる queue 列。"""
    grids, queues = [np.zeros((ROWS, COLS), np.int8)], []
    for _ in range(n - 1):
        g = grids[-1].copy()
        if rng.random() < .3 and g.any():
            g[rng.integers(ROWS), :] = 0
        else:
            col = rng.integers(COLS)
            g[ROWS - 1 - rng.integers(3), col] = rng.integers(1, 5)
            g[ROWS - 4, (col + 1) % COLS] = rng.integers(1, 5)
        grids.append(g)
    queues = rng.integers(0, 5, size=(n, 4)).astype(np.int8)
    return np.stack(grids), queues


@pytest.mark.parametrize("seed", range(5))
def test_outputs_depend_only_on_past_readings(seed: int) -> None:
    """打ち切り再生: 行 r の出力は queue[:r+1] と盤面[:r+3] だけで決まる (未来の表示の読みは使わない)。"""
    rng = np.random.default_rng(seed)
    grids, queues = random_sequence(rng, 40)
    causal, _ = correct_sequence(grids, queues)
    full, _, _ = truth_sequence(grids, queues)
    for r in range(len(grids)):
        c_short, _ = correct_sequence(grids[:r + 1], queues[:r + 1])
        assert np.array_equal(c_short[r], causal[r])
        t_short, _ = truth_row_queue(r, grids[:r + 3], queues[:r + 1], c_short[r])
        assert np.array_equal(t_short, full[r])


@pytest.mark.parametrize("seed", range(5))
def test_truth_selection_only_uses_displayed_pairs(seed: int) -> None:
    """真値選択の出力色は、t 以前に表示枠で読めた組か因果規則の出力のどちらか。"""
    rng = np.random.default_rng(seed + 10)
    grids, queues = random_sequence(rng, 40)
    causal, _ = correct_sequence(grids, queues)
    out, _, chosen = truth_sequence(grids, queues)
    for r in range(len(grids)):
        shown = {pair_key(*p) for p in displayed_slots(queues, r, 2)}
        for slot in range(2):
            pair = tuple(int(v) for v in out[r, 2 * slot:2 * slot + 2])
            if chosen[r, slot]:
                assert pair_key(*pair) in shown
            else:
                assert pair == tuple(int(v) for v in causal[r, 2 * slot:2 * slot + 2])


def test_truth_selection_picks_placed_pair_from_previous_display() -> None:
    """連鎖後の行: 手中の組は前の行の next にだけ出ている → それを選ぶ。"""
    g0 = grid({(12, 0): 1, (12, 1): 1, (12, 2): 1, (11, 0): 1})
    g1 = grid({(12, 5): 2})                      # 連鎖で消えた後
    g2 = grid({(12, 5): 2, (12, 3): 3, (11, 3): 4})   # P_k=(3,4) を設置
    g3 = grid({(12, 5): 2, (12, 3): 3, (11, 3): 4, (12, 0): 1, (11, 0): 2})  # P_k+1=(1,2)
    grids = np.stack([g0, g1, g2, g3])
    queues = np.array([[3, 4, 1, 2], [3, 4, 1, 2], [1, 2, 5, 5], [5, 5, 1, 1]], np.int8)
    out, _, chosen = truth_sequence(grids, queues)
    assert out[1].tolist() == [3, 4, 1, 2] and chosen[1].tolist() == [1, 1]


def test_frozen_queue_keeps_segment_start_and_restores_after_chain() -> None:
    board_a, board_b = grid({(12, 0): 1}), grid({(12, 0): 1, (12, 1): 2, (11, 1): 3})
    aligner = FrozenQueue()
    assert aligner.update(board_a, (1, 2, 3, 4)) == (1, 2, 3, 4)        # 確定前は生
    assert aligner.update(board_a, (1, 2, 3, 4)) == (1, 2, 3, 4)        # 確定・凍結
    assert aligner.update(board_a, (3, 4, 5, 5)) == (1, 2, 3, 4)        # 繰り上がっても凍結
    assert aligner.update(board_a, (3, 4, 5, 5)) == (1, 2, 3, 4)
    aligner.leave()                                                     # 連鎖 (非STABLE)
    aligner.update(board_b, (5, 5, 1, 1))
    assert aligner.update(board_b, (5, 5, 1, 1)) == (3, 4, 5, 5)        # 手中の組 (3,4) を戻す


def test_queue_alignment_resets_per_game_and_skips_non_stable() -> None:
    board = Board.from_list(grid({(12, 0): 1}).tolist())
    stable = SimpleNamespace(state=BoardState.STABLE, confirmed_board=board, next_pair=(1, 2), dnext_pair=(3, 4))
    moving = SimpleNamespace(state="CHAIN", confirmed_board=board, next_pair=(1, 2), dnext_pair=(3, 4))
    align = QueueAlignment()
    assert align.observe((stable, moving), 0, BoardState.STABLE) == ((1, 2, 3, 4), None)
    first = align.sides[0]
    align.observe((stable, moving), 1, BoardState.STABLE)
    assert align.sides[0] is not first


def fake_side(queue: tuple[int, int, int, int]) -> SimpleNamespace:
    return SimpleNamespace(state=BoardState.STABLE, confirmed_board=Board(),
                           next_pair=queue[:2], dnext_pair=queue[2:])


def test_overlay_default_off_keeps_raw_queue() -> None:
    """既定OFF: 履歴の queue は従来どおり生の next/dnext。"""
    overlay = ExchangeEventOverlay(SimpleNamespace(count_features=True), None, None)
    assert overlay._queue_alignment is None
    overlay._remember((fake_side((1, 2, 3, 4)), fake_side((5, 5, 1, 1))), None, 1.)
    assert overlay._history[0][-1].queue.tolist() == [1, 2, 3, 4]
    assert overlay._history[1][-1].queue.tolist() == [5, 5, 1, 1]


def test_overlay_on_uses_aligned_queue() -> None:
    overlay = ExchangeEventOverlay(SimpleNamespace(count_features=True), None, None, queue_alignment=True)
    overlay._aligned_queue = ((9, 9, 9, 9), None)
    overlay._remember((fake_side((1, 2, 3, 4)), fake_side((5, 5, 1, 1))), None, 1.)
    assert overlay._history[0][-1].queue.tolist() == [9, 9, 9, 9]
    assert overlay._history[1][-1].queue.tolist() == [5, 5, 1, 1]


def test_shared_directory_key_is_optional(tmp_path) -> None:
    (tmp_path / "old").mkdir()
    (tmp_path / "old" / "manifest.json").write_text(json.dumps(dict(models=dict(S1_prime_light={}))))
    assert shared_model_directory(tmp_path / "old") == tmp_path / "exchange_event_v1"
    (tmp_path / "new").mkdir()
    (tmp_path / "new" / "manifest.json").write_text(json.dumps(dict(
        models=dict(S1_prime_light={}), shared_directory="exchange_event_v5_common")))
    assert shared_model_directory(tmp_path / "new") == tmp_path / "exchange_event_v5_common"
