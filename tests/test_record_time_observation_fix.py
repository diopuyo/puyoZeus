"""記録時 観測優先 (2026-09-19、既定 OFF) の挙動を固定する.

なぜこの機能があるか
--------------------
記録されるのは「前回の記録と盤面が変わった瞬間」だけ (`last_emitted_grid`)。
一方、確定盤面を画面へ合わせ直せるのは事後復旧ゲートだけで、そこは
**8 処理frame 連続で CNN==HSV が一致すること**を要求する。
つまり **盤面が変わった瞬間には、復旧ゲートは原理的に間に合わない。**

5動画の実測 (母数 15,413 区間): 記録された盤面の食い違いのうち、
盤面が画面へ寄って解けたものが **52.08%**、画面が盤面へ寄ったものが **15.58%**。
記録の場面では画面の方が正しいことが 3.3 倍多い。

**既定 OFF**。本番採用は user 承認を得てから。
"""
from __future__ import annotations

import pytest
import numpy as np
from pathlib import Path
from typing import Any

from scripts import collect_boards_lean as CBL
from src.board import Board, BOARD_ROWS, COLOR_UNKNOWN
from src.board_state_machine import BoardState
from src.chain import ChainSimulator
from src.self_supervised.physical_consistency import check_gravity_rule

BOTTOM_ROW = BOARD_ROWS - 1
ABOVE_ROW = BOTTOM_ROW - 1
TEST_COLUMN = 0


class _Reader:
    """HSV 盤面を差し替えるための最小の読み取り器。"""

    def __init__(self, hsv: Board) -> None:
        self._hsv = hsv

    def read_board_hsv_only(self, frame: Any, region: Any) -> Board:
        return self._hsv


def _fix(board: Board, cnn: Board, hsv: Board | None = None) -> tuple[Board, int]:
    return CBL._apply_record_time_observation_fix(
        board, "1P", cnn, object(), _Reader(hsv if hsv is not None else cnn))


def test_fills_a_cell_that_has_support_below() -> None:
    """下が埋まっていれば、画面に見えるぷよを書き足す。"""
    board = Board()
    board.set(12, 0, 1)
    obs = Board()
    obs.set(12, 0, 1)
    obs.set(11, 0, 3)
    fixed, n = _fix(board, obs)
    assert n == 1
    assert fixed.get(11, 0) == 3


def test_rejects_a_cell_that_would_float() -> None:
    """下に何も無いセルは書き足さない (浮きぷよを作らない)。"""
    board = Board()
    obs = Board()
    obs.set(10, 0, 3)
    fixed, n = _fix(board, obs)
    assert n == 0
    assert fixed.get(10, 0) == 0


def test_fills_a_whole_column_because_candidates_support_each_other() -> None:
    """列に連なっていれば、下から順にまとめて書き足せる。

    ここが列デッドロックの解消点。1セルずつ見ると下が空で拒否されるが、
    候補同士は支え合えるので列単位なら通る。
    """
    board = Board()
    obs = Board()
    obs.set(12, 0, 1)
    obs.set(11, 0, 3)
    obs.set(10, 0, 4)
    fixed, n = _fix(board, obs)
    assert n == 3
    assert [fixed.get(r, 0) for r in (12, 11, 10)] == [1, 3, 4]


def test_removes_a_puyo_that_is_not_on_screen() -> None:
    """列の一番上を消す安全な補正は維持する。"""
    board = Board()
    board.set(12, 0, 1)
    board.set(11, 0, 5)
    obs = Board()
    obs.set(12, 0, 1)
    fixed, n = _fix(board, obs)
    assert n == 1
    assert fixed.get(11, 0) == 0


def test_does_nothing_when_the_two_observers_disagree() -> None:
    """CNN と HSV が割れている間は動かさない (片方だけを信じない)。"""
    board = Board()
    board.set(12, 0, 1)
    cnn = Board()
    cnn.set(12, 0, 1)
    cnn.set(11, 0, 3)
    hsv = Board()
    hsv.set(12, 0, 1)
    hsv.set(11, 0, 4)
    fixed, n = _fix(board, cnn, hsv)
    assert n == 0
    assert fixed.get(11, 0) == 0


def test_does_not_touch_the_hidden_row() -> None:
    """隠し段 (row 0) は HSV 側が推論で埋めるので触らない。"""
    board = Board()
    for r in range(1, 13):
        board.set(r, 0, 1)
    obs = Board()
    for r in range(1, 13):
        obs.set(r, 0, 1)
    obs.set(0, 0, 3)
    fixed, n = _fix(board, obs)
    assert n == 0
    assert fixed.get(0, 0) == 0


def test_returns_the_same_board_when_inputs_are_missing() -> None:
    """観測が無ければ何もしない (fail-safe)。"""
    board = Board()
    board.set(12, 0, 1)
    same, n = CBL._apply_record_time_observation_fix(
        board, "1P", None, None, None)
    assert n == 0
    assert same.get(12, 0) == 1


@pytest.mark.parametrize("flag", [
    "enable_record_time_observation_fix", "enable_record_time_observation_fix_raw",
])
def test_flag_defaults_to_off_everywhere(flag: str) -> None:
    """既定 OFF を、口・収集本体の両方で固定する (勝手に本番へ入らないこと)。"""
    import inspect

    assert inspect.signature(CBL.collect_lean).parameters[flag].default is False
    assert inspect.signature(
        CBL._process_side_lean).parameters[flag].default is False


@pytest.mark.parametrize("existing_top", [0, 2])
def test_removed_support_never_creates_floating_record(
    existing_top: int, tmp_path: Path,
) -> None:
    """支えの削除と追加を同時評価し、実NPZにも浮きを保存しない。"""
    board, obs = Board(), Board()
    board.set(BOTTOM_ROW, TEST_COLUMN, 1)
    board.set(ABOVE_ROW, TEST_COLUMN, existing_top)
    obs.set(ABOVE_ROW, TEST_COLUMN, 2)
    # 別列の正常ぷよを残し、全消し除外による保存0件を合格にしない。
    board.set(BOTTOM_ROW, TEST_COLUMN + 1, 3)
    obs.set(BOTTOM_ROW, TEST_COLUMN + 1, 3)
    original = board.grid_bytes()
    fixed, _ = _fix(board, obs)
    assert check_gravity_rule(fixed)[0]
    assert board.grid_bytes() == original
    acc, state = CBL._LeanNpzAccumulator(), CBL._SideState()
    state.prev_tsumo_count = 1
    CBL._process_side_lean(
        acc, state, "1P", board, BoardState.STABLE, 100, "test", 1.0, 30,
        tsumo_count=2, exclude_phantom=True, enable_move_segmented_recording=True,
        enable_physics_persistence_filter=True, physics_sim=ChainSimulator(),
        enable_record_time_observation_fix_raw=True, raw_cnn_board=obs,
        frame_bgr=object(), image_reader=_Reader(obs),
    )
    acc.assign_won_labels({0: {"1P": 100, "2P": 0}})
    path = tmp_path / "recorded.npz"
    acc.save(path)
    with np.load(path) as saved:
        assert len(saved["grids"]) == 1
        assert all(check_gravity_rule(Board.from_list(grid.tolist()))[0]
                   for grid in saved["grids"])


def test_rejected_column_keeps_other_column_correction() -> None:
    """支えを壊す列だけを戻し、独立な列の正常補正は残す。"""
    board, obs = Board(), Board()
    board.set(BOTTOM_ROW, 0, 1)
    board.set(ABOVE_ROW, 0, 2)
    obs.set(ABOVE_ROW, 0, 2)
    obs.set(BOTTOM_ROW, 1, 3)
    fixed, count = _fix(board, obs)
    assert fixed.get(BOTTOM_ROW, 0) == 1
    assert fixed.get(BOTTOM_ROW, 1) == 3
    assert count == 1 and check_gravity_rule(fixed)[0]


@pytest.mark.parametrize("normal,raw,expected", [
    (False, True, 1), (True, True, 1), (True, False, 2),
])
def test_missing_raw_observation_never_falls_back_to_filtered(
    normal: bool, raw: bool, expected: int, tmp_path: Path,
) -> None:
    """raw欠測時は確定盤面を保ち、通常モードだけfilteredで補正する。"""
    board, observed = Board(), Board()
    board.set(BOTTOM_ROW, TEST_COLUMN, 1)
    observed.set(BOTTOM_ROW, TEST_COLUMN, 2)
    original = board.grid_bytes()
    acc, state = CBL._LeanNpzAccumulator(), CBL._SideState()
    state.prev_tsumo_count = 1
    CBL._process_side_lean(
        acc, state, "1P", board, BoardState.STABLE, 100, "test", 1.0, 30,
        tsumo_count=2, exclude_phantom=True, enable_move_segmented_recording=True,
        enable_record_time_observation_fix=normal,
        enable_record_time_observation_fix_raw=raw, raw_cnn_board=None,
        cnn_board=observed, frame_bgr=object(), image_reader=_Reader(observed),
    )
    acc.assign_won_labels({0: {"1P": 100, "2P": 0}})
    path = tmp_path / "raw_missing.npz"
    acc.save(path)
    with np.load(path) as saved:
        assert len(saved["grids"]) == 1
        assert saved["grids"][0][BOTTOM_ROW][TEST_COLUMN] == expected
    assert board.grid_bytes() == original


def test_unknown_and_hidden_row_keep_existing_gravity_semantics() -> None:
    """UNKNOWNは観測保留、隠し段は補正対象外という既存契約を維持する。"""
    board = Board()
    board.set(BOTTOM_ROW, 0, COLOR_UNKNOWN)
    board.set(0, 1, 4)
    obs = board.copy()
    obs.set(ABOVE_ROW, 0, 2)
    obs.set(0, 1, 0)
    fixed, count = _fix(board, obs)
    assert fixed.get(BOTTOM_ROW, 0) == COLOR_UNKNOWN
    assert fixed.get(ABOVE_ROW, 0) == 2
    assert fixed.get(0, 1) == 4
    assert count == 1
