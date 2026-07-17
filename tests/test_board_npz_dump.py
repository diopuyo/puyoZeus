"""盤面グリッド npz ダンプと build_board_pairs のユニットテスト。

テスト方針:
- 動画認識・重い collect 実行は一切しない。
- 合成 Board・合成 npz・合成 labeled_win.csv でロジックのみ検証する。
"""
from __future__ import annotations

import math
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.board import (
    BOARD_COLS,
    BOARD_ROWS,
    COLOR_RED,
    COLOR_BLUE,
    COLOR_GREEN,
    Board,
)


# ============================
# ヘルパ: 合成盤面ビルダー
# ============================

def _make_board(color: int = COLOR_RED) -> Board:
    """下段 2 行を指定色で埋めた合成盤面を返す。"""
    g = [[0] * BOARD_COLS for _ in range(BOARD_ROWS)]
    for col in range(BOARD_COLS):
        g[12][col] = color
        g[11][col] = color
    return Board.from_list(g)


# ============================
# _BoardNpzAccumulator のテスト
# ============================

class TestBoardNpzAccumulator:
    """_BoardNpzAccumulator の append / save / shape を検証する。"""

    def _import_acc(self):
        """collect_indicators_v2 から _BoardNpzAccumulator をインポートする。"""
        import importlib, sys
        # モジュールが未登録でも動くよう明示 import
        import scripts.collect_indicators_v2 as mod
        return mod._BoardNpzAccumulator

    def test_empty_accumulator_save(self, tmp_path: Path) -> None:
        """空バッファを保存しても npz が生成されること。"""
        Acc = self._import_acc()
        acc = Acc()
        out = tmp_path / "empty.npz"
        acc.save(out)
        data = np.load(str(out), allow_pickle=True)
        assert data["grids"].shape == (0,)  # 空配列

    def test_single_append_shape(self, tmp_path: Path) -> None:
        """1 件 append → npz grids が (1, 13, 6) int8 になること。"""
        Acc = self._import_acc()
        acc = Acc()
        board = _make_board(COLOR_RED)
        acc.append(board._grid, "v29", "1P", 12.5, 0, 100)
        out = tmp_path / "single.npz"
        acc.save(out)
        data = np.load(str(out), allow_pickle=True)
        assert data["grids"].shape == (1, BOARD_ROWS, BOARD_COLS)
        assert data["grids"].dtype == np.int8

    def test_multiple_append_order(self, tmp_path: Path) -> None:
        """複数 append の順序が npz に保持されること。"""
        Acc = self._import_acc()
        acc = Acc()
        board_r = _make_board(COLOR_RED)
        board_b = _make_board(COLOR_BLUE)
        board_g = _make_board(COLOR_GREEN)
        acc.append(board_r._grid, "v29", "1P", 10.0, 0, 10)
        acc.append(board_b._grid, "v29", "2P", 10.1, 0, 10)
        acc.append(board_g._grid, "v29", "1P", 20.0, 1, 20)
        out = tmp_path / "multi.npz"
        acc.save(out)
        data = np.load(str(out), allow_pickle=True)
        grids = data["grids"]
        assert grids.shape == (3, BOARD_ROWS, BOARD_COLS)
        # 順序確認: 0 番目は赤ぷよのみ (下段 2 行)
        assert grids[0, 12, 0] == COLOR_RED
        assert grids[1, 12, 0] == COLOR_BLUE
        assert grids[2, 12, 0] == COLOR_GREEN

    def test_meta_arrays_match_grids(self, tmp_path: Path) -> None:
        """メタ配列 (video_id / side / t_sec / game_idx / frame_idx) が grids と同長。"""
        Acc = self._import_acc()
        acc = Acc()
        for i in range(5):
            acc.append(_make_board()._grid, f"v{i:02d}", "1P", float(i * 10), i, i * 30)
        out = tmp_path / "meta.npz"
        acc.save(out)
        data = np.load(str(out), allow_pickle=True)
        n = data["grids"].shape[0]
        assert len(data["video_id"]) == n
        assert len(data["side"]) == n
        assert len(data["t_sec"]) == n
        assert len(data["game_idx"]) == n
        assert len(data["frame_idx"]) == n

    def test_grid_values_preserved(self, tmp_path: Path) -> None:
        """append した盤面色値が npz に正確に保存されること。"""
        Acc = self._import_acc()
        acc = Acc()
        board = _make_board(COLOR_GREEN)
        acc.append(board._grid, "v30", "2P", 5.0, 0, 50)
        out = tmp_path / "vals.npz"
        acc.save(out)
        data = np.load(str(out), allow_pickle=True)
        # 下段 2 行が全て COLOR_GREEN=3
        restored = data["grids"][0]
        assert int(restored[12, 0]) == COLOR_GREEN
        assert int(restored[11, 3]) == COLOR_GREEN
        # 上段は 0 (EMPTY)
        assert int(restored[0, 0]) == 0

    def test_grid_is_copied_not_referenced(self, tmp_path: Path) -> None:
        """append 後に元 grid を書き換えても npz の値が変わらないこと (copy)。"""
        Acc = self._import_acc()
        acc = Acc()
        board = _make_board(COLOR_RED)
        grid_ref = board._grid
        acc.append(grid_ref, "v29", "1P", 1.0, 0, 1)
        # append 後に書き換え
        board._grid[12, 0] = COLOR_BLUE
        out = tmp_path / "copy_test.npz"
        acc.save(out)
        data = np.load(str(out), allow_pickle=True)
        # npz の値は変わっていないはず
        assert int(data["grids"][0, 12, 0]) == COLOR_RED

    def test_side_values_stored_correctly(self, tmp_path: Path) -> None:
        """side 文字列が '1P' / '2P' として正確に保存されること。"""
        Acc = self._import_acc()
        acc = Acc()
        acc.append(_make_board()._grid, "v29", "1P", 1.0, 0, 1)
        acc.append(_make_board()._grid, "v29", "2P", 1.1, 0, 1)
        out = tmp_path / "side.npz"
        acc.save(out)
        data = np.load(str(out), allow_pickle=True)
        sides = data["side"].tolist()
        assert sides[0] == "1P"
        assert sides[1] == "2P"


# ============================
# build_board_pairs のテスト
# ============================

def _write_npz(
    path: Path,
    grids: np.ndarray,
    video_ids: list[str],
    sides: list[str],
    t_secs: list[float],
    game_idxs: list[int],
    frame_idxs: list[int],
) -> None:
    """テスト用 npz を書き出すヘルパ。"""
    np.savez_compressed(
        str(path),
        grids=grids.astype(np.int8),
        video_id=np.array(video_ids),
        side=np.array(sides),
        t_sec=np.array(t_secs, dtype=np.float32),
        game_idx=np.array(game_idxs, dtype=np.int32),
        frame_idx=np.array(frame_idxs, dtype=np.int32),
    )


def _write_labeled_win(
    path: Path,
    rows: list[dict],
) -> None:
    """テスト用 labeled_win.csv を書き出すヘルパ。空リストでもヘッダ行を出力する。"""
    cols = ["video_id", "side", "t_sec", "game_idx", "won"]
    df = pd.DataFrame(rows, columns=cols) if rows else pd.DataFrame(columns=cols)
    df.to_csv(str(path), index=False)


class TestBuildBoardPairs:
    """build_board_pairs.py のペア化ロジックを検証する。"""

    def _import_build(self):
        import scripts.build_board_pairs as mod
        return mod

    def _make_grid(self, color: int) -> np.ndarray:
        g = np.zeros((BOARD_ROWS, BOARD_COLS), dtype=np.int8)
        g[12, :] = color
        return g

    def test_basic_pairing(self, tmp_path: Path) -> None:
        """1P/2P が近傍 t_sec でペア化され 1 ペアになること。"""
        mod = self._import_build()
        # 盤面グリッド: 1P=赤, 2P=青
        grid_1p = self._make_grid(COLOR_RED)
        grid_2p = self._make_grid(COLOR_BLUE)
        grids = np.stack([grid_1p, grid_2p])
        npz_dir = tmp_path / "npz"
        npz_dir.mkdir()
        _write_npz(
            npz_dir / "v29.npz",
            grids=grids,
            video_ids=["v29", "v29"],
            sides=["1P", "2P"],
            t_secs=[10.0, 10.2],
            game_idxs=[0, 0],
            frame_idxs=[100, 100],
        )
        _write_labeled_win(
            tmp_path / "labeled_win.csv",
            [
                {"video_id": "v29", "side": "1P", "t_sec": 10.0, "game_idx": 0, "won": 1.0},
                {"video_id": "v29", "side": "2P", "t_sec": 10.2, "game_idx": 0, "won": 0.0},
            ],
        )
        result = mod.build_pairs(npz_dir, tmp_path / "labeled_win.csv")
        assert len(result.board_1p) == 1
        assert len(result.board_2p) == 1
        assert int(result.board_1p[0, 12, 0]) == COLOR_RED
        assert int(result.board_2p[0, 12, 0]) == COLOR_BLUE

    def test_won_label_attached(self, tmp_path: Path) -> None:
        """won ラベルが 1P 視点で正しく付与されること。"""
        mod = self._import_build()
        grids = np.stack([self._make_grid(COLOR_RED), self._make_grid(COLOR_BLUE)])
        npz_dir = tmp_path / "npz"
        npz_dir.mkdir()
        _write_npz(
            npz_dir / "v29.npz",
            grids=grids,
            video_ids=["v29", "v29"],
            sides=["1P", "2P"],
            t_secs=[5.0, 5.0],
            game_idxs=[0, 0],
            frame_idxs=[50, 50],
        )
        _write_labeled_win(
            tmp_path / "labeled_win.csv",
            [
                {"video_id": "v29", "side": "1P", "t_sec": 5.0, "game_idx": 0, "won": 1.0},
                {"video_id": "v29", "side": "2P", "t_sec": 5.0, "game_idx": 0, "won": 0.0},
            ],
        )
        result = mod.build_pairs(npz_dir, tmp_path / "labeled_win.csv")
        assert len(result.won) == 1
        assert float(result.won[0]) == 1.0  # 1P 側 won

    def test_too_far_t_sec_not_paired(self, tmp_path: Path) -> None:
        """t_sec 差が MAX_PAIR_T_DIFF_SEC を超えるとペア化されないこと。"""
        mod = self._import_build()
        grids = np.stack([self._make_grid(COLOR_RED), self._make_grid(COLOR_BLUE)])
        npz_dir = tmp_path / "npz"
        npz_dir.mkdir()
        _write_npz(
            npz_dir / "v29.npz",
            grids=grids,
            video_ids=["v29", "v29"],
            sides=["1P", "2P"],
            t_secs=[10.0, 13.0],   # 差 3.0s > MAX_PAIR_T_DIFF_SEC=2.0
            game_idxs=[0, 0],
            frame_idxs=[100, 100],
        )
        _write_labeled_win(tmp_path / "labeled_win.csv", [])
        result = mod.build_pairs(npz_dir, tmp_path / "labeled_win.csv")
        assert len(result.board_1p) == 0

    def test_different_game_idx_not_paired(self, tmp_path: Path) -> None:
        """game_idx が異なる 1P/2P はペア化されないこと。"""
        mod = self._import_build()
        grids = np.stack([self._make_grid(COLOR_RED), self._make_grid(COLOR_BLUE)])
        npz_dir = tmp_path / "npz"
        npz_dir.mkdir()
        _write_npz(
            npz_dir / "v29.npz",
            grids=grids,
            video_ids=["v29", "v29"],
            sides=["1P", "2P"],
            t_secs=[10.0, 10.1],
            game_idxs=[0, 1],  # game_idx 違い
            frame_idxs=[100, 100],
        )
        _write_labeled_win(tmp_path / "labeled_win.csv", [])
        result = mod.build_pairs(npz_dir, tmp_path / "labeled_win.csv")
        assert len(result.board_1p) == 0

    def test_multiple_pairs_multiple_videos(self, tmp_path: Path) -> None:
        """複数動画・複数 game_idx からペアが正しく結合されること。"""
        mod = self._import_build()
        npz_dir = tmp_path / "npz"
        npz_dir.mkdir()
        # v29: game_idx=0 で 2 ペア
        g_v29 = np.stack([
            self._make_grid(COLOR_RED),    # 1P t=10
            self._make_grid(COLOR_BLUE),   # 2P t=10
            self._make_grid(COLOR_GREEN),  # 1P t=20
            self._make_grid(COLOR_RED),    # 2P t=20
        ])
        _write_npz(
            npz_dir / "v29.npz",
            grids=g_v29,
            video_ids=["v29"] * 4,
            sides=["1P", "2P", "1P", "2P"],
            t_secs=[10.0, 10.0, 20.0, 20.0],
            game_idxs=[0, 0, 0, 0],
            frame_idxs=[100, 100, 200, 200],
        )
        # v30: game_idx=0 で 1 ペア
        g_v30 = np.stack([
            self._make_grid(COLOR_BLUE),
            self._make_grid(COLOR_GREEN),
        ])
        _write_npz(
            npz_dir / "v30.npz",
            grids=g_v30,
            video_ids=["v30", "v30"],
            sides=["1P", "2P"],
            t_secs=[5.0, 5.0],
            game_idxs=[0, 0],
            frame_idxs=[50, 50],
        )
        _write_labeled_win(tmp_path / "labeled_win.csv", [])
        result = mod.build_pairs(npz_dir, tmp_path / "labeled_win.csv")
        # v29 x 2 + v30 x 1 = 合計 3 ペア
        assert len(result.board_1p) == 3

    def test_won_nan_when_no_label(self, tmp_path: Path) -> None:
        """labeled_win.csv に対応エントリがない場合 won=NaN になること。"""
        mod = self._import_build()
        grids = np.stack([self._make_grid(COLOR_RED), self._make_grid(COLOR_BLUE)])
        npz_dir = tmp_path / "npz"
        npz_dir.mkdir()
        _write_npz(
            npz_dir / "v29.npz",
            grids=grids,
            video_ids=["v29", "v29"],
            sides=["1P", "2P"],
            t_secs=[10.0, 10.0],
            game_idxs=[0, 0],
            frame_idxs=[100, 100],
        )
        # labeled_win に v29 の行なし
        _write_labeled_win(tmp_path / "labeled_win.csv", [
            {"video_id": "v99", "side": "1P", "t_sec": 10.0, "game_idx": 0, "won": 1.0},
        ])
        result = mod.build_pairs(npz_dir, tmp_path / "labeled_win.csv")
        assert len(result.won) == 1
        assert math.isnan(float(result.won[0]))

    def test_output_shapes(self, tmp_path: Path) -> None:
        """board_1p / board_2p / won / t_sec / game_idx の shape が揃うこと。

        1P/2P の t_sec 差を 0.1s に設定し、確実に 3 ペアが生成されることを確認する。
        """
        mod = self._import_build()
        n = 3
        # 1P: t=0, 10, 20  /  2P: t=0.1, 10.1, 20.1  → 差 0.1s < 2.0s で全ペア成立
        sides_list = []
        t_secs_list = []
        for i in range(n):
            sides_list.extend(["1P", "2P"])
            t_secs_list.extend([float(i * 10), float(i * 10) + 0.1])
        grids = np.stack([
            self._make_grid(COLOR_RED if s == "1P" else COLOR_BLUE)
            for s in sides_list
        ])
        npz_dir = tmp_path / "npz"
        npz_dir.mkdir()
        _write_npz(
            npz_dir / "v29.npz",
            grids=grids,
            video_ids=["v29"] * (n * 2),
            sides=sides_list,
            t_secs=t_secs_list,
            game_idxs=[0] * (n * 2),
            frame_idxs=list(range(n * 2)),
        )
        _write_labeled_win(tmp_path / "labeled_win.csv", [])
        result = mod.build_pairs(npz_dir, tmp_path / "labeled_win.csv")
        assert result.board_1p.shape == (n, BOARD_ROWS, BOARD_COLS)
        assert result.board_2p.shape == (n, BOARD_ROWS, BOARD_COLS)
        assert len(result.won) == n
        assert len(result.t_sec) == n
        assert len(result.game_idx) == n
