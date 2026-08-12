"""scripts/build_labeled_win_from_npz.py のユニットテスト (2026-08-12 選択肢C MVP)。

テスト方針:
- 動画認識は一切しない。合成 npz (boards_lean 形式) を作って変換を検証する。
- 受け入れ基準: center_bulge が出力CSVに乗ること、既存 pair_sides_for_win /
  build_features (scripts/model_indicator_win.py) が無改修で読めること。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.board import BOARD_COLS, BOARD_ROWS, COLOR_RED

import scripts.build_labeled_win_from_npz as blwn


def _make_grid(height: int, color: int = COLOR_RED) -> np.ndarray:
    """全列を height 段まで積んだ合成グリッドを返す ((13,6) int8)。"""
    g = np.zeros((BOARD_ROWS, BOARD_COLS), dtype=np.int8)
    for col in range(BOARD_COLS):
        for row in range(BOARD_ROWS - 1, BOARD_ROWS - 1 - height, -1):
            g[row, col] = color
    return g


def _write_synthetic_npz(path: Path, n: int = 4) -> None:
    """1P/2P 交互の合成 boards_lean npz を書き出す (won 整合済み)。"""
    grids = np.array([_make_grid(height=3 + i) for i in range(n)], dtype=np.int8)
    video_id = np.array(["video_test"] * n)
    side = np.array(["1P" if i % 2 == 0 else "2P" for i in range(n)])
    t_sec = np.array([float(i) * 0.5 for i in range(n)], dtype=np.float32)
    game_idx = np.zeros(n, dtype=np.int32)
    frame_idx = np.arange(n, dtype=np.int32)
    won = np.array([1.0 if i % 2 == 0 else 0.0 for i in range(n)], dtype=np.float32)
    score = np.full(n, -1, dtype=np.int32)
    np.savez_compressed(
        str(path), grids=grids, video_id=video_id, side=side, t_sec=t_sec,
        game_idx=game_idx, frame_idx=frame_idx, won=won, score=score,
    )


def test_convert_one_npz_includes_center_bulge(tmp_path: Path) -> None:
    """出力行に center_bulge / center_bulge_raw が含まれること (受け入れ基準)。"""
    npz_path = tmp_path / "synthetic.npz"
    _write_synthetic_npz(npz_path, n=4)
    registry = blwn._resolve_indicator_registry("light")
    rows = blwn.convert_one_npz(npz_path, registry)
    assert len(rows) == 4
    for row in rows:
        assert "center_bulge" in row
        assert "center_bulge_raw" in row
        assert 0.0 <= row["center_bulge"] <= 1.0


def test_convert_one_npz_flat_board_center_bulge_is_half(tmp_path: Path) -> None:
    """全列同高 (フラット) な合成盤面は center_bulge=0.5 になること。"""
    npz_path = tmp_path / "flat.npz"
    _write_synthetic_npz(npz_path, n=2)
    registry = blwn._resolve_indicator_registry("light")
    rows = blwn.convert_one_npz(npz_path, registry)
    for row in rows:
        assert row["center_bulge"] == pytest.approx(0.5)
        assert row["center_bulge_raw"] == pytest.approx(0.0)


def test_full_profile_includes_heavy_indicators(tmp_path: Path) -> None:
    """--profile full では current_max_chain 等の重い指標も出ること。"""
    npz_path = tmp_path / "synthetic.npz"
    _write_synthetic_npz(npz_path, n=2)
    registry = blwn._resolve_indicator_registry("full")
    rows = blwn.convert_one_npz(npz_path, registry)
    assert "current_max_chain" in rows[0]
    assert "dig_resistance" in rows[0]


def test_convert_dir_writes_csv_with_expected_columns(tmp_path: Path) -> None:
    """convert_dir が CSV を書き出し、メタ列+center_bulge列を含むこと。"""
    npz_dir = tmp_path / "npz"
    npz_dir.mkdir()
    _write_synthetic_npz(npz_dir / "a.npz", n=4)
    out_csv = tmp_path / "out.csv"
    n_rows, _elapsed = blwn.convert_dir(npz_dir, out_csv, profile="light")
    assert n_rows == 4
    assert out_csv.exists()
    import pandas as pd
    df = pd.read_csv(out_csv)
    assert "center_bulge" in df.columns
    assert "won" in df.columns
    assert len(df) == 4


def test_output_compatible_with_pair_sides_for_win(tmp_path: Path) -> None:
    """既存 pair_sides_for_win / build_features が無改修で読めること
    (薄い委譲構造の受け入れ基準: 変換ツールは指標計算を indicators_v2 に
    委譲するだけで、下流の学習パイプラインには一切手を入れない)。
    """
    npz_dir = tmp_path / "npz"
    npz_dir.mkdir()
    _write_synthetic_npz(npz_dir / "a.npz", n=4)
    out_csv = tmp_path / "out.csv"
    blwn.convert_dir(npz_dir, out_csv, profile="light")

    from scripts.model_indicator_win import (
        load_labeled_csv, pair_sides_for_win, build_features, _get_indicator_cols,
    )
    df = load_labeled_csv(str(out_csv))
    paired = pair_sides_for_win(df, max_tdiff=1.0)
    cols = _get_indicator_cols(paired)
    assert "center_bulge" in cols
    feat = build_features(paired, cols)
    assert "center_bulge_diff" in feat.columns


def test_approx_tsumo_is_rank_within_group() -> None:
    """_approx_tsumo は (video_id, side, game_idx) 内で t_sec 順位を振ること。"""
    rows = [
        {"video_id": "v1", "side": "1P", "game_idx": 0, "t_sec": 3.0},
        {"video_id": "v1", "side": "1P", "game_idx": 0, "t_sec": 1.0},
        {"video_id": "v1", "side": "1P", "game_idx": 0, "t_sec": 2.0},
    ]
    blwn._approx_tsumo(rows)
    by_t = {r["t_sec"]: r["tsumo"] for r in rows}
    assert by_t[1.0] == 0
    assert by_t[2.0] == 1
    assert by_t[3.0] == 2
