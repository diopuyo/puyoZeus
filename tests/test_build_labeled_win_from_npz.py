"""scripts/build_labeled_win_from_npz.py のユニットテスト。

テスト方針:
- 動画認識は一切しない。合成 npz (boards_lean 形式) を作って変換を検証する。
- 受け入れ基準 (2026-08-12 選択肢C MVP): center_bulge 分解列が出力CSVに乗る
  こと、既存 pair_sides_for_win / build_features (scripts/model_indicator_
  win.py) が無改修で読めること。
- 指標大整理 (2026-08-12 user確定、docs/INDICATOR_REORG_PROPOSAL_2026-08-12.md
  「決定記録」節) の反映確認: a-1 (*_raw 削除) / b-1 (center_bulge 分解) /
  b-2 (相手との差 diff_ 列、merge_asof 意味論)。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.board import BOARD_COLS, BOARD_ROWS, COLOR_OJAMA, COLOR_RED

import scripts.build_labeled_win_from_npz as blwn


def _make_grid(height: int, color: int = COLOR_RED) -> np.ndarray:
    """全列を height 段まで積んだ合成グリッドを返す ((13,6) int8)。"""
    g = np.zeros((BOARD_ROWS, BOARD_COLS), dtype=np.int8)
    for col in range(BOARD_COLS):
        for row in range(BOARD_ROWS - 1, BOARD_ROWS - 1 - height, -1):
            g[row, col] = color
    return g


def _write_synthetic_npz(
    path: Path, n: int = 4, t_step: float = 0.5, video_id: str = "video_test",
) -> None:
    """1P/2P 交互の合成 boards_lean npz を書き出す (won 整合済み)。

    高さは i ごとに変える (3+i 段) ため、diff_max_column_height 等が
    0 以外の値になり、merge_asof の対応付けをそのまま検証できる。
    """
    grids = np.array([_make_grid(height=3 + i) for i in range(n)], dtype=np.int8)
    video_id_arr = np.array([video_id] * n)
    side = np.array(["1P" if i % 2 == 0 else "2P" for i in range(n)])
    t_sec = np.array([float(i) * t_step for i in range(n)], dtype=np.float32)
    game_idx = np.zeros(n, dtype=np.int32)
    frame_idx = np.arange(n, dtype=np.int32)
    won = np.array([1.0 if i % 2 == 0 else 0.0 for i in range(n)], dtype=np.float32)
    score = np.full(n, -1, dtype=np.int32)
    np.savez_compressed(
        str(path), grids=grids, video_id=video_id_arr, side=side, t_sec=t_sec,
        game_idx=game_idx, frame_idx=frame_idx, won=won, score=score,
    )


# ============================
# a-1: *_raw 列の完全削除
# ============================


def test_convert_one_npz_never_emits_raw_columns(tmp_path: Path) -> None:
    """出力行に *_raw 列が一切含まれないこと (a-1 決定記録)。"""
    npz_path = tmp_path / "synthetic.npz"
    _write_synthetic_npz(npz_path, n=4)
    registry = blwn._resolve_indicator_registry("full")
    rows = blwn.convert_one_npz(npz_path, registry)
    for row in rows:
        assert not any(k.endswith("_raw") for k in row), row.keys()


def test_full_profile_excludes_saturated_chain_count(tmp_path: Path) -> None:
    """full profile のレジストリから saturated_chain_count が消えていること
    (current_max_chain と完全一致していたための a-1 削除)。"""
    registry = blwn._resolve_indicator_registry("full")
    assert "saturated_chain_count" not in registry
    assert "current_max_chain" in registry  # 削除しすぎていないことの確認


def test_csv_output_has_no_raw_columns(tmp_path: Path) -> None:
    """convert_dir が書き出す CSV ヘッダに *_raw 列が無いこと (a-1)。"""
    npz_dir = tmp_path / "npz"
    npz_dir.mkdir()
    _write_synthetic_npz(npz_dir / "a.npz", n=4)
    out_csv = tmp_path / "out.csv"
    blwn.convert_dir(npz_dir, out_csv, profile="light")
    df = pd.read_csv(out_csv)
    assert not any(c.endswith("_raw") for c in df.columns)


# ============================
# b-1: center_bulge の色ぷよ/おじゃま分解
# ============================


def test_convert_one_npz_includes_center_bulge_decomposition(tmp_path: Path) -> None:
    """合成版 center_bulge でなく分解2列 (own+diff) が出ること (b-1)。"""
    npz_path = tmp_path / "synthetic.npz"
    _write_synthetic_npz(npz_path, n=4)
    registry = blwn._resolve_indicator_registry("light")
    rows = blwn.convert_one_npz(npz_path, registry)
    assert len(rows) == 4
    for row in rows:
        assert "center_bulge" not in row  # 合成版はこのツールでは出力しない
        assert "center_bulge_color" in row
        assert "center_bulge_ojama" in row
        assert 0.0 <= row["center_bulge_color"] <= 1.0
        assert 0.0 <= row["center_bulge_ojama"] <= 1.0


def test_convert_one_npz_flat_board_center_bulge_color_is_half(tmp_path: Path) -> None:
    """全列同高 (フラット) な合成盤面は center_bulge_color=0.5 になること
    (own値の検証。合成盤面は全セルが色ぷよのため center_bulge_ojama も0.5)。"""
    npz_path = tmp_path / "flat.npz"
    _write_synthetic_npz(npz_path, n=2)
    registry = blwn._resolve_indicator_registry("light")
    rows = blwn.convert_one_npz(npz_path, registry)
    for row in rows:
        assert row["center_bulge_color"] == pytest.approx(0.5)
        assert row["center_bulge_ojama"] == pytest.approx(0.5)


def test_ojama_board_has_zero_center_bulge_color(tmp_path: Path) -> None:
    """全てお邪魔の盤面では center_bulge_color がフラット (色ぷよが無いため)。

    中央列 (2,3) のみ高さ10のお邪魔タワー (BASE_ROWS=3 段クリップ後も
    center_bulge_ojama が明確に0.5超になる高さが必要)。
    """
    npz_path = tmp_path / "ojama.npz"
    g = np.zeros((BOARD_ROWS, BOARD_COLS), dtype=np.int8)
    tower_height = 10
    for col in (2, 3):
        for row in range(BOARD_ROWS - tower_height, BOARD_ROWS):
            g[row, col] = COLOR_OJAMA
    grids = np.array([g], dtype=np.int8)
    np.savez_compressed(
        str(npz_path), grids=grids, video_id=np.array(["v"]), side=np.array(["1P"]),
        t_sec=np.array([0.0], dtype=np.float32), game_idx=np.zeros(1, dtype=np.int32),
        frame_idx=np.zeros(1, dtype=np.int32), won=np.array([1.0], dtype=np.float32),
        score=np.array([-1], dtype=np.int32),
    )
    registry = blwn._resolve_indicator_registry("light")
    rows = blwn.convert_one_npz(npz_path, registry)
    assert rows[0]["center_bulge_color"] == pytest.approx(0.5)
    assert rows[0]["center_bulge_ojama"] > 0.5


def test_full_profile_includes_heavy_indicators(tmp_path: Path) -> None:
    """--profile full では current_max_chain 等の重い指標も出ること。"""
    npz_path = tmp_path / "synthetic.npz"
    _write_synthetic_npz(npz_path, n=2)
    registry = blwn._resolve_indicator_registry("full")
    rows = blwn.convert_one_npz(npz_path, registry)
    assert "current_max_chain" in rows[0]
    assert "dig_resistance" in rows[0]


def test_convert_dir_writes_csv_with_expected_columns(tmp_path: Path) -> None:
    """convert_dir が CSV を書き出し、メタ列+center_bulge分解列を含むこと。"""
    npz_dir = tmp_path / "npz"
    npz_dir.mkdir()
    _write_synthetic_npz(npz_dir / "a.npz", n=4)
    out_csv = tmp_path / "out.csv"
    n_rows, _elapsed = blwn.convert_dir(npz_dir, out_csv, profile="light")
    assert n_rows == 4
    assert out_csv.exists()
    df = pd.read_csv(out_csv)
    assert "center_bulge_color" in df.columns
    assert "diff_center_bulge_color" in df.columns
    assert "won" in df.columns
    assert len(df) == 4


# ============================
# b-2: 「相手との差」列 (diff_) の merge_asof 意味論
# ============================


def test_diff_columns_replace_own_for_replace_target_columns(tmp_path: Path) -> None:
    """DIFF_REPLACE_OWN_COLUMNS 対象 (max_column_height 等) は own が CSV に
    出ず、diff_ 版のみ出ること (b-2 決定記録)。"""
    npz_dir = tmp_path / "npz"
    npz_dir.mkdir()
    _write_synthetic_npz(npz_dir / "a.npz", n=4)
    out_csv = tmp_path / "out.csv"
    blwn.convert_dir(npz_dir, out_csv, profile="light")
    df = pd.read_csv(out_csv)
    assert "max_column_height" not in df.columns
    assert "diff_max_column_height" in df.columns
    assert "conn_pair_count" not in df.columns
    assert "diff_conn_pair_count" in df.columns


def test_diff_columns_keep_own_for_pair_columns(tmp_path: Path) -> None:
    """DIFF_KEEP_OWN_PAIR_COLUMNS 対象 (色ぷよ総数等) は own と diff の両方が
    残ること (user指示8/12: 色ぷよ総数はおじゃま総数とペアで見る)。"""
    npz_dir = tmp_path / "npz"
    npz_dir.mkdir()
    _write_synthetic_npz(npz_dir / "a.npz", n=4)
    out_csv = tmp_path / "out.csv"
    blwn.convert_dir(npz_dir, out_csv, profile="light")
    df = pd.read_csv(out_csv)
    assert "board_color_puyo_total" in df.columns
    assert "diff_board_color_puyo_total" in df.columns
    assert "board_ojama_count" in df.columns
    assert "diff_board_ojama_count" in df.columns


def test_diff_column_exempt_conn_triple_count_has_no_diff(tmp_path: Path) -> None:
    """conn_triple_count は diff化しない例外 (own のみ)。"""
    npz_dir = tmp_path / "npz"
    npz_dir.mkdir()
    _write_synthetic_npz(npz_dir / "a.npz", n=4)
    out_csv = tmp_path / "out.csv"
    blwn.convert_dir(npz_dir, out_csv, profile="light")
    df = pd.read_csv(out_csv)
    assert "conn_triple_count" in df.columns
    assert "diff_conn_triple_count" not in df.columns


def test_diff_uses_opponent_most_recent_value_not_nearest(tmp_path: Path) -> None:
    """diff_<col> が「相手の直近確定値 (backward)」であり、時間的に後の値を
    使わないこと (merge_asof の意味論そのものの検証、b-2)。

    合成盤面: 1P@t=0.0 (height=3), 2P@t=0.5 (height=4),
              1P@t=1.0 (height=5), 2P@t=1.5 (height=6)。
    1P@t=1.0 の相手直近値は 2P@t=0.5 (height=4) であるべき
    (2P@t=1.5 の height=6 を使ってはいけない = 未来を見ない)。
    """
    npz_path = tmp_path / "synthetic.npz"
    _write_synthetic_npz(npz_path, n=4, t_step=0.5)
    registry = blwn._resolve_indicator_registry("light")
    rows = blwn.convert_one_npz(npz_path, registry)
    row_1p_t1 = next(r for r in rows if r["side"] == "1P" and r["t_sec"] == pytest.approx(1.0))
    # height=5 の1P (t=1.0) 自身の max_column_height は own では 5/12。
    # 相手 (2P@t=0.5, height=4) との差 = (5-4)/12 = 1/12。
    expected_diff = (5.0 / 12.0) - (4.0 / 12.0)
    assert row_1p_t1["diff_max_column_height"] == pytest.approx(expected_diff, abs=1e-6)


def test_diff_is_nan_when_opponent_has_no_prior_snapshot(tmp_path: Path) -> None:
    """相手側の確定値がまだ無い先頭区間は diff が NaN になること (b-2、
    対応が付かない場合はデータを消さずNaNで残す設計)。"""
    npz_path = tmp_path / "synthetic.npz"
    _write_synthetic_npz(npz_path, n=4, t_step=0.5)
    registry = blwn._resolve_indicator_registry("light")
    rows = blwn.convert_one_npz(npz_path, registry)
    row_1p_t0 = next(r for r in rows if r["side"] == "1P" and r["t_sec"] == pytest.approx(0.0))
    assert np.isnan(row_1p_t0["diff_max_column_height"])


def test_pair_interaction_columns_present_and_consistent(tmp_path: Path) -> None:
    """color_ojama_ratio_own / color_diff_x_ojama_diff が出力され、own比率式・
    diff積式と一致すること (b-2 user指示8/12)。"""
    npz_path = tmp_path / "synthetic.npz"
    _write_synthetic_npz(npz_path, n=4, t_step=0.5)
    registry = blwn._resolve_indicator_registry("light")
    rows = blwn.convert_one_npz(npz_path, registry)
    row = next(r for r in rows if r["side"] == "1P" and r["t_sec"] == pytest.approx(1.0))
    color = row["board_color_puyo_total"]
    ojama = row["board_ojama_count"]
    expected_ratio = color / (color + ojama + blwn.COLOR_OJAMA_RATIO_EPS)
    assert row["color_ojama_ratio_own"] == pytest.approx(expected_ratio)
    expected_interaction = (
        row["diff_board_color_puyo_total"] * row["diff_board_ojama_count"]
    )
    assert row["color_diff_x_ojama_diff"] == pytest.approx(expected_interaction)


# ============================
# 全消しボーナス予約中フラグ (all_clear_bonus_pending) — 2026-08-12
# user伝授 (設計訂正版: 瞬間の空盤面でなく「ボーナス未消費」の持続状態)
# ============================


def _write_npz_with_scores(
    path: Path,
    sides: list[str],
    t_secs: list[float],
    scores: list[int],
    chain_mechanisms: list[str] | None = None,
    game_idx: list[int] | None = None,
) -> None:
    """任意の score/chain_mechanism 列を持つ合成 boards_lean npz を書く。

    all_clear_bonus_pending 系のテストは盤面の見た目でなく score の遷移を
    厳密に制御したいため専用ヘルパーを使う (盤面は全行同一の適当な形で良い)。
    """
    n = len(sides)
    grid = _make_grid(height=3)
    grids = np.array([grid] * n, dtype=np.int8)
    if chain_mechanisms is None:
        chain_mechanisms = [""] * n
    if game_idx is None:
        game_idx = [0] * n
    np.savez_compressed(
        str(path), grids=grids, video_id=np.array(["v"] * n), side=np.array(sides),
        t_sec=np.array(t_secs, dtype=np.float32), game_idx=np.array(game_idx, dtype=np.int32),
        frame_idx=np.arange(n, dtype=np.int32), won=np.array([1.0] * n, dtype=np.float32),
        score=np.array(scores, dtype=np.int32),
        chain_mechanism=np.array(chain_mechanisms),
    )


def test_all_clear_bonus_pending_turns_on_for_untagged_large_jump(tmp_path: Path) -> None:
    """連鎖タグ無しの2100点ジャンプ (全消しボーナス相当) で ON になること。

    通常の落下ボーナス上限 (250) を大きく超え、chain_mechanism も空
    (連鎖の外) なジャンプ = 全消しボーナス計上、というuser指定の検出方法。
    """
    npz_path = tmp_path / "on.npz"
    _write_npz_with_scores(
        npz_path, sides=["1P", "1P"], t_secs=[0.0, 1.0],
        scores=[1000, 1000 + blwn.ALL_CLEAR_BONUS_SCORE], chain_mechanisms=["", ""],
    )
    registry = blwn._resolve_indicator_registry("light")
    rows = blwn.convert_one_npz(npz_path, registry)
    rows_sorted = sorted(rows, key=lambda r: r["t_sec"])
    assert rows_sorted[0]["all_clear_bonus_pending"] == pytest.approx(0.0)
    assert rows_sorted[1]["all_clear_bonus_pending"] == pytest.approx(1.0)


def test_all_clear_bonus_pending_turns_off_on_next_chain(tmp_path: Path) -> None:
    """ON 状態で次の連鎖 (タグ付き得点増分) が来ると OFF (ボーナス消費) になること。"""
    npz_path = tmp_path / "off.npz"
    _write_npz_with_scores(
        npz_path, sides=["1P", "1P", "1P"], t_secs=[0.0, 1.0, 2.0],
        scores=[1000, 1000 + blwn.ALL_CLEAR_BONUS_SCORE, 1000 + blwn.ALL_CLEAR_BONUS_SCORE + 300],
        chain_mechanisms=["", "", "formula"],
    )
    registry = blwn._resolve_indicator_registry("light")
    rows = blwn.convert_one_npz(npz_path, registry)
    rows_sorted = sorted(rows, key=lambda r: r["t_sec"])
    assert rows_sorted[1]["all_clear_bonus_pending"] == pytest.approx(1.0)  # ON
    assert rows_sorted[2]["all_clear_bonus_pending"] == pytest.approx(0.0)  # 次の連鎖でOFF


def test_all_clear_bonus_pending_unchanged_for_small_jump(tmp_path: Path) -> None:
    """通常の落下ボーナス相当 (250点以下) のジャンプでは状態が変化しないこと。"""
    npz_path = tmp_path / "nochange.npz"
    _write_npz_with_scores(
        npz_path, sides=["1P", "1P"], t_secs=[0.0, 1.0],
        scores=[1000, 1000 + blwn.MAX_DROP_BONUS_SCORE], chain_mechanisms=["", ""],
    )
    registry = blwn._resolve_indicator_registry("light")
    rows = blwn.convert_one_npz(npz_path, registry)
    rows_sorted = sorted(rows, key=lambda r: r["t_sec"])
    assert rows_sorted[0]["all_clear_bonus_pending"] == pytest.approx(0.0)
    assert rows_sorted[1]["all_clear_bonus_pending"] == pytest.approx(0.0)


def test_opp_all_clear_bonus_pending_carries_opponent_raw_value_not_diff(
    tmp_path: Path,
) -> None:
    """opp_all_clear_bonus_pending は相手の直近own値そのもの (diff ではない)
    であること (b-2 拡張、2026-08-12 user伝授: フラグの差分は無意味)。

    1P@t=0.0 (score=1000, base) → 1P@t=1.0 で+2100ジャンプ (ON) →
    2P@t=1.5 の相手直近値は 1P@t=1.0 (ON=1) のはず。
    """
    npz_path = tmp_path / "carry.npz"
    _write_npz_with_scores(
        npz_path, sides=["1P", "1P", "2P"], t_secs=[0.0, 1.0, 1.5],
        scores=[1000, 1000 + blwn.ALL_CLEAR_BONUS_SCORE, 500],
        chain_mechanisms=["", "", ""],
    )
    registry = blwn._resolve_indicator_registry("light")
    rows = blwn.convert_one_npz(npz_path, registry)
    row_2p = next(r for r in rows if r["side"] == "2P")
    assert row_2p["opp_all_clear_bonus_pending"] == pytest.approx(1.0)
    assert "diff_all_clear_bonus_pending" not in row_2p  # carry対象は diff_ を作らない


def test_all_clear_bonus_pending_is_nan_when_score_unreliable(tmp_path: Path) -> None:
    """score が全行 -1 (OCR破綻、c26/c58/c69相当) だと2行目以降がNaNになること。

    1行目は「新しい試合はボーナス未保持から始まる」という論理的事実のため
    score の可読性に関わらず 0.0 になる (実装上の仕様、docstring参照)。
    """
    npz_path = tmp_path / "broken.npz"
    _write_npz_with_scores(
        npz_path, sides=["1P", "1P", "1P"], t_secs=[0.0, 1.0, 2.0],
        scores=[-1, -1, -1], chain_mechanisms=["", "", ""],
    )
    registry = blwn._resolve_indicator_registry("light")
    rows = blwn.convert_one_npz(npz_path, registry)
    rows_sorted = sorted(rows, key=lambda r: r["t_sec"])
    assert rows_sorted[0]["all_clear_bonus_pending"] == pytest.approx(0.0)
    assert np.isnan(rows_sorted[1]["all_clear_bonus_pending"])
    assert np.isnan(rows_sorted[2]["all_clear_bonus_pending"])


def test_csv_output_includes_all_clear_bonus_pending_columns(tmp_path: Path) -> None:
    """convert_dir の CSV に all_clear_bonus_pending / opp_ 版が乗ること。"""
    npz_dir = tmp_path / "npz"
    npz_dir.mkdir()
    _write_synthetic_npz(npz_dir / "a.npz", n=4)
    out_csv = tmp_path / "out.csv"
    blwn.convert_dir(npz_dir, out_csv, profile="light")
    df = pd.read_csv(out_csv)
    assert "all_clear_bonus_pending" in df.columns
    assert "opp_all_clear_bonus_pending" in df.columns
    assert "diff_all_clear_bonus_pending" not in df.columns


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
    assert "center_bulge_color" in cols
    feat = build_features(paired, cols)
    assert "center_bulge_color_diff" in feat.columns


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
