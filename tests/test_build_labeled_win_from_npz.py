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


# ============================
# Rust ネイティブ拡張 (puyo_core) 載せ替え パリティ (2026-08-13 追加)
# ============================
# full profile 重い4列 (current_max_chain/dig_resistance/ukeyasusa/
# sub_chain_count) の native 分岐が既存 Python 実装と完全一致することを
# 実盤面 (data/indicators_v2/boards_lean_phase_l_2026-08-11/*.npz) 1,000件超
# で検証する。tests/test_puyo_core_parity.py と同じ「実データ+skip設計」を
# 踏襲する (拡張未導入・データ不足環境では skip、フォールバック同士の
# 自明一致を「パリティ確認」と誤認しないため)。

from src.board import Board  # noqa: E402
from src.board import COLOR_UNKNOWN as _COLOR_UNKNOWN  # noqa: E402
from src.chain import ChainSimulator as _ChainSimulator  # noqa: E402

_NATIVE_PARITY_DATA_DIR = (
    Path(__file__).resolve().parent.parent
    / "data" / "indicators_v2" / "boards_lean_phase_l_2026-08-11"
)
_NATIVE_PARITY_TARGET_BOARDS: int = 1000
_NATIVE_PARITY_RNG_SEED: int = 20260813

pytestmark_native_parity = pytest.mark.skipif(
    not blwn._PUYO_CORE_AVAILABLE,
    reason="puyo_core ネイティブ拡張が未ビルド (maturin develop 要)",
)


def _load_native_parity_sample_boards() -> "list":
    """複数 npz から実盤面をサンプルして Board リストを返す (UNKNOWN含む盤面は除外、
    tests/test_puyo_core_parity.py::_load_sample_boards と同一方針)。
    """
    npz_files = sorted(_NATIVE_PARITY_DATA_DIR.glob("*.npz"))
    if not npz_files:
        pytest.skip(f"評価データが見つからない: {_NATIVE_PARITY_DATA_DIR}")
    rng = np.random.RandomState(_NATIVE_PARITY_RNG_SEED)
    order = rng.permutation(len(npz_files))
    boards: "list" = []
    per_file = max(1, _NATIVE_PARITY_TARGET_BOARDS // 10)
    for idx in order:
        data = np.load(str(npz_files[idx]), allow_pickle=True)
        grids = data["grids"]
        n = grids.shape[0]
        if n == 0:
            continue
        picked = rng.choice(n, size=min(per_file, n), replace=False)
        for i in picked:
            grid = grids[i].astype(np.uint8)
            if np.any(grid == _COLOR_UNKNOWN):
                continue
            board = Board()
            board._grid = grid
            boards.append(board)
        if len(boards) >= _NATIVE_PARITY_TARGET_BOARDS:
            break
    return boards


@pytest.fixture(scope="module")
def native_parity_boards() -> "list":
    boards = _load_native_parity_sample_boards()
    if len(boards) < 1000:
        pytest.skip(f"実盤面サンプルが1000未満 ({len(boards)}件)。データ不足のためスキップ")
    return boards


@pytest.fixture()
def deterministic_drop_ojama(monkeypatch: pytest.MonkeyPatch) -> None:
    """`ChainSimulator.drop_ojama` の端数列選択をテスト内で決定的にする。

    本番コード (dig_resistance/ukeyasusa) は `drop_ojama(board, n_ojama)` を
    seed無しで呼ぶため、おじゃま個数が6の倍数でない限り毎回OS乱数由来の
    非決定的な結果になる (既存の既知の性質、この載せ替えでは変えない)。
    Python版/native版のどちらも `ChainSimulator.drop_ojama` を通るため、
    同一 n_ojama に対して決定的な seed を強制すれば両者は同一の
    おじゃま落下盤面を得る (パリティ検証のための一時パッチ、本番動作は
    無変更)。
    """
    original = _ChainSimulator.drop_ojama

    def _patched(self: "_ChainSimulator", board: Board, ojama_count: int, seed=None):
        del seed  # テスト用に無視し、ojama_countのみに依存する決定的seedへ差替え
        return original(self, board, ojama_count, seed=1_000_000 + ojama_count)

    monkeypatch.setattr(_ChainSimulator, "drop_ojama", _patched)


@pytestmark_native_parity
class TestNativeHeavyIndicatorParity:
    """full profile 重い4列 native 分岐 vs 既存 Python 実装の完全一致確認。"""

    def test_current_max_chain_matches(self, native_parity_boards: "list") -> None:
        """current_max_chain: 乱数を含まないため無条件で完全一致するはず。"""
        mismatches = []
        for i, board in enumerate(native_parity_boards):
            py_val = blwn.GRID_ONLY_HEAVY_INDICATORS["current_max_chain"](board)
            native_val = blwn.GRID_ONLY_HEAVY_INDICATORS_NATIVE["current_max_chain"](
                board, use_native=True,
            )
            if py_val.score != native_val.score or py_val.raw != native_val.raw:
                mismatches.append((i, py_val, native_val))
        assert not mismatches, (
            f"{len(mismatches)}/{len(native_parity_boards)} 件不一致 (先頭5件): "
            f"{mismatches[:5]}"
        )

    def test_sub_chain_count_matches(self, native_parity_boards: "list") -> None:
        """sub_chain_count: 乱数を含まないため無条件で完全一致するはず。"""
        mismatches = []
        for i, board in enumerate(native_parity_boards):
            py_val = blwn.GRID_ONLY_HEAVY_INDICATORS["sub_chain_count"](board)
            native_val = blwn.GRID_ONLY_HEAVY_INDICATORS_NATIVE["sub_chain_count"](
                board, use_native=True,
            )
            if py_val.score != native_val.score or py_val.raw != native_val.raw:
                mismatches.append((i, py_val, native_val))
        assert not mismatches, (
            f"{len(mismatches)}/{len(native_parity_boards)} 件不一致 (先頭5件): "
            f"{mismatches[:5]}"
        )

    def test_dig_resistance_matches_with_fixed_ojama_seed(
        self, native_parity_boards: "list", deterministic_drop_ojama: None,
    ) -> None:
        """dig_resistance: おじゃま落下の乱数を固定した上で完全一致するはず。"""
        mismatches = []
        for i, board in enumerate(native_parity_boards):
            py_val = blwn.GRID_ONLY_HEAVY_INDICATORS["dig_resistance"](board)
            native_val = blwn.GRID_ONLY_HEAVY_INDICATORS_NATIVE["dig_resistance"](
                board, use_native=True,
            )
            if py_val.score != native_val.score:
                mismatches.append((i, py_val.score, native_val.score))
        assert not mismatches, (
            f"{len(mismatches)}/{len(native_parity_boards)} 件不一致 (先頭5件): "
            f"{mismatches[:5]}"
        )

    def test_ukeyasusa_matches_with_fixed_ojama_seed(
        self, native_parity_boards: "list", deterministic_drop_ojama: None,
    ) -> None:
        """ukeyasusa: 内部で dig_resistance を使うため同様に乱数固定で検証。"""
        mismatches = []
        for i, board in enumerate(native_parity_boards):
            py_val = blwn.GRID_ONLY_HEAVY_INDICATORS["ukeyasusa"](board)
            native_val = blwn.GRID_ONLY_HEAVY_INDICATORS_NATIVE["ukeyasusa"](
                board, use_native=True,
            )
            if py_val.score != native_val.score:
                mismatches.append((i, py_val.score, native_val.score))
        assert not mismatches, (
            f"{len(mismatches)}/{len(native_parity_boards)} 件不一致 (先頭5件): "
            f"{mismatches[:5]}"
        )

    def test_use_native_false_delegates_to_python(
        self, native_parity_boards: "list",
    ) -> None:
        """use_native=False で native 分岐が完全無効化されること
        (current_max_chain のみ代表確認、乱数なしで安全に検証可能)。
        """
        board = native_parity_boards[0]
        py_val = blwn.GRID_ONLY_HEAVY_INDICATORS["current_max_chain"](board)
        native_off = blwn.GRID_ONLY_HEAVY_INDICATORS_NATIVE["current_max_chain"](
            board, use_native=False,
        )
        assert py_val.score == native_off.score
        assert py_val.raw == native_off.raw

    def test_resolve_indicator_registry_use_native_false_matches_python(
        self, native_parity_boards: "list",
    ) -> None:
        """`_resolve_indicator_registry("full", use_native=False)` が
        既存 Python 実装のみのレジストリと同値を返すこと (統合経路確認)。
        """
        board = native_parity_boards[0]
        registry_off = blwn._resolve_indicator_registry("full", use_native=False)
        assert (
            registry_off["current_max_chain"](board).score
            == blwn.GRID_ONLY_HEAVY_INDICATORS["current_max_chain"](board).score
        )

    @pytest.mark.parametrize(
        "name",
        [
            "immediate_fire_power", "chain_efficiency", "second_chain_potential",
            "ignition_point_count", "multi_color_ignition",
        ],
    )
    def test_a1_native_reconnected_indicators_match_python(
        self, native_parity_boards: "list", name: str,
    ) -> None:
        """A-1 (2026-08-13) で native化した5列が既存 Python 実装と完全一致すること。

        乱数を含まない (drop_ojama を使わない) ため無条件で完全一致するはず。
        """
        mismatches = []
        for i, board in enumerate(native_parity_boards):
            py_val = blwn.GRID_ONLY_HEAVY_INDICATORS[name](board)
            native_val = blwn.GRID_ONLY_HEAVY_INDICATORS_NATIVE[name](
                board, use_native=True,
            )
            if py_val.score != native_val.score or py_val.raw != native_val.raw:
                mismatches.append((i, py_val, native_val))
        assert not mismatches, (
            f"{name}: {len(mismatches)}/{len(native_parity_boards)} 件不一致 "
            f"(先頭5件): {mismatches[:5]}"
        )

    # min_puyos_to_ignite / saturation_chain は Python版が実測 2000ms/行・
    # 12500ms/行と極めて重い (緊急native化の動機そのもの) ため、1000件
    # フルサンプルで比較すると Python側だけで数十分〜数時間かかりテスト
    # として非現実的。少数サンプル (先頭8件) に限定して確認する。
    _EXPENSIVE_PARITY_SAMPLE_SIZE: int = 4

    def test_min_puyos_to_ignite_native_matches_python(
        self, native_parity_boards: "list",
    ) -> None:
        """min_puyos_to_ignite: 緊急native化 (実測2000ms/行) の完全一致確認
        (小サンプル、乱数を含まないため無条件で完全一致するはず)。"""
        sample = native_parity_boards[: self._EXPENSIVE_PARITY_SAMPLE_SIZE]
        mismatches = []
        for i, board in enumerate(sample):
            py_val = blwn.GRID_ONLY_HEAVY_INDICATORS["min_puyos_to_ignite"](board)
            native_val = blwn.GRID_ONLY_HEAVY_INDICATORS_NATIVE["min_puyos_to_ignite"](
                board, use_native=True,
            )
            if py_val.score != native_val.score or py_val.raw != native_val.raw:
                mismatches.append((i, py_val, native_val))
        assert not mismatches, (
            f"min_puyos_to_ignite: {len(mismatches)}/{len(sample)} 件不一致: {mismatches}"
        )

    def test_saturation_chain_native_matches_python(
        self, native_parity_boards: "list",
    ) -> None:
        """saturation_chain: 緊急native化 (実測12500ms/行) の完全一致確認
        (小サンプル、乱数を含まないため無条件で完全一致するはず)。"""
        sample = native_parity_boards[: self._EXPENSIVE_PARITY_SAMPLE_SIZE]
        mismatches = []
        for i, board in enumerate(sample):
            py_val = blwn.GRID_ONLY_HEAVY_INDICATORS["saturation_chain"](board)
            native_val = blwn.GRID_ONLY_HEAVY_INDICATORS_NATIVE["saturation_chain"](
                board, use_native=True,
            )
            if py_val.score != native_val.score or py_val.raw != native_val.raw:
                mismatches.append((i, py_val, native_val))
        assert not mismatches, (
            f"saturation_chain: {len(mismatches)}/{len(sample)} 件不一致: {mismatches}"
        )

    # simultaneous_pop_richness の Python版は実測≈149ms/行 (min_puyos_to_
    # ignite/saturation_chain ほど極端ではないが軽くない) のため、1000件
    # フルサンプルではなく中サンプルで比較する (タスク#10仕上げ、2026-08-13)。
    _SIMULTANEOUS_POP_PARITY_SAMPLE_SIZE: int = 60

    def test_simultaneous_pop_richness_native_matches_python(
        self, native_parity_boards: "list",
    ) -> None:
        """simultaneous_pop_richness: タスク#10「移植1」
        (`native/puyo_core::simulate_chain_with_steps`) で native化した
        native分岐が既存 Python 実装と完全一致すること (乱数を含まないため
        無条件で完全一致するはず)。"""
        sample = native_parity_boards[: self._SIMULTANEOUS_POP_PARITY_SAMPLE_SIZE]
        mismatches = []
        for i, board in enumerate(sample):
            py_val = blwn.GRID_ONLY_HEAVY_INDICATORS["simultaneous_pop_richness"](board)
            native_val = blwn.GRID_ONLY_HEAVY_INDICATORS_NATIVE["simultaneous_pop_richness"](
                board, use_native=True,
            )
            if py_val.score != native_val.score or py_val.raw != native_val.raw:
                mismatches.append((i, py_val, native_val))
        assert not mismatches, (
            f"simultaneous_pop_richness: {len(mismatches)}/{len(sample)} 件不一致 "
            f"(先頭5件): {mismatches[:5]}"
        )

    def test_simultaneous_pop_richness_use_native_false_delegates_to_python(
        self, native_parity_boards: "list",
    ) -> None:
        """use_native=False で native 分岐が完全無効化されること。"""
        board = native_parity_boards[0]
        py_val = blwn.GRID_ONLY_HEAVY_INDICATORS["simultaneous_pop_richness"](board)
        native_off = blwn.GRID_ONLY_HEAVY_INDICATORS_NATIVE["simultaneous_pop_richness"](
            board, use_native=False,
        )
        assert py_val.score == native_off.score
        assert py_val.raw == native_off.raw


# ============================
# A-1: 脱落11列の再接続 (2026-08-13 ラウンド2提案書、next非依存の10列)
# ============================


def test_full_profile_includes_a1_reconnected_columns(tmp_path: Path) -> None:
    """A-1 再接続10列が full profile のレジストリに存在すること。"""
    registry = blwn._resolve_indicator_registry("full")
    for name in (
        "immediate_fire_power", "chain_efficiency", "min_puyos_to_ignite",
        "second_chain_potential", "main_linked_pair_count", "isolated_pair_count",
        "main_linked_ratio", "ignition_point_count", "multi_color_ignition",
        "simultaneous_pop_richness",
    ):
        assert name in registry, f"{name} が full profile レジストリに無い"


def test_light_profile_excludes_a1_reconnected_columns() -> None:
    """A-1 再接続10列は light profile には含めない (重い連鎖シミュ系のため)。"""
    registry = blwn._resolve_indicator_registry("light")
    for name in ("immediate_fire_power", "min_puyos_to_ignite", "ignition_point_count"):
        assert name not in registry


def test_a1_reconnected_columns_produce_values_in_unit_range(tmp_path: Path) -> None:
    """A-1 再接続10列が実際に計算され 0〜1 の score を返すこと (NaN/欠損でない)。"""
    npz_path = tmp_path / "synthetic.npz"
    _write_synthetic_npz(npz_path, n=3)
    registry = blwn._resolve_indicator_registry("full")
    rows = blwn.convert_one_npz(npz_path, registry)
    target_cols = (
        "immediate_fire_power", "chain_efficiency", "min_puyos_to_ignite",
        "second_chain_potential", "main_linked_pair_count", "isolated_pair_count",
        "main_linked_ratio", "ignition_point_count", "multi_color_ignition",
        "simultaneous_pop_richness",
    )
    for row in rows:
        for col in target_cols:
            assert col in row, col
            assert not np.isnan(row[col]), col
            assert 0.0 <= row[col] <= 1.0, (col, row[col])


def test_reach_fire_power_not_reconnected(tmp_path: Path) -> None:
    """reach_fire_power (next_pair必須) は grid-only レジストリの型と非互換の
    ため対象外のままであること (production_config.KNOWN_PIPELINE_GAPS に
    別エージェントが既に文書化済み、本ツールでの対応は不要)。"""
    registry = blwn._resolve_indicator_registry("full")
    assert "reach_fire_power" not in registry


# ============================
# A-2: 壊れ動画の隔離 (2026-08-13)
# ============================


def test_broken_videos_constant_matches_expected_four() -> None:
    """BROKEN_VIDEOS が仕様通りの4本 (c26/c30/c58/c69) であること。"""
    assert blwn.BROKEN_VIDEOS == ("c26", "c30", "c58", "c69")


def test_convert_dir_excludes_broken_videos_by_default(
    tmp_path: Path, capsys: pytest.CaptureFixture,
) -> None:
    """既定 (exclude_broken=True) で BROKEN_VIDEOS の npz が変換対象から
    除外され、隔離した動画・行数がログ出力されること (黙って落とさない)。"""
    npz_dir = tmp_path / "npz"
    npz_dir.mkdir()
    _write_synthetic_npz(npz_dir / "c26.npz", n=4, video_id="video_c26")
    _write_synthetic_npz(npz_dir / "good.npz", n=3, video_id="video_good")
    out_csv = tmp_path / "out.csv"
    n_rows, _elapsed = blwn.convert_dir(npz_dir, out_csv, profile="light")
    assert n_rows == 3  # good.npz のみ (c26.npz の4行は隔離)
    df = pd.read_csv(out_csv)
    assert set(df["video_id"]) == {"video_good"}
    captured = capsys.readouterr()
    assert "exclude-broken" in captured.out
    assert "c26.npz" in captured.out
    assert "4" in captured.out  # 隔離した行数


def test_convert_dir_no_exclude_broken_keeps_all(tmp_path: Path) -> None:
    """--no-exclude-broken 相当 (exclude_broken=False) では全npzが変換されること。"""
    npz_dir = tmp_path / "npz"
    npz_dir.mkdir()
    _write_synthetic_npz(npz_dir / "c26.npz", n=4, video_id="video_c26")
    _write_synthetic_npz(npz_dir / "good.npz", n=3, video_id="video_good")
    out_csv = tmp_path / "out.csv"
    n_rows, _elapsed = blwn.convert_dir(
        npz_dir, out_csv, profile="light", exclude_broken=False,
    )
    assert n_rows == 7
    df = pd.read_csv(out_csv)
    assert set(df["video_id"]) == {"video_c26", "video_good"}


# ============================
# A-4: 全消しボーナスの真値差し替え (2026-08-13)
# ============================


def _write_npz_with_all_clear_truth(
    path: Path, n: int, all_clear_values: list[int],
) -> None:
    """all_clear_pending 真値列 (VideoChainTracker由来) を持つ合成npzを書く。"""
    grids = np.array([_make_grid(height=3) for _ in range(n)], dtype=np.int8)
    np.savez_compressed(
        str(path), grids=grids, video_id=np.array(["v"] * n), side=np.array(["1P"] * n),
        t_sec=np.arange(n, dtype=np.float32), game_idx=np.zeros(n, dtype=np.int32),
        frame_idx=np.arange(n, dtype=np.int32), won=np.array([1.0] * n, dtype=np.float32),
        score=np.full(n, -1, dtype=np.int32),
        all_clear_pending=np.array(all_clear_values, dtype=np.int8),
    )


def test_all_clear_truth_used_when_present_in_npz(tmp_path: Path) -> None:
    """npz に all_clear_pending 真値列があればそれをそのまま採用し、
    all_clear_source が真値 (0.0) になること (A-4)。"""
    npz_path = tmp_path / "truth.npz"
    _write_npz_with_all_clear_truth(npz_path, n=4, all_clear_values=[0, 1, 1, 0])
    registry = blwn._resolve_indicator_registry("light")
    rows = blwn.convert_one_npz(npz_path, registry)
    rows_sorted = sorted(rows, key=lambda r: r["t_sec"])
    assert [r["all_clear_bonus_pending"] for r in rows_sorted] == [0.0, 1.0, 1.0, 0.0]
    assert all(r["all_clear_source"] == blwn.ALL_CLEAR_SOURCE_TRUTH for r in rows_sorted)


def test_all_clear_approx_fallback_sets_source_flag(tmp_path: Path) -> None:
    """npz に all_clear_pending が無い旧npzでは近似ヒューリスティックに
    フォールバックし、all_clear_source が近似 (1.0) になること (A-4)。"""
    npz_path = tmp_path / "no_truth.npz"
    _write_synthetic_npz(npz_path, n=3)
    registry = blwn._resolve_indicator_registry("light")
    rows = blwn.convert_one_npz(npz_path, registry)
    assert all(r["all_clear_source"] == blwn.ALL_CLEAR_SOURCE_APPROX for r in rows)


def test_csv_output_includes_all_clear_source_column(tmp_path: Path) -> None:
    """convert_dir の CSV に all_clear_source 列が乗ること。"""
    npz_dir = tmp_path / "npz"
    npz_dir.mkdir()
    _write_synthetic_npz(npz_dir / "a.npz", n=3)
    out_csv = tmp_path / "out.csv"
    blwn.convert_dir(npz_dir, out_csv, profile="light")
    df = pd.read_csv(out_csv)
    assert "all_clear_source" in df.columns


# ============================
# C-3: 飽和連鎖量 (saturation_chain) — opt-in化 (2026-08-13 コスト実測後、user方針決定)
# ============================
# 実測で1行8〜18秒という桁違いのコストが判明したため、既定 full profile
# からは除外し `--with-saturation-chain` の明示指定時のみ含める。
# GRID_ONLY_HEAVY_INDICATORS (カタログ) 自体には残るため、DIFF分類テスト
# (tests/test_indicator_pipeline_registry_2026-08-13.py) やネイティブ
# パリティテスト (上記 TestNativeHeavyIndicatorParity) には影響しない。


def test_full_profile_excludes_saturation_chain_by_default() -> None:
    """saturation_chain は実測コスト (1行8〜18秒) が桁違いのため、既定
    (with_saturation_chain 未指定) の full profile レジストリには
    含まれないこと (opt-in化、2026-08-13 user方針決定)。"""
    registry = blwn._resolve_indicator_registry("full")
    assert "saturation_chain" not in registry


def test_full_profile_includes_saturation_chain_when_opted_in() -> None:
    """--with-saturation-chain 相当 (with_saturation_chain=True) を明示した
    場合のみ saturation_chain が full profile レジストリに含まれること。"""
    registry = blwn._resolve_indicator_registry("full", with_saturation_chain=True)
    assert "saturation_chain" in registry


def test_light_profile_excludes_saturation_chain() -> None:
    """saturation_chain は重い (ビームサーチ+takapt探索) ため light には
    with_saturation_chain の値にかかわらず含めない (light は最初から
    heavy 系を除外する分岐のため)。"""
    registry = blwn._resolve_indicator_registry("light", with_saturation_chain=True)
    assert "saturation_chain" not in registry


def test_optional_heavy_indicator_names_matches_saturation_chain_only() -> None:
    """OPTIONAL_HEAVY_INDICATOR_NAMES が現状 saturation_chain のみである
    こと (想定外の列がサイレントに opt-in 対象へ紛れ込むことを防ぐ)。"""
    assert blwn.OPTIONAL_HEAVY_INDICATOR_NAMES == frozenset({"saturation_chain"})


def test_convert_dir_excludes_saturation_chain_column_by_default(
    tmp_path: Path,
) -> None:
    """convert_dir が既定 (with_saturation_chain 未指定) では CSV に
    saturation_chain 列を出力しないこと。"""
    npz_dir = tmp_path / "npz"
    npz_dir.mkdir()
    _write_synthetic_npz(npz_dir / "a.npz", n=2)
    out_csv = tmp_path / "out.csv"
    blwn.convert_dir(npz_dir, out_csv, profile="full")
    df = pd.read_csv(out_csv)
    assert "saturation_chain" not in df.columns


def test_convert_dir_includes_saturation_chain_column_when_opted_in(
    tmp_path: Path,
) -> None:
    """convert_dir に with_saturation_chain=True (--with-saturation-chain
    相当) を渡すと CSV に saturation_chain 列が出力され、値が 0〜1 に
    収まること。"""
    npz_dir = tmp_path / "npz"
    npz_dir.mkdir()
    _write_synthetic_npz(npz_dir / "a.npz", n=2)
    out_csv = tmp_path / "out.csv"
    blwn.convert_dir(npz_dir, out_csv, profile="full", with_saturation_chain=True)
    df = pd.read_csv(out_csv)
    assert "saturation_chain" in df.columns
    assert df["saturation_chain"].between(0.0, 1.0).all()


# ============================
# saturation_chain_upper: 上部限定軽量版 (2026-08-13 user簡略化決定)
# ============================
# saturation_chain (opt-in) とは異なり、閾値未満は count_puyos() 1回だけの
# 軽量ゲート判定 (NaN即返し) のため opt-in化は不要 = full profile の既定に
# 含める (OPTIONAL_HEAVY_INDICATOR_NAMES には追加していない、
# `test_optional_heavy_indicator_names_matches_saturation_chain_only` が
# 引き続き saturation_chain のみであることを保証する)。


def test_full_profile_includes_saturation_chain_upper_by_default() -> None:
    """saturation_chain_upper はフラグ無しで full profile の既定に含まれる
    こと (saturation_chain の opt-in化とは異なる設計、モジュール docstring
    「saturation_chain_upper」節参照)。"""
    registry = blwn._resolve_indicator_registry("full")
    assert "saturation_chain_upper" in registry


def test_light_profile_excludes_saturation_chain_upper() -> None:
    """saturation_chain_upper は重い (ビームサーチ+takapt探索) ため light
    には含めない (heavy 系除外の既存分岐に従う)。"""
    registry = blwn._resolve_indicator_registry("light")
    assert "saturation_chain_upper" not in registry


def _write_mixed_fill_npz(path: Path) -> None:
    """充填率が閾値未満/以上の盤面を1本ずつ含む合成 npz を書き出す。

    height=3 (fill=18/78≈0.23、閾値未満) と height=12 (fill=72/78≈0.923、
    閾値以上) を1P/2Pに割り当てる (`_write_synthetic_npz` と同じ won 整合
    パターン、`_make_grid` を再利用)。
    """
    grids = np.array([_make_grid(height=3), _make_grid(height=12)], dtype=np.int8)
    video_id_arr = np.array(["video_test"] * 2)
    side = np.array(["1P", "2P"])
    t_sec = np.array([0.0, 0.5], dtype=np.float32)
    game_idx = np.zeros(2, dtype=np.int32)
    frame_idx = np.arange(2, dtype=np.int32)
    won = np.array([1.0, 0.0], dtype=np.float32)
    score = np.full(2, -1, dtype=np.int32)
    np.savez_compressed(
        str(path), grids=grids, video_id=video_id_arr, side=side, t_sec=t_sec,
        game_idx=game_idx, frame_idx=frame_idx, won=won, score=score,
    )


def test_convert_dir_includes_saturation_chain_upper_column_by_default(
    tmp_path: Path,
) -> None:
    """convert_dir が既定 (フラグ無し) で saturation_chain_upper 列を出力し、
    閾値未満は NaN・閾値以上は 0〜1 の有限値になること。"""
    npz_dir = tmp_path / "npz"
    npz_dir.mkdir()
    _write_mixed_fill_npz(npz_dir / "a.npz")
    out_csv = tmp_path / "out.csv"
    blwn.convert_dir(npz_dir, out_csv, profile="full")
    df = pd.read_csv(out_csv)
    assert "saturation_chain_upper" in df.columns
    # board_puyo_total (own, DIFF_KEEP_OWN_PAIR_COLUMNS のため置換されず残る)
    # で低充填/高充填の行を判別する。
    low_fill_row = df[df["board_puyo_total"] < 0.5]
    high_fill_row = df[df["board_puyo_total"] >= 0.5]
    assert len(low_fill_row) == 1 and len(high_fill_row) == 1
    assert low_fill_row["saturation_chain_upper"].isna().all()
    assert high_fill_row["saturation_chain_upper"].between(0.0, 1.0).all()


_SAT_UPPER_PARITY_TARGET_BOARDS: int = 40


def _load_saturation_chain_upper_parity_boards() -> "list":
    """充填率 >= SATURATION_UPPER_MIN_FILL の実盤面のみを収集する専用サンプラー。

    `native_parity_boards` (ランダム抽出) だと対象 (実測 0.68% 出現率、
    `src/indicators_v2.py::SATURATION_UPPER_MIN_FILL` 直上コメント参照) が
    ほぼ集まらないため、閾値を満たす盤面だけを走査して集める専用実装。
    """
    npz_files = sorted(_NATIVE_PARITY_DATA_DIR.glob("*.npz"))
    if not npz_files:
        pytest.skip(f"評価データが見つからない: {_NATIVE_PARITY_DATA_DIR}")
    boards: "list" = []
    for p in npz_files:
        data = np.load(str(p), allow_pickle=True)
        grids = data["grids"]
        for i in range(grids.shape[0]):
            grid = grids[i].astype(np.uint8)
            if np.any(grid == _COLOR_UNKNOWN):
                continue
            board = Board()
            board._grid = grid
            if board.count_puyos() / blwn.iv.FULL_BOARD_CAP >= blwn.iv.SATURATION_UPPER_MIN_FILL:
                boards.append(board)
        if len(boards) >= _SAT_UPPER_PARITY_TARGET_BOARDS:
            break
    return boards


@pytest.fixture(scope="module")
def saturation_chain_upper_parity_boards() -> "list":
    boards = _load_saturation_chain_upper_parity_boards()
    if len(boards) < 5:
        pytest.skip(f"閾値以上の実盤面サンプルが不足 ({len(boards)}件)")
    return boards


@pytestmark_native_parity
def test_saturation_chain_upper_native_matches_python(
    saturation_chain_upper_parity_boards: "list",
) -> None:
    """saturation_chain_upper の native 分岐 (終端測定のみ native化) が
    既存 Python 実装と完全一致すること (閾値以上の実盤面のみ)。"""
    mismatches = []
    for i, board in enumerate(saturation_chain_upper_parity_boards):
        py_val = blwn.GRID_ONLY_HEAVY_INDICATORS["saturation_chain_upper"](board)
        native_val = blwn.GRID_ONLY_HEAVY_INDICATORS_NATIVE["saturation_chain_upper"](
            board, use_native=True,
        )
        if py_val.score != native_val.score or py_val.raw != native_val.raw:
            mismatches.append((i, py_val, native_val))
    assert not mismatches, (
        f"saturation_chain_upper: {len(mismatches)}/"
        f"{len(saturation_chain_upper_parity_boards)} 件不一致: {mismatches}"
    )


def test_saturation_chain_upper_below_threshold_skips_native_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """閾値未満は native 呼び出し自体を行わずに NaN を返すこと (無駄な
    Rust ラウンドトリップ回避の確認、`_native_saturation_chain` を呼んだら
    失敗させて検知する)。"""
    def _fail(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("閾値未満で _native_saturation_chain が呼ばれた")

    monkeypatch.setattr(blwn, "_native_saturation_chain", _fail)
    board = Board()  # 空盤面 (fill=0 < 閾値)
    v = blwn._native_saturation_chain_upper(board)
    assert v.score != v.score  # NaN


# ============================
# C-4: 盤面直読みの新指標3種 (2026-08-13)
# ============================


def test_light_profile_includes_c4_cheap_indicators(tmp_path: Path) -> None:
    """color_diversity_evenness / buried_hole_count が light profile に
    含まれ、実際の値が 0〜1 に収まること (C-4)。"""
    npz_path = tmp_path / "synthetic.npz"
    _write_synthetic_npz(npz_path, n=3)
    registry = blwn._resolve_indicator_registry("light")
    assert "color_diversity_evenness" in registry
    assert "buried_hole_count" in registry
    rows = blwn.convert_one_npz(npz_path, registry)
    for row in rows:
        assert 0.0 <= row["color_diversity_evenness"] <= 1.0
        assert 0.0 <= row["buried_hole_count"] <= 1.0


def test_full_profile_includes_chain_articulation_point_count(tmp_path: Path) -> None:
    """chain_articulation_point_count (重いため full profile 限定) が
    含まれ、実際の値が 0〜1 に収まること (C-4)。"""
    npz_path = tmp_path / "synthetic.npz"
    _write_synthetic_npz(npz_path, n=2)
    registry = blwn._resolve_indicator_registry("full")
    assert "chain_articulation_point_count" in registry
    rows = blwn.convert_one_npz(npz_path, registry)
    for row in rows:
        assert 0.0 <= row["chain_articulation_point_count"] <= 1.0


def test_light_profile_excludes_chain_articulation_point_count() -> None:
    """chain_articulation_point_count は light profile には含めない。"""
    registry = blwn._resolve_indicator_registry("light")
    assert "chain_articulation_point_count" not in registry


# ============================
# タスク#8: おじゃま収支のCSV統合 (2026-08-13、docs/CROSS_CUTTING_AUDIT_
# 2026-08-13.md P4 決着の反映)
# ============================


def _write_npz_with_ojama_truth(
    path: Path,
    sides: list[str],
    t_secs: list[float],
    net_balance: list[float],
    forecast: list[float],
    heights: list[int] | None = None,
    game_idx: list[int] | None = None,
) -> None:
    """ojama_net_balance/ojama_forecast 真値列付きの合成npzを書く。

    all_clear系テストの `_write_npz_with_scores` と同じ流儀 (盤面高さ以外を
    厳密に制御する専用ヘルパー)。heights 未指定時は全行 height=3 の同一盤面
    (猶予量計算をしないテスト向け)。
    """
    n = len(sides)
    if heights is None:
        heights = [3] * n
    grids = np.array([_make_grid(height=h) for h in heights], dtype=np.int8)
    if game_idx is None:
        game_idx = [0] * n
    np.savez_compressed(
        str(path), grids=grids, video_id=np.array(["v"] * n), side=np.array(sides),
        t_sec=np.array(t_secs, dtype=np.float32), game_idx=np.array(game_idx, dtype=np.int32),
        frame_idx=np.arange(n, dtype=np.int32), won=np.array([1.0] * n, dtype=np.float32),
        score=np.full(n, -1, dtype=np.int32),
        ojama_net_balance=np.array(net_balance, dtype=np.float32),
        ojama_forecast=np.array(forecast, dtype=np.float32),
    )


def test_ojama_truth_columns_present_and_normalized(tmp_path: Path) -> None:
    """npz に真値列があれば own-perspective の値を 0-1 正規化して出力すること。"""
    npz_path = tmp_path / "truth.npz"
    _write_npz_with_ojama_truth(
        npz_path, sides=["1P", "1P"], t_secs=[0.0, 1.0],
        net_balance=[0.0, 36.0], forecast=[0.0, 10.0],
    )
    registry = blwn._resolve_indicator_registry("light")
    rows = blwn.convert_one_npz(npz_path, registry)
    rows_sorted = sorted(rows, key=lambda r: r["t_sec"])
    assert rows_sorted[0]["ojama_net_balance"] == pytest.approx(0.5)
    assert rows_sorted[1]["ojama_net_balance"] == pytest.approx(0.75)
    assert rows_sorted[0]["ojama_forecast"] == pytest.approx(0.0)
    assert rows_sorted[1]["ojama_forecast"] == pytest.approx(10.0 / 72.0)
    assert all(r["ojama_source"] == blwn.OJAMA_SOURCE_TRUTH for r in rows_sorted)


def test_ojama_truth_columns_nan_and_missing_source_for_old_npz(tmp_path: Path) -> None:
    """真値列の無い旧npzでは NaN + ojama_source=MISSING になること。"""
    npz_path = tmp_path / "no_truth.npz"
    _write_synthetic_npz(npz_path, n=3)
    registry = blwn._resolve_indicator_registry("light")
    rows = blwn.convert_one_npz(npz_path, registry)
    for r in rows:
        assert np.isnan(r["ojama_net_balance"])
        assert np.isnan(r["ojama_forecast"])
        assert np.isnan(r["ojama_net_balance_synced"])
        assert r["ojama_source"] == blwn.OJAMA_SOURCE_MISSING


def test_ojama_net_balance_synced_matches_own_minus_opp_over_two(tmp_path: Path) -> None:
    """synced = (own - 相手の直近確定値) / 2 と厳密一致すること (b-2 と同じ
    merge_asof backward パターンの再利用、OJAMA_TRUTH_COLUMNS 直上コメント
    「P4決着」節参照)。

    先頭行 (1P@t=0.0) は相手側の直近確定値が無いため NaN になる
    (`test_diff_is_nan_when_opponent_has_no_prior_snapshot` と同じ仕様)。
    """
    npz_path = tmp_path / "sync.npz"
    _write_npz_with_ojama_truth(
        npz_path, sides=["1P", "2P", "1P", "2P"], t_secs=[0.0, 0.5, 1.0, 1.5],
        net_balance=[10.0, -8.0, 6.0, -4.0], forecast=[0.0, 0.0, 0.0, 0.0],
    )
    registry = blwn._resolve_indicator_registry("light")
    rows = blwn.convert_one_npz(npz_path, registry)
    rows_sorted = sorted(rows, key=lambda r: r["t_sec"])
    assert np.isnan(rows_sorted[0]["ojama_net_balance_synced"])  # 1P@0.0: 相手未確定
    expected_raw = [(-8.0 - 10.0) / 2.0, (6.0 - (-8.0)) / 2.0, (-4.0 - 6.0) / 2.0]
    for row, exp in zip(rows_sorted[1:], expected_raw):
        expected_score = blwn.iv.ojama_net_balance(exp).score
        assert row["ojama_net_balance_synced"] == pytest.approx(expected_score)


def test_ojama_margin_boundary_zero_when_capacity_equals_forecast(tmp_path: Path) -> None:
    """猶予量: 満杯盤面 (吸収余力=0) + 飛来量0 のとき境界値0.5になること
    (収支ゼロ相当、`iv.ojama_net_balance` の慣習と揃える)。"""
    npz_path = tmp_path / "margin_zero.npz"
    _write_npz_with_ojama_truth(
        npz_path, sides=["1P"], t_secs=[0.0], net_balance=[0.0], forecast=[0.0],
        heights=[12],  # 満杯盤面 (absorption raw = ON_FIELD_CAP - 72 = 0)
    )
    registry = blwn._resolve_indicator_registry("light")
    rows = blwn.convert_one_npz(npz_path, registry)
    assert rows[0]["ojama_margin"] == pytest.approx(0.5)


def test_ojama_margin_boundary_one_when_empty_board_no_incoming(tmp_path: Path) -> None:
    """猶予量: 空盤面 (吸収余力=満タン72) + 飛来量0 で上限1.0になること。"""
    npz_path = tmp_path / "margin_full.npz"
    _write_npz_with_ojama_truth(
        npz_path, sides=["1P"], t_secs=[0.0], net_balance=[0.0], forecast=[0.0],
        heights=[0],
    )
    registry = blwn._resolve_indicator_registry("light")
    rows = blwn.convert_one_npz(npz_path, registry)
    assert rows[0]["ojama_margin"] == pytest.approx(1.0)


def test_ojama_margin_clamped_to_zero_when_forecast_far_exceeds_capacity(
    tmp_path: Path,
) -> None:
    """猶予量: 満杯盤面+容量を大幅超過する飛来量で下限0.0にクランプされること。"""
    npz_path = tmp_path / "margin_overflow.npz"
    _write_npz_with_ojama_truth(
        npz_path, sides=["1P"], t_secs=[0.0], net_balance=[0.0], forecast=[200.0],
        heights=[12],
    )
    registry = blwn._resolve_indicator_registry("light")
    rows = blwn.convert_one_npz(npz_path, registry)
    assert rows[0]["ojama_margin"] == pytest.approx(0.0)


def test_ojama_margin_matches_absorption_capacity_formula(tmp_path: Path) -> None:
    """ojama_margin の raw が (ON_FIELD_CAP-count_puyos)-forecast_raw の
    正規化と一致すること (board_puyo_total score からの逆算が正しいことの
    直接検証、境界値以外の一般ケース)。"""
    npz_path = tmp_path / "margin_formula.npz"
    _write_npz_with_ojama_truth(
        npz_path, sides=["1P"], t_secs=[0.0], net_balance=[0.0], forecast=[15.0],
        heights=[5],  # 6列 x 5段 = 30個
    )
    registry = blwn._resolve_indicator_registry("light")
    rows = blwn.convert_one_npz(npz_path, registry)
    absorption_raw = blwn.iv.ON_FIELD_CAP - 30.0
    expected_score = blwn.iv.ojama_net_balance(absorption_raw - 15.0).score
    assert rows[0]["ojama_margin"] == pytest.approx(expected_score)


def test_ojama_margin_nan_when_forecast_unavailable(tmp_path: Path) -> None:
    """forecast が取得不能 (旧npz) なら ojama_margin も NaN になること。"""
    npz_path = tmp_path / "margin_nan.npz"
    _write_synthetic_npz(npz_path, n=2)
    registry = blwn._resolve_indicator_registry("light")
    rows = blwn.convert_one_npz(npz_path, registry)
    for r in rows:
        assert np.isnan(r["ojama_margin"])


def test_csv_output_includes_ojama_truth_columns(tmp_path: Path) -> None:
    """convert_dir の CSV に真値系5列が乗ること。"""
    npz_dir = tmp_path / "npz"
    npz_dir.mkdir()
    _write_synthetic_npz(npz_dir / "a.npz", n=4)
    out_csv = tmp_path / "out.csv"
    blwn.convert_dir(npz_dir, out_csv, profile="light")
    df = pd.read_csv(out_csv)
    for col in blwn.OJAMA_TRUTH_COLUMNS:
        assert col in df.columns


def test_ojama_truth_columns_have_no_diff_or_carry_variants(tmp_path: Path) -> None:
    """おじゃま収支の真値系は grid-only レジストリ外のため diff_/opp_ が
    生成されないこと (b-2 DIFF_* 5分類には意図的に含めない設計、
    OJAMA_TRUTH_COLUMNS 直上コメント参照)。"""
    npz_dir = tmp_path / "npz"
    npz_dir.mkdir()
    _write_synthetic_npz(npz_dir / "a.npz", n=4)
    out_csv = tmp_path / "out.csv"
    blwn.convert_dir(npz_dir, out_csv, profile="light")
    df = pd.read_csv(out_csv)
    for col in (
        "ojama_net_balance", "ojama_forecast", "ojama_net_balance_synced", "ojama_margin",
    ):
        assert f"diff_{col}" not in df.columns
        assert f"opp_{col}" not in df.columns


def test_no_temp_ojama_columns_leak_into_rows(tmp_path: Path) -> None:
    """内部一時列 (アンダースコア始まりの raw 保持用列) が最終行dictに
    残らないこと (CSVには元々乗らないが、行dict自体の掃除も確認する)。"""
    npz_path = tmp_path / "truth.npz"
    _write_npz_with_ojama_truth(
        npz_path, sides=["1P", "2P"], t_secs=[0.0, 0.5],
        net_balance=[5.0, -5.0], forecast=[1.0, 2.0],
    )
    registry = blwn._resolve_indicator_registry("light")
    rows = blwn.convert_one_npz(npz_path, registry)
    for r in rows:
        assert not any(k.startswith("_ojama") for k in r), r.keys()


# ============================
# W12 (2026-08-16、根治P4第一歩): 0-1正規化前の生値2列
# (ojama_net_balance_uncapped/ojama_forecast_uncapped)
# ============================


def test_ojama_uncapped_columns_preserve_value_beyond_saturation(
    tmp_path: Path,
) -> None:
    """予告個数が ON_FIELD_CAP(=72) を超えても uncapped 列は真の生値を
    保持し、正規化列 (ojama_forecast) のように 1.0 に飽和しないこと。"""
    npz_path = tmp_path / "beyond_cap.npz"
    _write_npz_with_ojama_truth(
        npz_path, sides=["1P", "1P"], t_secs=[0.0, 1.0],
        net_balance=[0.0, 0.0], forecast=[72.0, 216.0],
    )
    registry = blwn._resolve_indicator_registry("light")
    rows = blwn.convert_one_npz(npz_path, registry)
    rows_sorted = sorted(rows, key=lambda r: r["t_sec"])
    # 正規化列は両方とも上限飽和で同じ値 (1.0) になる。
    assert rows_sorted[0]["ojama_forecast"] == pytest.approx(1.0)
    assert rows_sorted[1]["ojama_forecast"] == pytest.approx(1.0)
    # uncapped 列は飽和せず72と216を区別できる。
    assert rows_sorted[0]["ojama_forecast_uncapped"] == pytest.approx(72.0)
    assert rows_sorted[1]["ojama_forecast_uncapped"] == pytest.approx(216.0)


def test_ojama_net_balance_uncapped_matches_raw_own_value(tmp_path: Path) -> None:
    """ojama_net_balance_uncapped は own視点の生の収支値そのものと一致
    すること (正規化されない)。"""
    npz_path = tmp_path / "net_uncapped.npz"
    _write_npz_with_ojama_truth(
        npz_path, sides=["1P", "2P"], t_secs=[0.0, 0.5],
        net_balance=[36.0, -80.0], forecast=[0.0, 0.0],
    )
    registry = blwn._resolve_indicator_registry("light")
    rows = blwn.convert_one_npz(npz_path, registry)
    rows_sorted = sorted(rows, key=lambda r: r["t_sec"])
    assert rows_sorted[0]["ojama_net_balance_uncapped"] == pytest.approx(36.0)
    assert rows_sorted[1]["ojama_net_balance_uncapped"] == pytest.approx(-80.0)


def test_ojama_uncapped_columns_nan_for_old_npz_without_truth(
    tmp_path: Path,
) -> None:
    """真値列の無い旧npzでは uncapped 列も NaN になること (0埋めしない、
    欠損の明示)。"""
    npz_path = tmp_path / "no_truth_uncapped.npz"
    _write_synthetic_npz(npz_path, n=3)
    registry = blwn._resolve_indicator_registry("light")
    rows = blwn.convert_one_npz(npz_path, registry)
    for r in rows:
        assert np.isnan(r["ojama_net_balance_uncapped"])
        assert np.isnan(r["ojama_forecast_uncapped"])


def test_csv_output_includes_ojama_uncapped_columns(tmp_path: Path) -> None:
    """convert_dir の CSV に uncapped 2列が乗ること (OJAMA_TRUTH_COLUMNS
    末尾追加分)。"""
    npz_dir = tmp_path / "npz"
    npz_dir.mkdir()
    _write_synthetic_npz(npz_dir / "a.npz", n=4)
    out_csv = tmp_path / "out.csv"
    blwn.convert_dir(npz_dir, out_csv, profile="light")
    df = pd.read_csv(out_csv)
    assert "ojama_net_balance_uncapped" in df.columns
    assert "ojama_forecast_uncapped" in df.columns


def test_ojama_uncapped_columns_have_no_diff_or_carry_variants(
    tmp_path: Path,
) -> None:
    """uncapped 2列も他の真値系と同じく diff_/opp_ が生成されないこと
    (own-perspectiveの収支そのものであり相手との差を取る意味が無いため、
    b-2 DIFF_* 5分類の対象外)。"""
    npz_dir = tmp_path / "npz"
    npz_dir.mkdir()
    _write_synthetic_npz(npz_dir / "a.npz", n=4)
    out_csv = tmp_path / "out.csv"
    blwn.convert_dir(npz_dir, out_csv, profile="light")
    df = pd.read_csv(out_csv)
    for col in ("ojama_net_balance_uncapped", "ojama_forecast_uncapped"):
        assert f"diff_{col}" not in df.columns
        assert f"opp_{col}" not in df.columns


def test_ojama_uncapped_columns_do_not_end_with_raw_suffix() -> None:
    """a-1 決定記録の「*_raw 列は全面禁止」ガード
    (test_convert_one_npz_never_emits_raw_columns/test_csv_output_has_no_
    raw_columns) と名前が衝突しないこと (OJAMA_TRUTH_COLUMNS 直上コメント
    「列名について (`_raw` を避けた理由)」節の意図を固定する回帰テスト)。"""
    for col in ("ojama_net_balance_uncapped", "ojama_forecast_uncapped"):
        assert col in blwn.OJAMA_TRUTH_COLUMNS
        assert not col.endswith("_raw")


def test_existing_ojama_truth_columns_unaffected_by_uncapped_addition(
    tmp_path: Path,
) -> None:
    """uncapped 2列の追加が既存の ojama_net_balance/ojama_forecast/
    ojama_source/ojama_net_balance_synced/ojama_margin の値を変えないこと
    (既存列は1つも壊さない、というタスク要件の直接検証)。"""
    npz_path = tmp_path / "unaffected.npz"
    _write_npz_with_ojama_truth(
        npz_path, sides=["1P", "2P"], t_secs=[0.0, 0.5],
        net_balance=[10.0, -8.0], forecast=[5.0, 3.0],
    )
    registry = blwn._resolve_indicator_registry("light")
    rows = blwn.convert_one_npz(npz_path, registry)
    rows_sorted = sorted(rows, key=lambda r: r["t_sec"])
    assert rows_sorted[0]["ojama_net_balance"] == pytest.approx(
        blwn.iv.ojama_net_balance(10.0).score,
    )
    assert rows_sorted[0]["ojama_forecast"] == pytest.approx(
        blwn.iv.ojama_forecast(5.0).score,
    )
    assert rows_sorted[0]["ojama_source"] == blwn.OJAMA_SOURCE_TRUTH


# ============================
# W12 (2026-08-16、アーキ設計確定分): ojama_forecast_log /
# ojama_forecast_progress_interaction / color_forecast_ratio_own
# ============================


def test_ojama_forecast_log_desaturates_beyond_on_field_cap(tmp_path: Path) -> None:
    """正規化列 ojama_forecast が72個で1.0に飽和する局面でも、
    ojama_forecast_log は72個(0.797付近)と216個(1.0)を区別できること。"""
    npz_path = tmp_path / "log_desat.npz"
    _write_npz_with_ojama_truth(
        npz_path, sides=["1P", "1P"], t_secs=[0.0, 1.0],
        net_balance=[0.0, 0.0], forecast=[72.0, 216.0],
    )
    registry = blwn._resolve_indicator_registry("light")
    rows = blwn.convert_one_npz(npz_path, registry)
    rows_sorted = sorted(rows, key=lambda r: r["t_sec"])
    assert rows_sorted[0]["ojama_forecast"] == pytest.approx(1.0)
    assert rows_sorted[1]["ojama_forecast"] == pytest.approx(1.0)
    import math
    expected_72 = math.log1p(72.0) / math.log1p(blwn.PENDING_ABS_CAP)
    assert rows_sorted[0]["ojama_forecast_log"] == pytest.approx(expected_72)
    assert rows_sorted[1]["ojama_forecast_log"] == pytest.approx(1.0)
    assert rows_sorted[0]["ojama_forecast_log"] < rows_sorted[1]["ojama_forecast_log"]


def test_ojama_forecast_log_nan_for_old_npz_without_truth(tmp_path: Path) -> None:
    """真値列の無い旧npzでは ojama_forecast_log も NaN になること。"""
    npz_path = tmp_path / "log_nan.npz"
    _write_synthetic_npz(npz_path, n=2)
    registry = blwn._resolve_indicator_registry("light")
    rows = blwn.convert_one_npz(npz_path, registry)
    for r in rows:
        assert np.isnan(r["ojama_forecast_log"])
        assert np.isnan(r["ojama_forecast_progress_interaction"])


def test_ojama_forecast_progress_interaction_matches_algebraic_formula(
    tmp_path: Path,
) -> None:
    """ojama_forecast_progress_interaction = ojama_forecast_log ×
    match_progress (match_progress は own board_puyo_total score と
    diff_board_puyo_total から (own+opp_asof)/2 として厳密に再現できること、
    b-2 の merge_asof backward パターンと同じ対応付け)。"""
    npz_path = tmp_path / "interaction.npz"
    _write_npz_with_ojama_truth(
        npz_path, sides=["1P", "2P", "1P", "2P"], t_secs=[0.0, 0.5, 1.0, 1.5],
        net_balance=[0.0, 0.0, 0.0, 0.0], forecast=[100.0, 100.0, 100.0, 100.0],
        heights=[2, 10, 4, 8],
    )
    registry = blwn._resolve_indicator_registry("light")
    rows = blwn.convert_one_npz(npz_path, registry)
    rows_sorted = sorted(rows, key=lambda r: r["t_sec"])
    import math
    forecast_log = math.log1p(100.0) / math.log1p(blwn.PENDING_ABS_CAP)
    # 先頭行 (1P@0.0) は相手の直近確定値が無いため NaN
    # (test_ojama_net_balance_synced_matches_own_minus_opp_over_two と同じ仕様)。
    assert np.isnan(rows_sorted[0]["ojama_forecast_progress_interaction"])
    expected_progress = [
        (10 * 6 / 72.0 + 2 * 6 / 72.0) / 2.0,   # 2P@0.5: own=10段, opp_asof=1P 2段
        (4 * 6 / 72.0 + 10 * 6 / 72.0) / 2.0,   # 1P@1.0: own=4段, opp_asof=2P 10段
        (8 * 6 / 72.0 + 4 * 6 / 72.0) / 2.0,    # 2P@1.5: own=8段, opp_asof=1P 4段
    ]
    for row, expected in zip(rows_sorted[1:], expected_progress):
        assert row["ojama_forecast_log"] == pytest.approx(forecast_log)
        assert row["ojama_forecast_progress_interaction"] == pytest.approx(
            forecast_log * expected,
        )


def test_color_forecast_ratio_own_matches_formula(tmp_path: Path) -> None:
    """color_forecast_ratio_own = color_raw/(color_raw+forecast_raw+EPS) と
    厳密一致すること (合成盤面は全て色ぷよのためcolor_raw=height*6)。"""
    npz_path = tmp_path / "color_ratio.npz"
    _write_npz_with_ojama_truth(
        npz_path, sides=["1P"], t_secs=[0.0], net_balance=[0.0], forecast=[15.0],
        heights=[5],  # 6列x5段=30個、全て色ぷよ
    )
    registry = blwn._resolve_indicator_registry("light")
    rows = blwn.convert_one_npz(npz_path, registry)
    color_raw = 30.0
    expected = color_raw / (color_raw + 15.0 + blwn.COLOR_OJAMA_RATIO_EPS)
    assert rows[0]["color_forecast_ratio_own"] == pytest.approx(expected)


def test_color_forecast_ratio_own_nan_for_old_npz_without_truth(
    tmp_path: Path,
) -> None:
    """真値列の無い旧npzでは color_forecast_ratio_own も NaN になること
    (forecast_raw が取得不能なため)。"""
    npz_path = tmp_path / "color_ratio_nan.npz"
    _write_synthetic_npz(npz_path, n=2)
    registry = blwn._resolve_indicator_registry("light")
    rows = blwn.convert_one_npz(npz_path, registry)
    for r in rows:
        assert np.isnan(r["color_forecast_ratio_own"])


def test_csv_output_includes_w12_architect_columns(tmp_path: Path) -> None:
    """convert_dir の CSV に W12 アーキ確定3列が乗ること。"""
    npz_dir = tmp_path / "npz"
    npz_dir.mkdir()
    _write_synthetic_npz(npz_dir / "a.npz", n=4)
    out_csv = tmp_path / "out.csv"
    blwn.convert_dir(npz_dir, out_csv, profile="light")
    df = pd.read_csv(out_csv)
    for col in (
        "ojama_forecast_log", "ojama_forecast_progress_interaction",
        "color_forecast_ratio_own",
    ):
        assert col in df.columns


def test_w12_architect_columns_have_no_diff_or_carry_variants(
    tmp_path: Path,
) -> None:
    """W12アーキ確定3列も diff_/opp_ が生成されないこと (own-perspective の
    絶対量/比率であり相手との差を取る意味が無いため、5分類・対称化の
    一括変換の対象外)。"""
    npz_dir = tmp_path / "npz"
    npz_dir.mkdir()
    _write_synthetic_npz(npz_dir / "a.npz", n=4)
    out_csv = tmp_path / "out.csv"
    blwn.convert_dir(npz_dir, out_csv, profile="light")
    df = pd.read_csv(out_csv)
    for col in (
        "ojama_forecast_log", "ojama_forecast_progress_interaction",
        "color_forecast_ratio_own",
    ):
        assert f"diff_{col}" not in df.columns
        assert f"opp_{col}" not in df.columns


def test_w12_architect_columns_reuse_existing_constants_not_new_ones() -> None:
    """PENDING_ABS_CAP は src.ojama_accounting からimportした既存216を
    使い、独自の新定数を作っていないこと (アーキ指示の直接検証)。"""
    from src.ojama_accounting import PENDING_ABS_CAP as _canonical_cap

    assert blwn.PENDING_ABS_CAP is _canonical_cap
    assert blwn.PENDING_ABS_CAP == 216


def test_existing_columns_unaffected_by_w12_architect_columns_addition(
    tmp_path: Path,
) -> None:
    """W12アーキ確定3列の追加が既存の ojama_net_balance/ojama_forecast/
    color_ojama_ratio_own 等の値を変えないこと (既存列は1つも壊さない)。"""
    npz_path = tmp_path / "unaffected2.npz"
    _write_npz_with_ojama_truth(
        npz_path, sides=["1P", "2P"], t_secs=[0.0, 0.5],
        net_balance=[10.0, -8.0], forecast=[5.0, 3.0],
    )
    registry = blwn._resolve_indicator_registry("light")
    rows = blwn.convert_one_npz(npz_path, registry)
    rows_sorted = sorted(rows, key=lambda r: r["t_sec"])
    assert rows_sorted[0]["ojama_net_balance"] == pytest.approx(
        blwn.iv.ojama_net_balance(10.0).score,
    )
    assert rows_sorted[0]["ojama_forecast"] == pytest.approx(
        blwn.iv.ojama_forecast(5.0).score,
    )
    color = float(rows_sorted[0]["board_color_puyo_total"])
    ojama = float(rows_sorted[0]["board_ojama_count"])
    assert rows_sorted[0]["color_ojama_ratio_own"] == pytest.approx(
        color / (color + ojama + blwn.COLOR_OJAMA_RATIO_EPS),
    )
