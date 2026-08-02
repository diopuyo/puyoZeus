"""scripts/compute_exchange_delta_winprob.py (Step3) のテスト。

ΔWinProb接続 (アーキ設計 案C) の Step3: 45指標(のうちboard-only版)→勝率
モデルの前後評価とΔWinProb計算・健全性チェック・npz盤面突合を検証する。
LR/isotonic 校正器は軽量な fake object でモックする (実データ学習は重いため)。
"""
from __future__ import annotations

import io
from contextlib import redirect_stdout

import numpy as np
import pandas as pd
import pytest

from src.board import BOARD_COLS, BOARD_ROWS, Board
from scripts.compute_exchange_delta_winprob import (
    BOARD_ONLY_INDICATOR_BASES,
    PhaseWinprobModel,
    T_SEC_MATCH_TOL_SEC,
    _VideoNpzCache,
    _assign_phase_by_puyo_tertile,
    _npz_stem_from_video_id,
    compute_board_only_features,
    compute_delta_winprob_for_event,
    print_sanity_checks,
    reconstruct_event_board_pair,
    winprob_attacker,
    winprob_to_score100,
)
from scripts.label_exchange_outcome import NpzRecord
from src.chain import ChainSimulator


# =============================================================================
# テスト用ヘルパー
# =============================================================================

def _empty_grid() -> list[list[int]]:
    return [[0] * BOARD_COLS for _ in range(BOARD_ROWS)]


def _board_with_cell(row: int, col: int, color: int) -> Board:
    grid = _empty_grid()
    grid[row][col] = color
    return Board.from_list(grid)


def _make_four_connect_board() -> Board:
    """最下段に赤4個 (1連鎖確定) を並べた盤面 (test_exchange_virtual_board.py と同型)。"""
    grid = _empty_grid()
    for col in range(4):
        grid[BOARD_ROWS - 1][col] = 1
    return Board.from_list(grid)


class _FakeScaler:
    """StandardScaler の代わり: 恒等変換 (テストの決定論性を優先)。"""

    def transform(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(x, dtype=float)


class _FakeLR:
    """LogisticRegression の代わり: 先頭指標の diff (index=2、attacker-opponent) をロジットとみなす。

    build_features / winprob_attacker の列順は base ごとに [v1, v2, v1-v2] の
    3つ組のため、index=2 が「attacker視点の先頭指標の優劣差」になる
    (attacker/opponent入れ替えテストで符号が反転することを検証するため)。
    """

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        logit = np.asarray(x, dtype=float)[:, 2]
        p = 1.0 / (1.0 + np.exp(-logit))
        return np.stack([1.0 - p, p], axis=1)


class _FakeIsotonic:
    """IsotonicRegression の代わり: 恒等写像 (校正の有無をテストで切替可能)。"""

    def predict(self, p: "list[float] | np.ndarray") -> np.ndarray:
        return np.asarray(p, dtype=float)


def _make_fake_model() -> PhaseWinprobModel:
    """fake部品からなる PhaseWinprobModel を組み立てる (実LR学習を回避)。"""
    return PhaseWinprobModel(
        scaler=_FakeScaler(), lr=_FakeLR(), isotonic=_FakeIsotonic(),
        oof_auc=0.75, n_train=1000,
    )


# =============================================================================
# BOARD_ONLY_INDICATOR_BASES / compute_board_only_features
# =============================================================================

def test_board_only_indicator_bases_nonempty_and_unique():
    """board-only指標一覧は重複なく、45指標のうちの真部分集合であること。"""
    assert len(BOARD_ONLY_INDICATOR_BASES) == len(set(BOARD_ONLY_INDICATOR_BASES))
    assert 15 <= len(BOARD_ONLY_INDICATOR_BASES) <= 30


def test_compute_board_only_features_returns_all_names_in_0_1_range():
    """全指標が計算され、スコアが0〜1に正規化されていること (CLAUDE.md規約)。"""
    board = _make_four_connect_board()
    sim = ChainSimulator()
    feats = compute_board_only_features(board, sim)
    assert set(feats.keys()) == set(BOARD_ONLY_INDICATOR_BASES)
    for name, v in feats.items():
        assert -1e-9 <= v <= 1.0 + 1e-9, f"{name} が0〜1範囲外: {v}"


# =============================================================================
# winprob_attacker / winprob_to_score100
# =============================================================================

def test_winprob_to_score100_is_0_to_100_percent():
    """p=0.5 が50、p=1.0 が100、p=0.0 が0になること (0〜100%スケール)。

    2026-08-02 修正: 旧実装は (p-0.5)*200 の対称アドバンテージスコア
    (-100〜+100) だったため、before/afterの差 (delta_winprob) が理論上
    ±200まで発生しうるバグがあった (main検品で実測max=190.10を確認)。
    winprob_before/after 自体を0〜100%の勝率に変更し、差が理論上も
    必ず±100に収まるよう修正した。
    """
    assert winprob_to_score100(0.5) == pytest.approx(50.0)
    assert winprob_to_score100(1.0) == pytest.approx(100.0)
    assert winprob_to_score100(0.0) == pytest.approx(0.0)


def test_delta_winprob_never_exceeds_100_in_magnitude():
    """winprob_before/afterが0〜100%なので、差は理論上も必ず|delta|<=100。"""
    for p_before in (0.0, 0.3, 0.5, 0.7, 1.0):
        for p_after in (0.0, 0.3, 0.5, 0.7, 1.0):
            wb, wa = winprob_to_score100(p_before), winprob_to_score100(p_after)
            assert abs(wa - wb) <= 100.0 + 1e-9


def test_winprob_attacker_uses_symmetric_feature_slots():
    """attacker/opponentを入れ替えると勝率もほぼ反転すること (fakeLRの1特徴目のみ差で判定)。"""
    models = {"中": _make_fake_model()}
    board = _make_four_connect_board()
    sim = ChainSimulator()
    feats_a = compute_board_only_features(board, sim)
    feats_b = {k: 0.0 for k in feats_a}  # 全指標0の対戦相手
    p_attacker_wins = winprob_attacker(models, "中", feats_a, feats_b)
    p_opponent_wins = winprob_attacker(models, "中", feats_b, feats_a)
    # 1特徴目 (base_1p) の符号が反転するため確率は対称に近い値になる
    assert p_attacker_wins > 0.5
    assert p_opponent_wins < 0.5


def test_winprob_attacker_raises_on_unknown_phase():
    """未学習の位相を指定したら KeyError (silent fallback 禁止)。"""
    models = {"中": _make_fake_model()}
    feats = {k: 0.5 for k in BOARD_ONLY_INDICATOR_BASES}
    with pytest.raises(KeyError):
        winprob_attacker(models, "序", feats, feats)


# =============================================================================
# compute_delta_winprob_for_event
# =============================================================================

def test_compute_delta_winprob_for_event_returns_result_with_expected_fields():
    """発火前後のΔWinProbが -100〜+100 範囲で返ること。"""
    models = {"中": _make_fake_model()}
    fire_board = _make_four_connect_board()
    opp_board = Board()
    sim = ChainSimulator()
    result = compute_delta_winprob_for_event(
        fire_board, opp_board, "中", net_ojama_after_pred=6.0,
        models=models, simulator=sim,
    )
    assert 0.0 <= result.winprob_before <= 100.0
    assert 0.0 <= result.winprob_after <= 100.0
    assert result.delta_winprob == pytest.approx(result.winprob_after - result.winprob_before)
    assert abs(result.delta_winprob) <= 100.0 + 1e-9
    assert isinstance(result.attacker_dead_after, bool)
    assert isinstance(result.opponent_dead_after, bool)


def test_compute_delta_winprob_for_event_propagates_nan_error():
    """net_ojama_after_pred が NaN なら Step2 由来の ValueError がそのまま伝播する。"""
    models = {"中": _make_fake_model()}
    fire_board = _make_four_connect_board()
    opp_board = Board()
    sim = ChainSimulator()
    with pytest.raises(ValueError):
        compute_delta_winprob_for_event(
            fire_board, opp_board, "中", net_ojama_after_pred=float("nan"),
            models=models, simulator=sim,
        )


# =============================================================================
# _assign_phase_by_puyo_tertile
# =============================================================================

def test_assign_phase_by_puyo_tertile_produces_three_labels():
    """3分位で 序/中/終 の3値に分割されること。"""
    values = np.linspace(0.0, 1.0, 300)
    labels, q_low, q_high = _assign_phase_by_puyo_tertile(values)
    assert set(labels) == {"序", "中", "終"}
    assert q_low < q_high
    assert (labels[values <= q_low] == "序").all()
    assert (labels[values > q_high] == "終").all()


# =============================================================================
# _npz_stem_from_video_id
# =============================================================================

def test_npz_stem_strips_video_prefix():
    """'video_c10' -> 'c10' (npzファイル名は prefix無し)。"""
    assert _npz_stem_from_video_id("video_c10") == "c10"


def test_npz_stem_passthrough_without_prefix():
    """既に prefix が無い場合はそのまま返す (後方互換の安全弁)。"""
    assert _npz_stem_from_video_id("c10") == "c10"


def test_build_demo_viz_video_id_normalization_matches_prefixed_column(tmp_path, monkeypatch):
    """--demo-video に prefix無し ("c50") を渡しても delta_df["video_id"]

    ("video_c50") と一致してイベントが見つかること (実データで発覚した
    突合漏れバグの回帰テスト)。npz読込・タイムライン生成は重いため
    _load_video_npz と _build_stable_timeline をモックする。
    """
    import scripts.compute_exchange_delta_winprob as m

    monkeypatch.setattr(m, "_load_video_npz", lambda video_id, npz_dir: object())
    monkeypatch.setattr(
        m, "_build_stable_timeline",
        lambda cache, game_idx, models, sim: pd.DataFrame({"t_sec": [0.0], "winprob_1p": [0.0]}),
    )
    monkeypatch.setattr(m, "render_demo_viz", lambda *a, **k: None)
    delta_df = pd.DataFrame({
        "video_id": ["video_c50", "video_c50"], "game_idx": [0, 0],
        "t_sec": [1.0, 2.0], "fire_side": ["1P", "1P"],
        "winprob_before": [0.0, 0.0], "winprob_after": [1.0, 1.0],
        "delta_winprob": [1.0, 1.0], "match_failed": [False, False],
    })
    out_path = m.build_demo_viz("c50", tmp_path, {"中": _make_fake_model()}, delta_df, tmp_path)
    assert out_path is not None


# =============================================================================
# reconstruct_event_board_pair (npz突合)
# =============================================================================

def _make_distinct_board(idx: int) -> Board:
    """index ごとに一意な単セル盤面 (等価比較で判別できるよう色/列を変える)。"""
    return _board_with_cell(BOARD_ROWS - 1, idx % BOARD_COLS, (idx % 5) + 1)


def _make_fire_side_grids() -> np.ndarray:
    """発火側(fire_side)の5フレーム分グリッド (index3で+200スコア発火)。"""
    return np.stack([_make_distinct_board(i).to_dict()["grid"] for i in range(5)]).astype(np.int8)


def _make_video_cache() -> _VideoNpzCache:
    """t_sec=[0,1,2,3,4] で index3にスコア+200ジャンプ(発火)する合成 NpzRecord ペア。

    fire側 board_ref_index は max(0, first_idx-1) = 2 になる想定
    (label_exchange_outcome._merge_fire_event_clusters の定義通り)。
    opp側は t_sec=[0.5,1.5,2.5,3.4,4.5] とし、発火時刻3.0に最も近いのは index3 (t=3.4)。
    """
    fire_grids = _make_fire_side_grids()
    opp_grids = np.stack([_make_distinct_board(10 + i).to_dict()["grid"] for i in range(5)]).astype(np.int8)
    r1p = NpzRecord(
        video_id="video_test", side="1P",
        t_sec=np.array([0.0, 1.0, 2.0, 3.0, 4.0], dtype=np.float32),
        game_idx=np.zeros(5, dtype=np.int32),
        grids=fire_grids,
        won=np.zeros(5, dtype=np.float32),
        score=np.array([0, 0, 0, 200, 200], dtype=np.int32),
    )
    r2p = NpzRecord(
        video_id="video_test", side="2P",
        t_sec=np.array([0.5, 1.5, 2.5, 3.4, 4.5], dtype=np.float32),
        game_idx=np.zeros(5, dtype=np.int32),
        grids=opp_grids,
        won=np.ones(5, dtype=np.float32),
        score=np.zeros(5, dtype=np.int32),
    )
    return _VideoNpzCache(r1p=r1p, r2p=r2p)


def test_reconstruct_event_board_pair_matches_expected_boards():
    """t_secが厳密一致する場合、正しいboard_ref_index/最近傍opp盤面を返す。"""
    cache = _make_video_cache()
    result = reconstruct_event_board_pair(cache, game_idx=0, t_sec=3.0, fire_side="1P")
    assert result is not None
    fire_board, opp_board = result
    assert fire_board == _make_distinct_board(2)  # board_ref_index=2
    assert opp_board == _make_distinct_board(13)  # opp index3 (t=3.4が最近傍)


def test_reconstruct_event_board_pair_returns_none_when_tolerance_exceeded():
    """t_secがクラスタ時刻から許容誤差を超えてズレていたら None (突合失敗)。"""
    cache = _make_video_cache()
    far_t = 3.0 + T_SEC_MATCH_TOL_SEC * 10
    result = reconstruct_event_board_pair(cache, game_idx=0, t_sec=far_t, fire_side="1P")
    assert result is None


def test_reconstruct_event_board_pair_returns_none_for_missing_game_idx():
    """存在しない game_idx を指定したら None。"""
    cache = _make_video_cache()
    result = reconstruct_event_board_pair(cache, game_idx=99, t_sec=3.0, fire_side="1P")
    assert result is None


# =============================================================================
# print_sanity_checks
# =============================================================================

def _make_sanity_df() -> pd.DataFrame:
    """健全性チェック用の合成 delta_winprob DataFrame。"""
    rows = []
    for i in range(20):
        rows.append({
            "video_id": "video_test", "game_idx": 0, "t_sec": float(i),
            "fire_side": "1P", "phase": "中" if i % 2 == 0 else "終",
            "match_failed": False,
            "winprob_before": 0.0, "winprob_after": float(10 - i),
            "delta_winprob": float(10 - i),
            "attacker_dead_after": i == 0, "opponent_dead_after": i < 3,
        })
    return pd.DataFrame(rows)


def test_print_sanity_checks_runs_without_error_and_reports_key_metrics():
    """健全性チェックが例外なく走り、主要な見出しを出力すること。"""
    df = _make_sanity_df()
    buf = io.StringIO()
    with redirect_stdout(buf):
        print_sanity_checks(df)
    out = buf.getvalue()
    assert "発火側有利方向" in out
    assert "窒息フラグ発生率" in out
    assert "発火したのに勝率が下がる" in out


def test_print_sanity_checks_handles_all_match_failed():
    """全行 match_failed=True でも例外を出さず0件として処理すること。"""
    df = _make_sanity_df()
    df["match_failed"] = True
    buf = io.StringIO()
    with redirect_stdout(buf):
        print_sanity_checks(df)
    assert "突合成功 0/20" in buf.getvalue()
