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

from src.board import BOARD_COLS, BOARD_ROWS, COLOR_BLUE, COLOR_RED, Board
from scripts.compute_exchange_delta_winprob import (
    BOARD_ONLY_INDICATOR_BASES,
    CHAIN_WINDOW_FALLBACK_TAIL_SEC,
    ChainInProgressWindow,
    EventActivityWindow,
    PhaseWinprobModel,
    T_SEC_MATCH_TOL_SEC,
    _VideoNpzCache,
    _assign_phase_by_puyo_tertile,
    _build_mirror_paired,
    _build_stable_timeline,
    LETHAL_CLAMP_FAVOR_PCT,
    _aggregate_known_pending_net_ojama,
    _find_active_chain_window,
    _is_airborne_at,
    _lethal_readout_clamp,
    _net_pending_after_cancellation,
    _realizable_counter_ojama,
    _next_own_stable_time,
    _npz_stem_from_video_id,
    apply_mutual_exchange_adjustment,
    build_chain_in_progress_windows,
    build_event_activity_windows,
    compute_board_only_features,
    compute_delta_winprob_for_event,
    find_mutual_exchange_partner,
    ignition_time_for_event,
    print_sanity_checks,
    reconstruct_event_board_pair,
    train_winprob_models,
    winprob_attacker,
    winprob_to_score100,
)
from scripts.label_exchange_outcome import NpzRecord
from scripts.model_indicator_win import build_features
from src.chain import ChainSimulator
from src.exchange_virtual_board import reconstruct_virtual_board_pair


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


# =============================================================================
# 2026-08-03 指摘1: 学習データ対称化 (_build_mirror_paired / train_winprob_models)
# =============================================================================

def test_build_mirror_paired_swaps_1p_2p_columns():
    """`_1p`/`_2p` suffix の列が丸ごと入れ替わること。"""
    paired = pd.DataFrame({"foo_1p": [1.0, 2.0], "foo_2p": [3.0, 4.0]})
    mirror = _build_mirror_paired(paired)
    assert list(mirror["foo_1p"]) == [3.0, 4.0]
    assert list(mirror["foo_2p"]) == [1.0, 2.0]


def test_build_mirror_paired_leaves_unpaired_columns_untouched():
    """対応する `_2p` 列が無い列 (t_diff等) はそのまま保持されること。"""
    paired = pd.DataFrame({"foo_1p": [1.0], "foo_2p": [2.0], "t_diff": [0.05]})
    mirror = _build_mirror_paired(paired)
    assert list(mirror["t_diff"]) == [0.05]


def test_build_features_diff_sign_flips_after_mirror():
    """鏡像複製後は diff (=1p-2p) の符号が反転すること (build_features経由)。"""
    paired = pd.DataFrame({"board_puyo_total_1p": [0.2], "board_puyo_total_2p": [0.7]})
    mirror = _build_mirror_paired(paired)
    feat_orig = build_features(paired, ["board_puyo_total"])
    feat_mirror = build_features(mirror, ["board_puyo_total"])
    assert feat_orig["board_puyo_total_diff"].iloc[0] == pytest.approx(
        -feat_mirror["board_puyo_total_diff"].iloc[0])
    # 1p/2p 自体も入れ替わっていること
    assert feat_mirror["board_puyo_total_1p"].iloc[0] == pytest.approx(0.7)
    assert feat_mirror["board_puyo_total_2p"].iloc[0] == pytest.approx(0.2)


def test_phase_metric_1p_plus_2p_sum_is_swap_invariant():
    """位相判定量 (1P+2P合計) は入替に対して不変であること (指摘1の実害根因の回帰テスト)。

    1P単独の値を位相判定に使うと、鏡像複製後に元サンプルと鏡像とで
    異なる位相バケツに分かれてしまい (train_winprob_models docstring
    参照)、位相別モデルの対称化が崩れて空盤面が50%にならない実害が
    出ていた (main実測で確認)。1P+2P合計ならこの実害が原理的に起きない
    ことを固定する。
    """
    paired = pd.DataFrame({
        "board_puyo_total_1p": [0.1, 0.9, 0.5],
        "board_puyo_total_2p": [0.8, 0.05, 0.5],
    })
    mirror = _build_mirror_paired(paired)
    combined_orig = (paired["board_puyo_total_1p"] + paired["board_puyo_total_2p"]).values
    combined_mirror = (mirror["board_puyo_total_1p"] + mirror["board_puyo_total_2p"]).values
    np.testing.assert_allclose(combined_orig, combined_mirror)


def _make_synthetic_labeled_win_csv(tmp_path, n_videos: int = 40, rows_per_video: int = 8,
                                    seed: int = 0):
    """train_winprob_models を実際に走らせる軽量な合成 labeled_win.csv を作る。

    691試合フル学習(数十秒)を避けるため、23 board-only 指標全てを持つ
    ランダムな合成データ (video_id 単位でグルーピング可能) を生成する。
    won の整合性 (won_1p + won_2p == 1) は pair_sides_for_win の要件。
    """
    rng = np.random.default_rng(seed)
    records: list[dict] = []
    for v in range(n_videos):
        vid = f"synthv{v}"
        for r in range(rows_per_video):
            t = r * 5.0
            won1 = int(rng.uniform() < 0.5)
            row1 = {"video_id": vid, "side": "1P", "t_sec": t, "won": won1}
            row2 = {"video_id": vid, "side": "2P", "t_sec": t + 0.05, "won": 1 - won1}
            for base in BOARD_ONLY_INDICATOR_BASES:
                row1[base] = float(rng.uniform(0.0, 1.0))
                row2[base] = float(rng.uniform(0.0, 1.0))
            records.append(row1)
            records.append(row2)
    df = pd.DataFrame(records)
    path = tmp_path / "synthetic_labeled_win.csv"
    df.to_csv(path, index=False)
    return path


def test_train_winprob_models_symmetric_board_gives_exactly_50pct(tmp_path):
    """対称化学習後、完全に対称な局面 (空盤面) の予測勝率がちょうど50%になること。

    2026-08-03 userレビュー指摘1 の受け入れテスト (main実測: 671試合の
    1P勝率52.0%をモデルが事前確率として学習していた)。数学的根拠は
    _build_mirror_paired docstring 参照。
    """
    csv_path = _make_synthetic_labeled_win_csv(tmp_path)
    models = train_winprob_models(csv_path)
    assert len(models) > 0
    sim = ChainSimulator()
    empty = Board.from_list([[0] * BOARD_COLS for _ in range(BOARD_ROWS)])
    feats = compute_board_only_features(empty, sim)
    for phase, _model in models.items():
        p = winprob_attacker(models, phase, feats, feats)
        assert p == pytest.approx(0.5, abs=1e-6), f"位相{phase}で対称局面が50%にならない: p={p}"


def test_train_winprob_models_nonempty_symmetric_board_gives_50pct(tmp_path):
    """空盤面に限らず、両者が全く同じ非空盤面でも50%になること。"""
    csv_path = _make_synthetic_labeled_win_csv(tmp_path)
    models = train_winprob_models(csv_path)
    sim = ChainSimulator()
    grid = [[0] * BOARD_COLS for _ in range(BOARD_ROWS)]
    grid[BOARD_ROWS - 1] = [1, 1, 2, 2, 3, 3]
    grid[BOARD_ROWS - 2] = [1, 1, 2, 2, 3, 3]
    board = Board.from_list(grid)
    feats = compute_board_only_features(board, sim)
    for phase in models:
        p = winprob_attacker(models, phase, feats, feats)
        assert p == pytest.approx(0.5, abs=1e-6)


# =============================================================================
# 2026-08-03 指摘2: タイムライン評価密度 (_build_stable_timeline)
# =============================================================================

def _make_dense_video_cache(n1: int = 20, n2: int = 15) -> _VideoNpzCache:
    """1P n1点・2P n2点、時刻が互いにずれた合成 npz キャッシュ (密度検証用)。"""
    grids1 = np.stack(
        [_make_distinct_board(i).to_dict()["grid"] for i in range(n1)]).astype(np.int8)
    grids2 = np.stack(
        [_make_distinct_board(100 + i).to_dict()["grid"] for i in range(n2)]).astype(np.int8)
    r1p = NpzRecord(
        video_id="video_dense", side="1P",
        t_sec=np.arange(n1, dtype=np.float32) * 1.0,
        game_idx=np.zeros(n1, dtype=np.int32), grids=grids1,
        won=np.zeros(n1, dtype=np.float32), score=np.zeros(n1, dtype=np.int32),
    )
    r2p = NpzRecord(
        video_id="video_dense", side="2P",
        t_sec=np.arange(n2, dtype=np.float32) * 1.3 + 0.4,
        game_idx=np.zeros(n2, dtype=np.int32), grids=grids2,
        won=np.ones(n2, dtype=np.float32), score=np.zeros(n2, dtype=np.int32),
    )
    return _VideoNpzCache(r1p=r1p, r2p=r2p)


def test_build_stable_timeline_no_longer_over_decimates():
    """旧実装は DEMO_FRAME_STRIDE=15 の二重間引きで n1点が n1/15点まで
    減っていた (main実測: c61 g16 で84秒中7点)。新実装は間引きせず、
    両サイドの和集合点数に近い点数を返すこと。
    """
    cache = _make_dense_video_cache(n1=20, n2=15)
    models = {"序": _make_fake_model(), "中": _make_fake_model(), "終": _make_fake_model()}
    sim = ChainSimulator()
    df = _build_stable_timeline(cache, game_idx=0, models=models, simulator=sim)
    # 和集合は最大 n1+n2=35点 (重複時刻ぶん減る可能性はあるが、少なくとも
    # 旧実装の 20/15≈1点 とは桁違いに多いはず)。
    assert len(df) >= 20, f"間引きが再発している可能性: {len(df)}点"
    assert len(df) <= 20 + 15


def test_build_stable_timeline_covers_union_of_both_sides_times():
    """評価時刻が両サイドの全STABLE時刻の和集合に含まれること (前方保持不能な
    最初の欠落区間を除く)。
    """
    cache = _make_dense_video_cache(n1=20, n2=15)
    models = {"序": _make_fake_model(), "中": _make_fake_model(), "終": _make_fake_model()}
    sim = ChainSimulator()
    df = _build_stable_timeline(cache, game_idx=0, models=models, simulator=sim)
    t_min_valid = max(cache.r1p.t_sec.min(), cache.r2p.t_sec.min())
    expected_times = sorted(
        t for t in set(cache.r1p.t_sec.tolist()) | set(cache.r2p.t_sec.tolist())
        if t >= t_min_valid
    )
    assert df["t_sec"].tolist() == pytest.approx(expected_times)


def test_build_stable_timeline_empty_side_returns_empty_df():
    """片側にフレームが無い場合は空 DataFrame を返す (例外を出さない)。"""
    cache = _make_dense_video_cache(n1=20, n2=15)
    models = {"中": _make_fake_model()}
    sim = ChainSimulator()
    df = _build_stable_timeline(cache, game_idx=999, models=models, simulator=sim)
    assert len(df) == 0
    assert list(df.columns) == ["t_sec", "winprob_1p", "is_uncertain"]


# =============================================================================
# 2026-08-03 方針(b): 表示側「凍結検知」による判定保留
# =============================================================================

def _make_video_cache_with_2p_freeze() -> _VideoNpzCache:
    """1Pは2秒刻みで20秒まで継続、2Pはt=4.5で更新が完全に止まる合成キャッシュ。

    FREEZE_DETECTION_THRESHOLD_SEC(=6.0秒)超の沈黙を作るための固定具。
    """
    n1 = 11
    t1 = np.arange(n1, dtype=np.float32) * 2.0  # 0,2,4,...,20
    grids1 = np.stack([_make_distinct_board(i).to_dict()["grid"] for i in range(n1)]).astype(np.int8)
    t2 = np.array([0.5, 2.5, 4.5], dtype=np.float32)  # ここで更新が止まる (=凍結)
    grids2 = np.stack([_make_distinct_board(100 + i).to_dict()["grid"] for i in range(3)]).astype(np.int8)
    r1p = NpzRecord(video_id="video_freeze", side="1P", t_sec=t1,
                    game_idx=np.zeros(n1, dtype=np.int32), grids=grids1,
                    won=np.zeros(n1, dtype=np.float32), score=np.zeros(n1, dtype=np.int32))
    r2p = NpzRecord(video_id="video_freeze", side="2P", t_sec=t2,
                    game_idx=np.zeros(3, dtype=np.int32), grids=grids2,
                    won=np.ones(3, dtype=np.float32), score=np.zeros(3, dtype=np.int32))
    return _VideoNpzCache(r1p=r1p, r2p=r2p)


def test_build_stable_timeline_flags_uncertain_after_freeze_threshold():
    """相手側(2P)がFREEZE_DETECTION_THRESHOLD_SEC秒を超えて更新が無い評価点は
    is_uncertain=True になる (方針(b))。
    """
    cache = _make_video_cache_with_2p_freeze()
    models = {"序": _make_fake_model(), "中": _make_fake_model(), "終": _make_fake_model()}
    sim = ChainSimulator()
    df = _build_stable_timeline(cache, game_idx=0, models=models, simulator=sim)
    before = df.loc[df["t_sec"] == 10.0, "is_uncertain"].iloc[0]   # staleness=5.5<=6.0
    after = df.loc[df["t_sec"] == 12.0, "is_uncertain"].iloc[0]    # staleness=7.5>6.0
    assert bool(before) is False
    assert bool(after) is True


def test_build_stable_timeline_holds_last_good_value_during_freeze():
    """凍結中は新規計算値でなく凍結直前の最後の確定値を保持する (方針(b))。"""
    cache = _make_video_cache_with_2p_freeze()
    models = {"序": _make_fake_model(), "中": _make_fake_model(), "終": _make_fake_model()}
    sim = ChainSimulator()
    df = _build_stable_timeline(cache, game_idx=0, models=models, simulator=sim)
    # t=10.0 (staleness=5.5<=6.0) が判定保留に入る直前の最後の確定値。
    last_good = float(df.loc[df["t_sec"] == 10.0, "winprob_1p"].iloc[0])
    frozen_values = df.loc[df["t_sec"] > 10.0, "winprob_1p"].to_numpy()
    assert len(frozen_values) > 0
    assert np.allclose(frozen_values, last_good)


def test_build_stable_timeline_no_freeze_never_uncertain():
    """両サイドとも通常更新が続く (_make_dense_video_cache) 場合は
    is_uncertain が常にFalse (誤検知しないことの確認)。
    """
    cache = _make_dense_video_cache(n1=20, n2=15)
    models = {"序": _make_fake_model(), "中": _make_fake_model(), "終": _make_fake_model()}
    sim = ChainSimulator()
    df = _build_stable_timeline(cache, game_idx=0, models=models, simulator=sim)
    assert not df["is_uncertain"].any()


# =============================================================================
# 2026-08-03 指摘2/Fix B: 連鎖中の仮想盤面ウィンドウ
# =============================================================================

def _make_events_df_for_chain_window(
    t_sec: float = 3.0, fire_side: str = "1P", approx_fire_chains: float = 2.0,
    net_ojama_pred: float = 40.0, game_idx: int = 0,
) -> pd.DataFrame:
    """_make_video_cache() (t=[0,1,2,3,4] 1P / t=[0.5,1.5,2.5,3.4,4.5] 2P、
    index3で+200スコア発火) と対になる1発火イベント分の DataFrame を作る。
    """
    return pd.DataFrame([{
        "t_sec": t_sec, "fire_side": fire_side, "approx_fire_chains": approx_fire_chains,
        "game_idx": game_idx, "stack_net_ojama_after_pred": net_ojama_pred,
    }])


def test_ignition_time_for_event_uses_prev_own_stable_time():
    """発火側自身の直前STABLE時刻を返す (Fix C、近似式は廃止済み)。"""
    cache = _make_video_cache()  # r1p.t_sec = [0,1,2,3,4]
    t = ignition_time_for_event(cache, "1P", game_idx=0, t_sec=3.0)
    assert t == pytest.approx(2.0)


def test_ignition_time_for_event_returns_t_sec_when_no_earlier_snapshot():
    """直前STABLEが無ければ t_sec そのもの (ウィンドウ長ゼロの安全弁)。"""
    cache = _make_video_cache()
    t = ignition_time_for_event(cache, "1P", game_idx=0, t_sec=0.0)
    assert t == pytest.approx(0.0)


def test_next_own_stable_time_finds_strictly_later_snapshot():
    """発火側自身の「次の(t_fireより厳密に後の)STABLE時刻」を返す。"""
    cache = _make_video_cache()  # r1p.t_sec = [0,1,2,3,4]
    t = _next_own_stable_time(cache, "1P", game_idx=0, t_fire=3.0)
    assert t == pytest.approx(4.0)


def test_next_own_stable_time_falls_back_when_no_later_snapshot():
    """次のSTABLEが無ければ CHAIN_WINDOW_FALLBACK_TAIL_SEC 後で打ち切る。"""
    cache = _make_video_cache()
    t = _next_own_stable_time(cache, "1P", game_idx=0, t_fire=4.0)
    assert t == pytest.approx(4.0 + CHAIN_WINDOW_FALLBACK_TAIL_SEC)


def test_build_chain_in_progress_windows_uses_virtual_board_pair():
    """ウィンドウの盤面が reconstruct_virtual_board_pair の出力と一致すること。"""
    cache = _make_video_cache()
    sim = ChainSimulator()
    events_df = _make_events_df_for_chain_window()
    windows = build_chain_in_progress_windows(events_df, cache, sim)
    assert len(windows) == 1
    w = windows[0]
    assert w.ignition_sec == pytest.approx(2.0)  # 発火側自身の直前STABLE (t=2.0)
    assert w.window_end_sec == pytest.approx(4.0)

    pair = reconstruct_event_board_pair(cache, 0, 3.0, "1P")
    fire_board, opp_board = pair
    vpair = reconstruct_virtual_board_pair(fire_board, opp_board, 40.0, simulator=sim)
    assert w.fire_side == "1P"
    assert w.board_after == vpair.attacker_board_after


def test_find_active_chain_window_returns_none_outside_range():
    windows = [ChainInProgressWindow(fire_side="1P", ignition_sec=2.0, window_end_sec=4.0,
                                     board_after=Board())]
    assert _find_active_chain_window(windows, 1.9) is None
    assert _find_active_chain_window(windows, 4.0) is None  # 終端は排他的


def test_find_active_chain_window_returns_window_inside_range():
    w = ChainInProgressWindow(fire_side="1P", ignition_sec=2.0, window_end_sec=4.0, board_after=Board())
    assert _find_active_chain_window([w], 3.0) is w


def test_build_stable_timeline_uses_virtual_board_inside_window():
    """指摘2の受け入れテスト: ウィンドウ内の評価時刻は仮想盤面の特徴量で計算されること。

    net_ojama_pred=-6.0 (負値=攻撃側自身が着弾を受ける想定) を使う。
    _make_distinct_board は全て単一セル(puyo数=1)のため、素の前方保持
    (差し替え無し) では board_puyo_total 差が常に0になり見分けが付かない
    (欠陥E-2修正で相手側は live のため、攻撃側の変化で区別する必要がある)。
    """
    cache = _make_video_cache()
    sim = ChainSimulator()
    models = {"序": _make_fake_model(), "中": _make_fake_model(), "終": _make_fake_model()}
    events_df = _make_events_df_for_chain_window(net_ojama_pred=-6.0)
    windows = build_chain_in_progress_windows(events_df, cache, sim)

    df_with_windows = _build_stable_timeline(cache, 0, models, sim, chain_windows=windows)
    df_without_windows = _build_stable_timeline(cache, 0, models, sim, chain_windows=None)

    # ウィンドウ [ignition=2.0, 4.0) に入る評価時刻 (2.5, 3.0, 3.4) は
    # 仮想盤面特徴量から計算した期待値と一致すること。
    def _row_at(df: pd.DataFrame, t: float) -> pd.Series:
        matches = df.loc[np.isclose(df["t_sec"].values, t)]
        assert len(matches) == 1, f"t={t} の行が見つからない: {df['t_sec'].tolist()}"
        return matches.iloc[0]

    # 発火側(1P)は固定仮想盤面、相手側(2P)はt=2.5時点のlive実測盤面
    # (=_make_distinct_board(12)、_make_video_cache のr2p.grids[2]) を使う
    # (欠陥E-2: 相手側は固定しない)。
    w = windows[0]
    f1 = compute_board_only_features(w.board_after, sim)
    f2 = compute_board_only_features(_make_distinct_board(12), sim)
    expected_in_window = winprob_to_score100(winprob_attacker(models, "中", f1, f2))
    assert _row_at(df_with_windows, 2.5)["winprob_1p"] == pytest.approx(expected_in_window)

    # ウィンドウ外 (t=0.5、両サイド最初のSTABLE時刻) は素の前方保持と一致
    # (差し替え無し版と同じ値)。
    assert _row_at(df_with_windows, 0.5)["winprob_1p"] == pytest.approx(
        _row_at(df_without_windows, 0.5)["winprob_1p"])

    # ウィンドウ内 (t=3.0) は差し替え無し版と異なる値になっているはず
    # (=前方保持の凍結盤面でなく仮想盤面が使われている証拠)。
    assert _row_at(df_with_windows, 3.0)["winprob_1p"] != pytest.approx(
        _row_at(df_without_windows, 3.0)["winprob_1p"])


def test_build_stable_timeline_chain_windows_default_none_backward_compat():
    """chain_windows 省略時は従来通り (既存呼出元の後方互換)。"""
    cache = _make_video_cache()
    sim = ChainSimulator()
    models = {"序": _make_fake_model(), "中": _make_fake_model(), "終": _make_fake_model()}
    df_explicit_none = _build_stable_timeline(cache, 0, models, sim, None)
    df_omitted = _build_stable_timeline(cache, 0, models, sim)
    pd.testing.assert_frame_equal(df_explicit_none, df_omitted)


# =============================================================================
# 2026-08-03 指摘 欠陥D/Fix F: 相打ち (時間的因果関係のある発火) の相殺
# =============================================================================
#
# _make_video_cache() (r1p.t_sec=[0,1,2,3,4] / r2p.t_sec=[0.5,1.5,2.5,3.4,4.5])
# を使う。1P発火(t=3.0, ignition=2.0) は 2P発火(t=2.5, ignition=1.5) の
# 飛行区間 [1.5, 2.5) の**外** (2.0 は 1.5〜2.5 の間ではあるので実は内側!
# 逆に 2P の点火(1.5)は1Pの飛行 [2.0,3.0) の外) — 一方向の因果:
#   1P点火(2.0) は 2P飛行中[1.5,2.5) に含まれる → 1P→2Pの相殺が成立
#   2P点火(1.5) は 1P飛行(まだ点火前、[2.0,3.0)) に含まれない → 相殺不成立
# つまり「1Pイベントのみ相殺され、2Pイベントは相殺されない」非対称な結果に
# なることを検証する (match_01/match_05 の実測パターンと同型)。

def _make_causal_events_df(net_1p: float = 100.0, net_2p: float = 30.0) -> pd.DataFrame:
    return pd.DataFrame([
        {"t_sec": 3.0, "fire_side": "1P", "game_idx": 0, "net_ojama_after": net_1p},
        {"t_sec": 2.5, "fire_side": "2P", "game_idx": 0, "net_ojama_after": net_2p},
    ])


def _make_non_overlapping_events_df() -> pd.DataFrame:
    """1P(t=3.0, ignition=2.0) と 2P(t=0.5, ignition=0.5未満=t_secに等しい)
    は因果的に絡まない (2Pの発火は1Pの点火より遥かに前に完了している)。
    """
    return pd.DataFrame([
        {"t_sec": 3.0, "fire_side": "1P", "game_idx": 0, "net_ojama_after": 100.0},
        {"t_sec": 0.5, "fire_side": "2P", "game_idx": 0, "net_ojama_after": 30.0},
    ])


def test_build_event_activity_windows_matches_fix_b_and_c_boundaries():
    """活動窓の開始/終了が Fix C(直前STABLE)/Fix B(次の自分STABLE)と一致すること。"""
    cache = _make_video_cache()
    events_df = _make_causal_events_df()
    windows = build_event_activity_windows(events_df, cache)
    by_side = {w.fire_side: w for w in windows}
    assert by_side["1P"].ignition_sec == pytest.approx(2.0)
    assert by_side["1P"].window_end_sec == pytest.approx(4.0)
    assert by_side["2P"].ignition_sec == pytest.approx(1.5)
    assert by_side["2P"].window_end_sec == pytest.approx(3.4)


def test_build_event_activity_windows_populates_fire_chain_count_from_approx_fire_chains():
    """approx_fire_chains 列があれば fire_chain_count に反映される (欠陥G2)。"""
    cache = _make_video_cache()
    events_df = _make_causal_events_df()
    events_df["approx_fire_chains"] = [8.0, 6.0]
    windows = build_event_activity_windows(events_df, cache)
    by_side = {w.fire_side: w for w in windows}
    assert by_side["1P"].fire_chain_count == pytest.approx(8.0)
    assert by_side["2P"].fire_chain_count == pytest.approx(6.0)


def test_build_event_activity_windows_defaults_fire_chain_count_when_column_missing_or_nan():
    """列が無い場合・NaNの場合は 0.0 既定値になる (後方互換、旧CSVでも動く)。"""
    cache = _make_video_cache()
    events_df_no_col = _make_causal_events_df()
    windows_no_col = build_event_activity_windows(events_df_no_col, cache)
    assert all(w.fire_chain_count == pytest.approx(0.0) for w in windows_no_col)

    events_df_with_nan = _make_causal_events_df()
    events_df_with_nan["approx_fire_chains"] = [np.nan, 6.0]
    windows_with_nan = build_event_activity_windows(events_df_with_nan, cache)
    by_side = {w.fire_side: w for w in windows_with_nan}
    assert by_side["1P"].fire_chain_count == pytest.approx(0.0)
    assert by_side["2P"].fire_chain_count == pytest.approx(6.0)


def test_is_airborne_at_true_when_moment_inside_ignition_to_completion():
    cache = _make_video_cache()
    windows = build_event_activity_windows(_make_causal_events_df(), cache)
    w2p = next(w for w in windows if w.fire_side == "2P")  # ignition=1.5, t_sec=2.5
    assert _is_airborne_at(w2p, 2.0) is True   # 1Pの点火時刻(2.0)は2Pの飛行中


def test_is_airborne_at_false_when_moment_before_ignition_or_after_completion():
    cache = _make_video_cache()
    windows = build_event_activity_windows(_make_causal_events_df(), cache)
    w1p = next(w for w in windows if w.fire_side == "1P")  # ignition=2.0, t_sec=3.0
    assert _is_airborne_at(w1p, 1.5) is False  # 2Pの点火時刻(1.5)はまだ1P点火前
    assert _is_airborne_at(w1p, 3.0) is False  # 終端は排他的 (完了後は空中でない)


def test_find_mutual_exchange_partner_one_directional_causality():
    """1Pの点火時刻には2Pが空中(партнер成立)、2Pの点火時刻には1Pは未点火(不成立)。"""
    cache = _make_video_cache()
    windows = build_event_activity_windows(_make_causal_events_df(), cache)
    target_1p = next(w for w in windows if w.fire_side == "1P")
    target_2p = next(w for w in windows if w.fire_side == "2P")

    partner_for_1p = find_mutual_exchange_partner(target_1p, windows)
    assert partner_for_1p is not None
    assert partner_for_1p.fire_side == "2P"

    partner_for_2p = find_mutual_exchange_partner(target_2p, windows)
    assert partner_for_2p is None  # 欠陥F: 未来の反撃を先取りしない


def test_find_mutual_exchange_partner_returns_none_when_causally_unrelated():
    cache = _make_video_cache()
    windows = build_event_activity_windows(_make_non_overlapping_events_df(), cache)
    target = next(w for w in windows if w.fire_side == "1P")
    assert find_mutual_exchange_partner(target, windows) is None


def test_find_mutual_exchange_partner_ignores_same_side():
    """fire_side が同じイベントは相打ち相手にならない (自分自身との誤検出防止)。"""
    a = EventActivityWindow(row_index=0, fire_side="1P", t_sec=3.0,
                            ignition_sec=2.0, window_end_sec=4.0, net_ojama_after=100.0)
    b = EventActivityWindow(row_index=1, fire_side="1P", t_sec=3.5,
                            ignition_sec=2.5, window_end_sec=4.5, net_ojama_after=50.0)
    assert find_mutual_exchange_partner(a, [a, b]) is None


def test_apply_mutual_exchange_adjustment_nets_only_the_causally_later_igniter():
    """指摘 欠陥F の受け入れテスト: 相殺されるのは「相手が既に空中だった側」
    (1P、point_1p=100-30=+70) のみで、先に点火した側(2P)は相殺されない
    (未来の反撃の先取り禁止、モデル予測値のまま)。
    """
    cache = _make_video_cache()
    events_df = _make_causal_events_df(net_1p=100.0, net_2p=30.0)
    events_df["stack_net_ojama_after_pred"] = [999.0, 888.0]  # 上書き対象(モデル予測の代わり)
    out = apply_mutual_exchange_adjustment(events_df, cache)

    row_1p = out.loc[out["fire_side"] == "1P"].iloc[0]
    row_2p = out.loc[out["fire_side"] == "2P"].iloc[0]
    assert bool(row_1p["is_mutual_exchange"]) is True
    assert row_1p["stack_net_ojama_after_pred"] == pytest.approx(70.0)
    assert row_1p["mutual_partner_t_sec"] == pytest.approx(2.5)

    assert bool(row_2p["is_mutual_exchange"]) is False
    assert row_2p["stack_net_ojama_after_pred"] == pytest.approx(888.0)  # 元の予測値のまま
    assert pd.isna(row_2p["mutual_partner_t_sec"])


def test_apply_mutual_exchange_adjustment_leaves_non_overlapping_rows_unchanged():
    """因果的に絡まない行は既存の予測値のまま (後方互換)。"""
    cache = _make_video_cache()
    events_df = _make_non_overlapping_events_df()
    events_df["stack_net_ojama_after_pred"] = [999.0, 888.0]
    out = apply_mutual_exchange_adjustment(events_df, cache)
    assert (~out["is_mutual_exchange"]).all()
    assert out["stack_net_ojama_after_pred"].tolist() == [999.0, 888.0]
    assert out["mutual_partner_t_sec"].isna().all()


# =============================================================================
# 2026-08-03 指摘 欠陥G→欠陥G改: 予告台帳の相殺会計 + 受け切れ判定
# =============================================================================

def _make_activity_window(
    fire_side: str, ignition_sec: float, t_sec: float, net_ojama_after: float,
    row_index: int = 0, receiver_baseline_ojama: float = 0.0, fire_chain_count: float = 0.0,
) -> EventActivityWindow:
    return EventActivityWindow(
        row_index=row_index, fire_side=fire_side, t_sec=t_sec, ignition_sec=ignition_sec,
        window_end_sec=t_sec + 100.0,  # 本テストでは Fix G は t_sec/ignition のみ見るため任意値
        net_ojama_after=net_ojama_after, receiver_baseline_ojama=receiver_baseline_ojama,
        fire_chain_count=fire_chain_count,
    )


class TestAggregateKnownPendingNetOjama:
    """予告台帳の集計 (空中/着弾待ち/着弾済み控除/因果整合)。"""

    def test_both_sides_in_flight_are_summed_separately(self) -> None:
        """t < t_sec (未生成・空中) は全額をそのまま計上する。"""
        empty1, empty2 = Board(), Board()
        windows = [
            _make_activity_window("1P", ignition_sec=1.0, t_sec=5.0, net_ojama_after=40.0, row_index=0),
            _make_activity_window("2P", ignition_sec=2.0, t_sec=6.0, net_ojama_after=15.0, row_index=1),
        ]
        attack_1p, attack_2p, _c1, _c2 = _aggregate_known_pending_net_ojama(windows, 3.0, empty1, empty2)
        assert attack_1p == pytest.approx(40.0)
        assert attack_2p == pytest.approx(15.0)

    def test_only_one_side_known(self) -> None:
        empty1, empty2 = Board(), Board()
        windows = [
            _make_activity_window("1P", ignition_sec=1.0, t_sec=5.0, net_ojama_after=40.0, row_index=0),
            _make_activity_window("2P", ignition_sec=10.0, t_sec=12.0, net_ojama_after=15.0, row_index=1),
        ]
        attack_1p, attack_2p, _c1, _c2 = _aggregate_known_pending_net_ojama(windows, 3.0, empty1, empty2)
        assert attack_1p == pytest.approx(40.0)
        assert attack_2p == pytest.approx(0.0)

    def test_no_events_known_gives_zero(self) -> None:
        empty1, empty2 = Board(), Board()
        windows = [_make_activity_window("1P", ignition_sec=1.0, t_sec=5.0, net_ojama_after=40.0)]
        attack_1p, attack_2p, _c1, _c2 = _aggregate_known_pending_net_ojama(windows, 0.5, empty1, empty2)
        assert (attack_1p, attack_2p) == (0.0, 0.0)

    def test_respects_time_causality_before_ignition(self) -> None:
        """点火前 (t < ignition_sec) は勘定に入れない (Fix F の因果を維持)。"""
        empty1, empty2 = Board(), Board()
        windows = [_make_activity_window("1P", ignition_sec=5.0, t_sec=8.0, net_ojama_after=40.0)]
        attack_1p, _c2, _ch1, _ch2 = _aggregate_known_pending_net_ojama(windows, 4.9, empty1, empty2)
        assert attack_1p == pytest.approx(0.0)

    def test_completed_event_still_counts_as_pending_if_not_yet_landed(self) -> None:
        """指摘 欠陥G改の核心: t_sec到達後(着弾待ち)でも、受け手盤面のおじゃまが
        まだ増えていなければ (=まだ着弾していない) 全額を予告台帳に残す
        (旧実装はここを0にしていたバグ)。
        """
        empty1, empty2 = Board(), Board()  # 1Pのおじゃまは基準値と同じ(未着弾)
        windows = [_make_activity_window(
            "2P", ignition_sec=0.0, t_sec=1.0, net_ojama_after=349.0, receiver_baseline_ojama=0.0)]
        # t=3.0 は t_sec(1.0) を過ぎている(着弾待ち) が、1Pの盤面おじゃま数(0)は
        # 基準値(0)から変化していない = まだ着弾していない
        _attack_1p, attack_2p, _c1, _c2 = _aggregate_known_pending_net_ojama(windows, 3.0, empty1, empty2)
        assert attack_2p == pytest.approx(349.0)

    def test_landed_amount_is_deducted_from_pending(self) -> None:
        """受け手盤面のおじゃまが基準値より増えていれば、その分だけ控除する
        (二重計上防止、E-2のライブ評価と整合させる)。
        """
        receiver_board = Board.from_list(
            [[0] * 6 for _ in range(12)] + [[9, 9, 9, 0, 0, 0]])  # おじゃま3個
        windows = [_make_activity_window(
            "2P", ignition_sec=0.0, t_sec=1.0, net_ojama_after=349.0, receiver_baseline_ojama=0.0)]
        _attack_1p, attack_2p, _c1, _c2 = _aggregate_known_pending_net_ojama(
            windows, 3.0, receiver_board, Board())
        assert attack_2p == pytest.approx(349.0 - 3.0)

    def test_main_worked_example_match02_2988s(self) -> None:
        """main実測 match_02 (2988s) の数値例をユニットレベルで再現する。

        2Pの349は送付済み・未着弾 (t_sec=2982.07到達済みだが1P盤面は未変化)、
        1Pの317は飛行中 (t_sec=2992.93未到達)。両方とも t=2988 で「点火済み」
        として台帳に乗ることを確認する (相殺自体は別関数で検証)。
        """
        board_1p, board_2p = Board(), Board()  # 単純化: おじゃま増分なし
        windows = [
            _make_activity_window("2P", ignition_sec=2975.0, t_sec=2982.07, net_ojama_after=349.0),
            _make_activity_window("1P", ignition_sec=2987.13, t_sec=2992.93, net_ojama_after=317.0),
        ]
        attack_1p, attack_2p, _c1, _c2 = _aggregate_known_pending_net_ojama(
            windows, 2988.0, board_1p, board_2p)
        assert attack_1p == pytest.approx(317.0)  # 1P発火分(飛行中、全額)
        assert attack_2p == pytest.approx(349.0)  # 2P発火分(着弾待ち、未着弾なので全額)
        # 相殺すると 1P に 32 残り、2P への正味送付は 0 になるはず (次のクラスで検証)。
        pending_1p, pending_2p = _net_pending_after_cancellation(attack_1p, attack_2p)
        assert pending_1p == pytest.approx(32.0)
        assert pending_2p == pytest.approx(0.0)


class TestNetPendingAfterCancellation:
    """相殺会計 (cancel_own_pending_then_send_surplus の再利用) の3方向。"""

    def test_1p_attack_larger_leaves_surplus_on_2p(self) -> None:
        pending_1p, pending_2p = _net_pending_after_cancellation(100.0, 30.0)
        assert pending_1p == pytest.approx(0.0)
        assert pending_2p == pytest.approx(70.0)

    def test_2p_attack_larger_leaves_surplus_on_1p(self) -> None:
        pending_1p, pending_2p = _net_pending_after_cancellation(30.0, 100.0)
        assert pending_1p == pytest.approx(70.0)
        assert pending_2p == pytest.approx(0.0)

    def test_equal_attacks_cancel_completely(self) -> None:
        pending_1p, pending_2p = _net_pending_after_cancellation(50.0, 50.0)
        assert (pending_1p, pending_2p) == (0.0, 0.0)


class TestLethalReadoutClamp:
    """受け切れ判定 (容量超/内、空中なし)。"""

    def test_no_airborne_events_returns_model_value_unchanged(self) -> None:
        empty1, empty2 = Board(), Board()
        sim = ChainSimulator()
        result = _lethal_readout_clamp([], 3.0, empty1, empty2, model_winprob_1p=62.0, simulator=sim)
        assert result == pytest.approx(62.0)

    def test_pending_within_capacity_does_not_clamp(self) -> None:
        """空盤面 (room=72) に対し pending=50 は容量内 -> 通常評価のまま。"""
        empty1, empty2 = Board(), Board()
        sim = ChainSimulator()
        windows = [_make_activity_window("2P", ignition_sec=0.0, t_sec=5.0, net_ojama_after=50.0)]
        result = _lethal_readout_clamp(windows, 2.0, empty1, empty2, model_winprob_1p=55.0, simulator=sim)
        assert result == pytest.approx(55.0)

    def test_pending_exceeds_capacity_clamps_toward_survivor_1p_favor(self) -> None:
        """2P発火の pending=200 が1Pの空盤面容量(72)を大幅超過 -> 2P有利にクランプしない
        (1Pが脅威を受ける側なので2P有利 = 1P視点は低いクランプになるはず)。
        """
        empty1, empty2 = Board(), Board()
        sim = ChainSimulator()
        windows = [_make_activity_window("2P", ignition_sec=0.0, t_sec=5.0, net_ojama_after=200.0)]
        result = _lethal_readout_clamp(windows, 2.0, empty1, empty2, model_winprob_1p=55.0, simulator=sim)
        assert result == pytest.approx(100.0 - LETHAL_CLAMP_FAVOR_PCT)

    def test_pending_exceeds_capacity_clamps_toward_1p_when_2p_threatened(self) -> None:
        """1P発火の pending=200 が2Pの空盤面容量を大幅超過 -> 1P有利に強くクランプ。"""
        empty1, empty2 = Board(), Board()
        sim = ChainSimulator()
        windows = [_make_activity_window("1P", ignition_sec=0.0, t_sec=5.0, net_ojama_after=200.0)]
        result = _lethal_readout_clamp(windows, 2.0, empty1, empty2, model_winprob_1p=45.0, simulator=sim)
        assert result == pytest.approx(LETHAL_CLAMP_FAVOR_PCT)

    def test_clamp_never_weakens_an_already_more_extreme_model_output(self) -> None:
        """モデル出力がクランプ値より既に極端な場合はそちらを維持する (max/minの意図)。"""
        empty1, empty2 = Board(), Board()
        sim = ChainSimulator()
        windows = [_make_activity_window("1P", ignition_sec=0.0, t_sec=5.0, net_ojama_after=200.0)]
        result = _lethal_readout_clamp(windows, 2.0, empty1, empty2, model_winprob_1p=99.0, simulator=sim)
        assert result == pytest.approx(99.0)


def _seed_board_with_small_counter() -> Board:
    """4連結1つ+未参加色ぷよ少量 (欠陥G2テスト用、expected_fire_power>0を保証)。

    tests/test_indicators_v2.py の _near_future_seed_board と同じ構図
    (盤面構築のみ、ロジックは再実装しない)。この盤面は k_hands=1..4 いずれでも
    raw=1.0 (お邪魔換算) の返し能力を持つ (2026-08-03 実測)。
    """
    g = [[0] * 6 for _ in range(BOARD_ROWS)]
    g[12][0] = COLOR_RED
    g[12][1] = COLOR_RED
    g[11][0] = COLOR_RED
    g[11][1] = COLOR_RED
    g[12][3] = COLOR_BLUE
    g[12][4] = COLOR_BLUE
    return Board.from_list(g)


class TestRealizableCounterOjama:
    """受け側の構築済み返し能力の計算 (欠陥G2、#24 sim部品の再利用確認)。"""

    def test_empty_board_has_zero_counter_regardless_of_chain_count(self) -> None:
        empty = Board()
        for chain in (0.0, 4.0, 8.0, 13.0):
            assert _realizable_counter_ojama(empty, chain) == pytest.approx(0.0)

    def test_board_with_built_groups_has_positive_counter(self) -> None:
        seed = _seed_board_with_small_counter()
        assert _realizable_counter_ojama(seed, attacker_chain_count=6.0) > 0.0

    def test_unknown_chain_count_default_still_returns_valid_hands(self) -> None:
        """fire_chain_count=0.0 (旧データ・列欠損時の既定値) でも例外にならない
        (estimate_available_hands(0)>=1 が保証、後方互換)。
        """
        seed = _seed_board_with_small_counter()
        result = _realizable_counter_ojama(seed, attacker_chain_count=0.0)
        assert result == result  # NaN でない


class TestLethalReadoutClampCounterDeduction:
    """受け切れ判定への返し控除の反映 (欠陥G2、P'>room / 0<P'<=room / P'<=0 の3方向)。"""

    def test_p_prime_exceeds_room_still_clamps(self) -> None:
        """返しを引いても十分大きい (P'>room) 場合はクランプが維持される。"""
        seed_defender = _seed_board_with_small_counter()  # room=66, 返し能力=1.0
        empty_attacker = Board()
        sim = ChainSimulator()
        windows = [_make_activity_window("2P", ignition_sec=0.0, t_sec=5.0, net_ojama_after=200.0)]
        result = _lethal_readout_clamp(
            windows, 2.0, seed_defender, empty_attacker, model_winprob_1p=55.0, simulator=sim)
        assert result == pytest.approx(100.0 - LETHAL_CLAMP_FAVOR_PCT)

    def test_p_prime_within_room_does_not_clamp_even_though_raw_p_exceeds_room(self) -> None:
        """match_04 (main実測) の核心: 返し控除前は room を超えていても
        (P=67 > room=66)、構築済み返し (1.0) を引くと P'=66<=room となり
        クランプしない (main実測「構築済み連鎖を無視した誤クランプ」の再現+修正確認)。
        """
        seed_defender = _seed_board_with_small_counter()  # room=66, 返し能力=1.0
        empty_attacker = Board()
        sim = ChainSimulator()
        windows = [_make_activity_window("2P", ignition_sec=0.0, t_sec=5.0, net_ojama_after=67.0)]
        result = _lethal_readout_clamp(
            windows, 2.0, seed_defender, empty_attacker, model_winprob_1p=55.0, simulator=sim)
        assert result == pytest.approx(55.0)  # 通常評価のまま (クランプなし)

    def test_p_prime_non_positive_short_circuits_before_room_check(self) -> None:
        """返しだけで完全に相殺できる (P<=返し能力) 場合は room 比較すら行わず
        通常評価のまま (P'<=0 の早期リターン経路)。
        """
        seed_defender = _seed_board_with_small_counter()
        empty_attacker = Board()
        sim = ChainSimulator()
        windows = [_make_activity_window("2P", ignition_sec=0.0, t_sec=5.0, net_ojama_after=1.0)]
        result = _lethal_readout_clamp(
            windows, 2.0, seed_defender, empty_attacker, model_winprob_1p=55.0, simulator=sim)
        assert result == pytest.approx(55.0)
