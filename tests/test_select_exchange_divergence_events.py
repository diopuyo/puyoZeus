"""#24 打ち合い計測器 Step5 (scripts/select_exchange_divergence_events.py) の単体テスト。

合成データによる軽量テストで以下3観点+αを担保する:
    1. 層別 (select_primary_events が位相ごとに正しい件数・正しい行を選ぶか)
    2. 3動画制約 (select_allowed_videos の集中度スコアリング + skip件数ログ)
    3. 乖離度計算 (add_rank_percentiles のパーセンタイル順位差)

本走行 (実データ・実学習) は行わない (attach_stacking_predictions の重い学習経路は
既存 tests/test_run_exchange_triple_comparison.py 側で別途カバー済みのため、
本ファイルでは1件だけ軽量スモークを追加する)。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.select_exchange_divergence_events import (
    EVENT_KEYS,
    MODEL_D_PRED_COL,
    STACK_PRED_COL,
    N_PRIMARY_PER_PHASE,
    add_rank_percentiles,
    attach_stacking_predictions,
    build_output_table,
    count_skipped_for_video_constraint,
    judge_closer_to_actual,
    score_videos_by_divergence_concentration,
    select_allowed_videos,
    select_primary_events,
    select_secondary_events,
)
from scripts.run_exchange_triple_comparison import SIM_SCORE_COL

INDICATOR_BASES = ["current_max_chain", "board_ojama_count"]


def _make_merged_df(n: int = 90, n_videos: int = 9, seed: int = 11) -> pd.DataFrame:
    """align_aug_with_model_d 済みの想定スキーマの合成 DataFrame を作る。"""
    rng = np.random.default_rng(seed)
    data: dict[str, np.ndarray] = {
        "video_id": rng.choice([f"video_t{i}" for i in range(n_videos)], size=n),
        "game_idx": rng.integers(0, 3, size=n),
        "t_sec": np.round(rng.uniform(0, 600, size=n), 3),
        "fire_side": rng.choice(["1P", "2P"], size=n),
        "phase": rng.choice(["序", "中", "終"], size=n, p=[0.3, 0.3, 0.4]),
    }
    for prefix in ("fire_", "opp_", "diff_"):
        for base in INDICATOR_BASES:
            data[f"{prefix}{base}"] = rng.normal(size=n)
    data["approx_fire_chains"] = rng.integers(0, 10, size=n).astype(float)
    data["taiou_success"] = rng.integers(0, 2, size=n)
    data["survived"] = rng.integers(0, 2, size=n)
    data["net_ojama_after"] = rng.normal(loc=50.0, scale=30.0, size=n)
    data[MODEL_D_PRED_COL] = rng.normal(loc=50.0, scale=30.0, size=n)
    data["sim_k_hands"] = rng.integers(1, 5, size=n).astype(float)
    data["sim_expected_counter_ojama"] = rng.normal(loc=20.0, scale=10.0, size=n)
    data[SIM_SCORE_COL] = rng.uniform(0.0, 1.0, size=n)
    return pd.DataFrame(data)


# =============================================================================
# 1. 乖離度計算
# =============================================================================

class TestAddRankPercentiles:
    def test_divergence_equals_manual_rank_diff(self) -> None:
        df = _make_merged_df(n=60, n_videos=6)
        out = add_rank_percentiles(df)
        expected = (df[MODEL_D_PRED_COL].rank(pct=True) - df[SIM_SCORE_COL].rank(pct=True)).abs()
        assert np.allclose(out["divergence_rank_pct"].values, expected.values)

    def test_divergence_is_zero_when_ranks_identical(self) -> None:
        n = 20
        df = pd.DataFrame({
            MODEL_D_PRED_COL: np.arange(n, dtype=float),
            SIM_SCORE_COL: np.arange(n, dtype=float) * 2.0,  # 順位は同一 (単調増加)
            "net_ojama_after": np.arange(n, dtype=float),
        })
        out = add_rank_percentiles(df)
        assert np.allclose(out["divergence_rank_pct"].values, 0.0)


# =============================================================================
# 2. 3動画制約
# =============================================================================

class TestVideoConstraint:
    def test_score_videos_ranks_concentrated_video_first(self) -> None:
        # video_hot は乖離度が大きい行を多数持つ、video_cold はほぼ0。
        df = pd.DataFrame({
            "video_id": ["video_hot"] * 10 + ["video_cold"] * 10,
            "divergence_rank_pct": [0.9] * 10 + [0.01] * 10,
        })
        scores = score_videos_by_divergence_concentration(df, top_n=10)
        assert scores.index[0] == "video_hot"

    def test_select_allowed_videos_caps_at_max(self) -> None:
        df = pd.DataFrame({
            "video_id": [f"video_v{i}" for i in range(5) for _ in range(4)],
            "divergence_rank_pct": [0.9 - 0.1 * i for i in range(5) for _ in range(4)],
        })
        allowed = select_allowed_videos(df, max_videos=3)
        assert len(allowed) == 3
        assert allowed[0] == "video_v0"  # 乖離度最大の動画が最優先

    def test_count_skipped_for_video_constraint_counts_excluded_rows(self) -> None:
        df_sorted = pd.DataFrame({"video_id": ["v_out", "v_out", "v_in", "v_in", "v_in"]})
        skipped = count_skipped_for_video_constraint(df_sorted, allowed_videos=["v_in"], n_needed=2)
        assert skipped == 2  # 先頭2件 (v_out) を飛ばして v_in を2件確保

    def test_count_skipped_zero_when_top_rows_all_allowed(self) -> None:
        df_sorted = pd.DataFrame({"video_id": ["v_in", "v_in", "v_out"]})
        skipped = count_skipped_for_video_constraint(df_sorted, allowed_videos=["v_in"], n_needed=2)
        assert skipped == 0


# =============================================================================
# 3. 層別選定 (位相ごとの主系列選定)
# =============================================================================

class TestSelectPrimaryEvents:
    def test_selects_exactly_n_per_phase_from_allowed_videos(self) -> None:
        df = _make_merged_df(n=200, n_videos=10)
        df = add_rank_percentiles(df)
        allowed = select_allowed_videos(df, max_videos=3)
        primary, skipped = select_primary_events(df, allowed, n_per_phase=N_PRIMARY_PER_PHASE)
        for phase in ("序", "中", "終"):
            n_selected = int((primary["phase"] == phase).sum())
            assert n_selected <= N_PRIMARY_PER_PHASE
            assert phase in skipped
        assert set(primary["video_id"].unique()) <= set(allowed)

    def test_selects_highest_divergence_rows_within_phase(self) -> None:
        df = pd.DataFrame({
            "video_id": ["v0"] * 6,
            "phase": ["序"] * 6,
            "divergence_rank_pct": [0.9, 0.1, 0.8, 0.2, 0.7, 0.05],
            "t_sec": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
        })
        primary, _skipped = select_primary_events(df, allowed_videos=["v0"], n_per_phase=3)
        assert sorted(primary["divergence_rank_pct"].tolist(), reverse=True) == [0.9, 0.8, 0.7]

    def test_logs_shortfall_when_phase_has_too_few_events(self) -> None:
        df = pd.DataFrame({
            "video_id": ["v0", "v0"],
            "phase": ["序", "序"],
            "divergence_rank_pct": [0.9, 0.5],
        })
        primary, skipped = select_primary_events(df, allowed_videos=["v0"], n_per_phase=4)
        assert len(primary) == 2  # 4件要求だが2件しか無い
        assert skipped["序"] == 0  # 不足は「制約で飛ばした」件数ではなく単純な母数不足


# =============================================================================
# 4. 副系列選定
# =============================================================================

class TestSelectSecondaryEvents:
    def test_excludes_primary_duplicates(self) -> None:
        df = pd.DataFrame({
            "video_id": ["v0", "v0", "v0"],
            "game_idx": [0, 0, 0],
            "t_sec": [1.0, 2.0, 3.0],
            "fire_side": ["1P", "1P", "1P"],
            "rank_pct_actual": [0.1, 0.5, 0.9],
            STACK_PRED_COL: [0.9, 0.5, 0.1],  # 残差はどの行も同程度 (|0.8|)
        })
        # (v0, 0, 1.0, 1P) を主系列で既に選定済みとして除外する。
        exclude_keys = {("v0", 0, 1.0, "1P")}
        secondary, skipped = select_secondary_events(
            df, allowed_videos=["v0"], exclude_keys=exclude_keys, n_secondary=1,
        )
        assert len(secondary) == 1
        assert secondary.iloc[0]["t_sec"] != 1.0

    def test_respects_video_allowlist(self) -> None:
        df = pd.DataFrame({
            "video_id": ["v_out", "v_in"],
            "game_idx": [0, 0],
            "t_sec": [1.0, 2.0],
            "fire_side": ["1P", "1P"],
            "rank_pct_actual": [0.1, 0.1],
            STACK_PRED_COL: [0.99, 0.9],  # v_outの方が残差大きいが動画制約外
        })
        secondary, skipped = select_secondary_events(
            df, allowed_videos=["v_in"], exclude_keys=set(), n_secondary=1,
        )
        assert list(secondary["video_id"]) == ["v_in"]
        assert skipped == 1


# =============================================================================
# 5. 機械判定 (順位ベース)
# =============================================================================

class TestJudgeCloserToActual:
    def test_model_d_wins_when_closer_in_rank(self) -> None:
        df = pd.DataFrame({
            "rank_pct_model_d": [0.52], "rank_pct_sim": [0.10], "rank_pct_actual": [0.50],
        })
        assert judge_closer_to_actual(df)[0] == "案D"

    def test_sim_wins_when_closer_in_rank(self) -> None:
        df = pd.DataFrame({
            "rank_pct_model_d": [0.05], "rank_pct_sim": [0.48], "rank_pct_actual": [0.50],
        })
        assert judge_closer_to_actual(df)[0] == "修正シミュ"


# =============================================================================
# 6. 出力テーブル組み立て + スタッキング再学習スモーク
# =============================================================================

class TestBuildOutputTable:
    def test_output_has_expected_columns_and_series_labels(self) -> None:
        df = _make_merged_df(n=60, n_videos=6)
        df = add_rank_percentiles(df)
        allowed = select_allowed_videos(df, max_videos=3)
        primary, _ = select_primary_events(df, allowed)
        df[STACK_PRED_COL] = df[MODEL_D_PRED_COL]  # スモーク用ダミー
        exclude_keys = set(primary[list(EVENT_KEYS)].apply(tuple, axis=1))
        secondary, _ = select_secondary_events(df, allowed, exclude_keys)
        output = build_output_table(primary, secondary)
        assert set(output["selection_series"].unique()) <= {"主系列", "副系列"}
        assert "closer_to_actual_rank_based" in output.columns


class TestAttachStackingPredictionsSmoke:
    """build_stacking_oof_predictions への配線が壊れていないかの軽量スモーク

    (詳細な特徴量構成テストは tests/test_run_exchange_triple_comparison.py 側で
    既にカバー済みのため、ここでは配線確認のみ)。
    """

    def test_attach_stacking_predictions_adds_finite_column(self) -> None:
        df = _make_merged_df(n=90, n_videos=9)
        out = attach_stacking_predictions(df, n_folds=3)
        assert STACK_PRED_COL in out.columns
        assert len(out) == len(df)
        assert not out[STACK_PRED_COL].isna().any()
