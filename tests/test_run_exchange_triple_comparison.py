"""#24 打ち合い計測器「三つ巴比較」駆動スクリプト
(scripts/run_exchange_triple_comparison.py) の単体テスト。

合成データによる軽量テストで以下3観点+αを担保する:
    1. 突合ロジック (align_aug_with_model_d): 一致/突合失敗/重複キー検出
    2. NaN除外ロジック (filter_nan_sim_rows)
    3. スタッキング特徴量構成 (build_feature_matrix の extra_feature_cols 拡張、
       build_stacking_oof_predictions が sim_* 列を含めて学習すること)

本走行 (実データ) は行わない (入力CSVがまだ生成中のため)。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.run_exchange_triple_comparison import (
    MERGE_KEYS,
    SIM_FEATURE_COLS,
    _log_sign_diagnostics,
    align_aug_with_model_d,
    build_predictor_set,
    build_stacking_oof_predictions,
    filter_nan_sim_rows,
    load_model_d_oof,
)
from scripts.train_exchange_model_d import build_feature_matrix, get_indicator_base_names

INDICATOR_BASES = ["current_max_chain", "board_ojama_count"]


def _make_synthetic_aug_df(n: int = 90, n_videos: int = 9, seed: int = 1) -> pd.DataFrame:
    """exchange_labels_*_aug.csv と同一スキーマ (sim_* 3列付き) の合成 DataFrame。"""
    rng = np.random.default_rng(seed)
    data: dict[str, np.ndarray] = {
        "video_id": rng.choice([f"video_t{i}" for i in range(n_videos)], size=n),
        "game_idx": rng.integers(0, 3, size=n),
        "t_sec": np.round(rng.uniform(0, 600, size=n), 3),
        "fire_side": rng.choice(["1P", "2P"], size=n),
        "phase": rng.choice(["序", "中", "終"], size=n, p=[0.2, 0.3, 0.5]),
    }
    for prefix in ("fire_", "opp_", "diff_"):
        for base in INDICATOR_BASES:
            data[f"{prefix}{base}"] = rng.normal(size=n)
    data["approx_fire_chains"] = rng.integers(0, 10, size=n).astype(float)
    data["taiou_success"] = rng.integers(0, 2, size=n)
    data["net_ojama_after"] = rng.normal(loc=50.0, scale=30.0, size=n)
    data["sim_k_hands"] = rng.integers(1, 5, size=n).astype(float)
    data["sim_expected_counter_ojama"] = rng.normal(loc=20.0, scale=10.0, size=n)
    data["sim_damage_score"] = rng.uniform(0.0, 1.0, size=n)
    # 一部行は sim_damage_score 計算不能 (npz境界ケース等) を模擬して NaN にする。
    nan_rows = rng.choice(n, size=max(1, n // 15), replace=False)
    data["sim_damage_score"][nan_rows] = np.nan
    return pd.DataFrame(data)


def _make_matching_oof_df(aug_df: pd.DataFrame, seed: int = 2) -> pd.DataFrame:
    """aug_df のキー列と一致する案D OOF 予測 (合成) を作る。"""
    rng = np.random.default_rng(seed)
    n = len(aug_df)
    out = aug_df[list(MERGE_KEYS)].copy()
    out["phase"] = aug_df["phase"].values
    out["taiou_success"] = aug_df["taiou_success"].values
    out["net_ojama_after"] = aug_df["net_ojama_after"].values
    out["prob_taiou_success_oof"] = rng.uniform(0.0, 1.0, size=n)
    out["net_ojama_after_oof_pred"] = rng.normal(loc=50.0, scale=30.0, size=n)
    return out


class TestLoadModelDOof:
    """load_model_d_oof がファイル欠如を明示的に検出するか。"""

    def test_missing_file_raises(self, tmp_path) -> None:
        with pytest.raises(FileNotFoundError):
            load_model_d_oof(tmp_path / "does_not_exist")

    def test_loads_existing_csv(self, tmp_path) -> None:
        aug_df = _make_synthetic_aug_df(n=20, n_videos=4)
        oof_df = _make_matching_oof_df(aug_df)
        model_d_dir = tmp_path / "model_d"
        model_d_dir.mkdir()
        oof_df.to_csv(model_d_dir / "oof_predictions.csv", index=False)
        loaded = load_model_d_oof(model_d_dir)
        assert len(loaded) == len(oof_df)


class TestAlignAugWithModelD:
    """突合ロジック: 一致/突合失敗行数/重複キー検出。"""

    def test_all_rows_match(self) -> None:
        aug_df = _make_synthetic_aug_df(n=60, n_videos=6)
        oof_df = _make_matching_oof_df(aug_df)
        merged = align_aug_with_model_d(aug_df, oof_df)
        assert len(merged) == len(aug_df)
        assert "prob_taiou_success_oof" in merged.columns
        assert "net_ojama_after_oof_pred" in merged.columns

    def test_unmatched_rows_are_excluded(self) -> None:
        aug_df = _make_synthetic_aug_df(n=60, n_videos=6)
        oof_df = _make_matching_oof_df(aug_df)
        # oof側の一部行を削除して突合失敗を作る (aug側のみに存在するキーが発生)。
        oof_df_short = oof_df.iloc[:-5].reset_index(drop=True)
        merged = align_aug_with_model_d(aug_df, oof_df_short)
        assert len(merged) == len(aug_df) - 5

    def test_duplicate_keys_in_aug_raises(self) -> None:
        aug_df = _make_synthetic_aug_df(n=30, n_videos=4)
        # 先頭行を複製して同一複合キーの重複を作る。
        dup_row = aug_df.iloc[[0]].copy()
        aug_df_dup = pd.concat([aug_df, dup_row], ignore_index=True)
        oof_df = _make_matching_oof_df(aug_df)
        with pytest.raises(ValueError, match="複合キー"):
            align_aug_with_model_d(aug_df_dup, oof_df)

    def test_duplicate_keys_in_oof_raises(self) -> None:
        aug_df = _make_synthetic_aug_df(n=30, n_videos=4)
        oof_df = _make_matching_oof_df(aug_df)
        dup_row = oof_df.iloc[[0]].copy()
        oof_df_dup = pd.concat([oof_df, dup_row], ignore_index=True)
        with pytest.raises(ValueError, match="複合キー"):
            align_aug_with_model_d(aug_df, oof_df_dup)


class TestFilterNanSimRows:
    """sim_damage_score NaN除外ロジック。"""

    def test_excludes_only_nan_rows(self) -> None:
        aug_df = _make_synthetic_aug_df(n=90, n_videos=9)
        n_nan_expected = int(aug_df["sim_damage_score"].isna().sum())
        assert n_nan_expected > 0  # 合成データにNaNが仕込まれていることの前提確認
        filtered = filter_nan_sim_rows(aug_df)
        assert len(filtered) == len(aug_df) - n_nan_expected
        assert filtered["sim_damage_score"].notna().all()

    def test_no_exclusion_when_no_nan(self) -> None:
        df = pd.DataFrame({"sim_damage_score": [0.1, 0.2, 0.3], "x": [1, 2, 3]})
        filtered = filter_nan_sim_rows(df)
        assert len(filtered) == 3


class TestStackingFeatureComposition:
    """build_feature_matrix の extra_feature_cols 拡張 (後方互換) + スタッキング学習。"""

    def test_build_feature_matrix_backward_compat_without_extra_cols(self) -> None:
        """extra_feature_cols を渡さない既存呼び出しは列数・列名が変わらない。"""
        aug_df = _make_synthetic_aug_df(n=40, n_videos=5)
        bases = get_indicator_base_names(aug_df)
        X, cols = build_feature_matrix(aug_df, bases)
        expected_n_cols = len(bases) * 3 + 3 + 2  # triad + phase(3) + side(2)
        assert X.shape == (len(aug_df), expected_n_cols)
        assert len(cols) == expected_n_cols
        assert all(not c.startswith("sim_") for c in cols)

    def test_build_feature_matrix_with_extra_cols_appends(self) -> None:
        aug_df = _make_synthetic_aug_df(n=40, n_videos=5)
        aug_df = aug_df[aug_df["sim_damage_score"].notna()].reset_index(drop=True)
        bases = get_indicator_base_names(aug_df)
        X, cols = build_feature_matrix(aug_df, bases, extra_feature_cols=list(SIM_FEATURE_COLS))
        expected_n_cols = len(bases) * 3 + 3 + 2 + len(SIM_FEATURE_COLS)
        assert X.shape == (len(aug_df), expected_n_cols)
        for sim_col in SIM_FEATURE_COLS:
            assert sim_col in cols
        sim_col_idx = cols.index("sim_damage_score")
        assert np.allclose(X[:, sim_col_idx], aug_df["sim_damage_score"].values)

    def test_build_stacking_oof_predictions_uses_sim_columns(self) -> None:
        aug_df = _make_synthetic_aug_df(n=90, n_videos=9)
        aug_df = filter_nan_sim_rows(aug_df)
        oof_proba, oof_pred, feature_names = build_stacking_oof_predictions(aug_df, n_folds=3)
        assert len(oof_proba) == len(aug_df)
        assert len(oof_pred) == len(aug_df)
        assert not np.isnan(oof_proba).any()
        assert not np.isnan(oof_pred).any()
        for sim_col in SIM_FEATURE_COLS:
            assert sim_col in feature_names


class TestBuildPredictorSet:
    """3予測器 (案D/修正シミュ/併用) が正しい列から組み立てられるか。"""

    def test_three_predictors_built_from_correct_columns(self) -> None:
        aug_df = _make_synthetic_aug_df(n=50, n_videos=6)
        oof_df = _make_matching_oof_df(aug_df)
        merged = align_aug_with_model_d(aug_df, oof_df)
        merged = filter_nan_sim_rows(merged)
        stack_proba = np.random.default_rng(0).uniform(size=len(merged))
        stack_pred = np.random.default_rng(1).normal(size=len(merged))
        predictors = build_predictor_set(merged, stack_proba, stack_pred)
        names = [p.name for p in predictors]
        assert names == ["案D", "修正シミュ", "併用(スタッキング)"]
        pred_d = predictors[0]
        assert np.array_equal(pred_d.prob_taiou_success, merged["prob_taiou_success_oof"].values)
        assert np.array_equal(pred_d.net_ojama_after_pred, merged["net_ojama_after_oof_pred"].values)
        pred_sim = predictors[1]
        # 2026-08-02バグ修正後: sim_damage_scoreは net_ojama_after と同じ向き
        # (大きいほど攻撃側に有利) だが taiou_success (受け手成功) とは逆向きの
        # ため、prob_taiou_success には 1 - sim_damage_score (符号反転) を使う。
        assert np.array_equal(pred_sim.prob_taiou_success, 1.0 - merged["sim_damage_score"].values)
        assert np.array_equal(pred_sim.net_ojama_after_pred, merged["sim_damage_score"].values)
        pred_stack = predictors[2]
        assert np.array_equal(pred_stack.prob_taiou_success, stack_proba)
        assert np.array_equal(pred_stack.net_ojama_after_pred, stack_pred)


class TestLogSignDiagnostics:
    """符号診断ログ (2026-08-02バグ修正後の想定符号: net_ojama_afterと正相関/

    taiou_successと負相関が健全) が正しい向きで警告を出すか。
    """

    def test_healthy_signs_do_not_warn(self, capsys) -> None:
        n = 200
        rng = np.random.default_rng(3)
        sim_score = rng.uniform(0.0, 1.0, size=n)
        # net_ojama_after は sim_score と同じ向き (正相関) になるよう構成。
        net_ojama_after = sim_score * 100.0 + rng.normal(scale=1.0, size=n)
        # taiou_success は sim_score が高いほど 0 になりやすい (負相関) よう構成。
        taiou_success = (rng.uniform(size=n) > sim_score).astype(int)
        df = pd.DataFrame({
            "sim_damage_score": sim_score,
            "net_ojama_after": net_ojama_after,
            "taiou_success": taiou_success,
        })
        _log_sign_diagnostics(df)
        out = capsys.readouterr().out
        assert "符号確認が必要" not in out

    def test_inverted_signs_trigger_warning(self, capsys) -> None:
        n = 200
        rng = np.random.default_rng(4)
        sim_score = rng.uniform(0.0, 1.0, size=n)
        # 想定と逆向き: net_ojama_after と負相関、taiou_success と正相関。
        net_ojama_after = -sim_score * 100.0 + rng.normal(scale=1.0, size=n)
        taiou_success = (rng.uniform(size=n) < sim_score).astype(int)
        df = pd.DataFrame({
            "sim_damage_score": sim_score,
            "net_ojama_after": net_ojama_after,
            "taiou_success": taiou_success,
        })
        _log_sign_diagnostics(df)
        out = capsys.readouterr().out
        assert "符号確認が必要" in out
