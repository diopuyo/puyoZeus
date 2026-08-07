"""#24 打ち合い計測器「案D」学習器 (scripts/train_exchange_model_d.py) の単体テスト。

小さな合成 CSV でエンドツーエンド (特徴量構築 -> OOF学習 -> permutation
importance -> harness へのレポート出力) が壊れていないことを確認する。
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd
import pytest

from scripts.train_exchange_model_d import (
    NON_FEATURE_COLS,
    build_feature_matrix,
    fit_final_models,
    get_indicator_base_names,
    main,
    save_model_bundle,
)

INDICATOR_BASES = ["current_max_chain", "board_ojama_count"]


def _make_synthetic_df(n: int = 80, n_videos: int = 8, seed: int = 42) -> pd.DataFrame:
    """exchange_labels.csv と同一スキーマの小さな合成 DataFrame を作る。"""
    rng = np.random.default_rng(seed)
    data: dict[str, np.ndarray] = {
        "video_id": rng.choice([f"video_t{i}" for i in range(n_videos)], size=n),
        "game_idx": rng.integers(0, 3, size=n),
        "t_sec": rng.uniform(0, 600, size=n),
        "fire_side": rng.choice(["1P", "2P"], size=n),
        "phase": rng.choice(["序", "中", "終"], size=n, p=[0.2, 0.3, 0.5]),
        "won": rng.integers(0, 2, size=n).astype(float),
    }
    for prefix in ("fire_", "opp_", "diff_"):
        for base in INDICATOR_BASES:
            data[f"{prefix}{base}"] = rng.normal(size=n)
    data["net_ojama"] = rng.normal(size=n)
    data["returned"] = rng.integers(0, 2, size=n)
    data["returned_competitive"] = rng.integers(0, 2, size=n)
    data["return_window_sec"] = rng.uniform(0, 2, size=n)
    data["approx_fire_chains"] = rng.integers(0, 10, size=n).astype(float)
    data["opp_buried"] = rng.integers(0, 2, size=n)
    data["taiou_success"] = rng.integers(0, 2, size=n)
    data["survived"] = rng.integers(0, 2, size=n)
    data["net_ojama_after"] = rng.normal(loc=50.0, scale=30.0, size=n)
    return pd.DataFrame(data)


class TestIndicatorBaseDetection:
    """33列決め打ち禁止 -> 動的検出が正しく機能するか。"""

    def test_detects_only_complete_triads(self) -> None:
        df = _make_synthetic_df()
        bases = get_indicator_base_names(df)
        assert bases == sorted(INDICATOR_BASES)

    def test_excludes_non_feature_cols_even_if_prefixed(self) -> None:
        df = _make_synthetic_df()
        df["fire_side_dummy"] = "x"  # opp_side_dummy/diff_side_dummy が無いので除外されるはず
        bases = get_indicator_base_names(df)
        assert "side_dummy" not in bases

    def test_non_feature_cols_constant_has_target_cols(self) -> None:
        assert "taiou_success" in NON_FEATURE_COLS
        assert "net_ojama_after" in NON_FEATURE_COLS
        assert "won" in NON_FEATURE_COLS  # 試合全体の勝敗はリーク防止のため特徴量対象外


class TestBuildFeatureMatrix:
    """fire/opp/diff 3つ組 + phase one-hot + fire_side one-hot が正しく組まれるか。"""

    def test_shape_and_columns(self) -> None:
        df = _make_synthetic_df()
        X, cols = build_feature_matrix(df, INDICATOR_BASES)
        expected_n_cols = len(INDICATOR_BASES) * 3 + 3 + 2  # triad + phase(3) + side(2)
        assert X.shape == (len(df), expected_n_cols)
        assert len(cols) == expected_n_cols
        assert "phase_序" in cols and "phase_中" in cols and "phase_終" in cols
        assert "fire_side_1P" in cols and "fire_side_2P" in cols

    def test_phase_onehot_values(self) -> None:
        df = _make_synthetic_df()
        X, cols = build_feature_matrix(df, INDICATOR_BASES)
        phase_col_idx = cols.index("phase_序")
        expected = (df["phase"].values == "序").astype(float)
        assert np.array_equal(X[:, phase_col_idx], expected)


class TestMainSmoke:
    """main() がエンドツーエンドでエラーなく完走し、成果物を出力するか。"""

    def test_main_runs_end_to_end(self, tmp_path, monkeypatch) -> None:
        df = _make_synthetic_df(n=120, n_videos=10)
        labels_path = tmp_path / "exchange_labels_synth.csv"
        df.to_csv(labels_path, index=False)
        out_dir = tmp_path / "exchange_model_d_out"

        argv = [
            "train_exchange_model_d.py",
            "--labels", str(labels_path),
            "--out-dir", str(out_dir),
            "--n-folds", "3",
        ]
        monkeypatch.setattr(sys, "argv", argv)
        main()

        assert (out_dir / "oof_predictions.csv").exists()
        assert (out_dir / "train_val_gap_cls.csv").exists()
        assert (out_dir / "train_val_gap_reg.csv").exists()
        assert (out_dir / "permutation_importance_cls.csv").exists()
        assert (out_dir / "permutation_importance_reg.csv").exists()
        assert (out_dir / "comparison_report.md").exists()
        assert (out_dir / "reliability_diagrams.png").exists()

        oof_df = pd.read_csv(out_dir / "oof_predictions.csv")
        assert len(oof_df) == len(df)
        assert oof_df["prob_taiou_success_oof"].notna().all()
        assert oof_df["net_ojama_after_oof_pred"].notna().all()

    def test_main_without_save_model_creates_no_model_file(self, tmp_path, monkeypatch) -> None:
        """--save-model 省略時 (既定) はモデルファイルを作らない (backward compat)。"""
        df = _make_synthetic_df(n=60, n_videos=6)
        labels_path = tmp_path / "exchange_labels_synth.csv"
        df.to_csv(labels_path, index=False)
        out_dir = tmp_path / "exchange_model_d_out_nomodel"

        argv = [
            "train_exchange_model_d.py",
            "--labels", str(labels_path), "--out-dir", str(out_dir), "--n-folds", "3",
        ]
        monkeypatch.setattr(sys, "argv", argv)
        main()
        assert not any(tmp_path.rglob("*.joblib"))


class TestSaveModel:
    """--save-model オプション (RT推論用モデル永続化) の単体・統合テスト。"""

    def test_main_with_save_model_creates_joblib_bundle(self, tmp_path, monkeypatch) -> None:
        df = _make_synthetic_df(n=80, n_videos=8)
        labels_path = tmp_path / "exchange_labels_synth.csv"
        df.to_csv(labels_path, index=False)
        out_dir = tmp_path / "exchange_model_d_out"
        model_path = tmp_path / "model.joblib"

        argv = [
            "train_exchange_model_d.py",
            "--labels", str(labels_path), "--out-dir", str(out_dir), "--n-folds", "3",
            "--save-model", str(model_path), "--model-date", "2026-08-02",
        ]
        monkeypatch.setattr(sys, "argv", argv)
        main()

        assert model_path.exists()
        import joblib
        bundle = joblib.load(model_path)
        assert set(bundle.keys()) >= {
            "cls_model", "reg_model", "indicator_bases", "feature_names",
            "phases", "fire_sides", "metadata",
        }
        assert bundle["metadata"]["labels_csv"] == str(labels_path)
        assert bundle["metadata"]["model_date"] == "2026-08-02"
        assert bundle["metadata"]["n_samples"] == len(df)

    def test_fit_final_models_predicts_without_error(self) -> None:
        df = _make_synthetic_df(n=60, n_videos=6)
        bases = get_indicator_base_names(df)
        X, _cols = build_feature_matrix(df, bases)
        y_cls = df["taiou_success"].astype(int).values
        y_reg = df["net_ojama_after"].astype(float).values
        cls_model, reg_model = fit_final_models(X, y_cls, y_reg)
        assert cls_model.predict_proba(X[:1]).shape == (1, 2)
        assert reg_model.predict(X[:1]).shape == (1,)

    def test_save_model_bundle_roundtrip_via_joblib(self, tmp_path) -> None:
        df = _make_synthetic_df(n=60, n_videos=6)
        bases = get_indicator_base_names(df)
        X, cols = build_feature_matrix(df, bases)
        y_cls = df["taiou_success"].astype(int).values
        y_reg = df["net_ojama_after"].astype(float).values
        cls_model, reg_model = fit_final_models(X, y_cls, y_reg)

        save_path = tmp_path / "bundle.joblib"
        save_model_bundle(cls_model, reg_model, bases, cols, "labels.csv", "2026-08-02", len(df), save_path)

        import joblib
        bundle = joblib.load(save_path)
        assert bundle["indicator_bases"] == bases
        assert bundle["feature_names"] == cols
        assert bundle["phases"] == ("序", "中", "終")
        assert bundle["fire_sides"] == ("1P", "2P")
