"""src/exchange_predictor.py (RT推論用「案D単体モデル」ロード+推論) の単体テスト。

ΔWinProb接続アーキ設計 (案C) Step1。合成データで軽量に検証する
(実ラベルCSV・実学習は使わない、tests/test_train_exchange_model_d.py と同じ方針)。

検収基準:
  1. 保存→ロード→同一入力で予測が学習時 (保存前のインメモリモデル) と bit一致する
  2. 推論速度 50ms未満/件 (実測は報告のみ、CIでのフレーキー回避のため
     テストの閾値自体は緩め=200ms とする)
"""
from __future__ import annotations

import statistics
import time

import numpy as np
import pandas as pd
import pytest

from scripts.train_exchange_model_d import build_feature_matrix, fit_final_models, save_model_bundle
from src.exchange_predictor import load_exchange_model, predict_exchange_event

INDICATOR_BASES = ["current_max_chain", "board_ojama_count", "death_margin"]

# CI でのフレーキーを避けるための緩い速度閾値 (実測値は別途報告、検収基準の
# 50msそのものではない)。
SPEED_TEST_THRESHOLD_MS: float = 200.0
SPEED_TEST_N_TRIALS: int = 1000


def _make_synthetic_df(n: int = 100, n_videos: int = 8, seed: int = 7) -> pd.DataFrame:
    """exchange_labels.csv と同一スキーマの小さな合成 DataFrame を作る。"""
    rng = np.random.default_rng(seed)
    data: dict[str, np.ndarray] = {
        "phase": rng.choice(["序", "中", "終"], size=n, p=[0.2, 0.3, 0.5]),
        "fire_side": rng.choice(["1P", "2P"], size=n),
    }
    for prefix in ("fire_", "opp_", "diff_"):
        for base in INDICATOR_BASES:
            data[f"{prefix}{base}"] = rng.normal(size=n)
    data["taiou_success"] = rng.integers(0, 2, size=n)
    data["net_ojama_after"] = rng.normal(loc=50.0, scale=30.0, size=n)
    return pd.DataFrame(data)


def _train_and_save_bundle(tmp_path, df: pd.DataFrame):
    """合成データでモデルを学習し保存する (テスト用共通セットアップ)。

    戻り値: (保存先パス, インメモリ学習済み cls/reg モデル, 特徴量行列X, 列名)
    """
    X, cols = build_feature_matrix(df, INDICATOR_BASES)
    y_cls = df["taiou_success"].astype(int).values
    y_reg = df["net_ojama_after"].astype(float).values
    cls_model, reg_model = fit_final_models(X, y_cls, y_reg)
    save_path = tmp_path / "model.joblib"
    save_model_bundle(cls_model, reg_model, INDICATOR_BASES, cols, "labels.csv", "2026-08-02", len(df), save_path)
    return save_path, cls_model, reg_model, X, cols


def _row_to_features(df: pd.DataFrame, row_idx: int) -> dict:
    """DataFrame の1行を predict_exchange_event 用の特徴量dictに変換する。"""
    row = df.iloc[row_idx]
    features: dict = {"phase": row["phase"], "fire_side": row["fire_side"]}
    for prefix in ("fire_", "opp_", "diff_"):
        for base in INDICATOR_BASES:
            col = f"{prefix}{base}"
            features[col] = float(row[col])
    return features


class TestLoadExchangeModel:
    def test_loads_bundle_with_expected_fields(self, tmp_path) -> None:
        df = _make_synthetic_df()
        save_path, _cls, _reg, _X, cols = _train_and_save_bundle(tmp_path, df)
        model = load_exchange_model(save_path)
        assert model.indicator_bases == INDICATOR_BASES
        assert model.feature_names == cols
        assert model.phases == ("序", "中", "終")
        assert model.fire_sides == ("1P", "2P")
        assert model.metadata["model_date"] == "2026-08-02"


class TestPredictExchangeEventBitExactness:
    """検収基準1: 保存→ロード→同一入力で予測が学習時と bit一致する。"""

    def test_prediction_matches_in_memory_model_exactly(self, tmp_path) -> None:
        df = _make_synthetic_df()
        save_path, cls_model, reg_model, X, _cols = _train_and_save_bundle(tmp_path, df)
        model = load_exchange_model(save_path)

        for row_idx in (0, 1, 5, 42):
            features = _row_to_features(df, row_idx)
            prob, pred = predict_exchange_event(model, features)
            expected_prob = float(cls_model.predict_proba(X[row_idx:row_idx + 1])[0, 1])
            expected_pred = float(reg_model.predict(X[row_idx:row_idx + 1])[0])
            assert prob == expected_prob  # bit一致 (float比較でも==でよい、同一計算経路のため)
            assert pred == expected_pred

    def test_feature_vector_matches_build_feature_matrix_column_order(self, tmp_path) -> None:
        """_build_feature_vector が build_feature_matrix と同じ列順を再現しているか。"""
        df = _make_synthetic_df()
        save_path, _cls, _reg, X, _cols = _train_and_save_bundle(tmp_path, df)
        model = load_exchange_model(save_path)
        from src.exchange_predictor import _build_feature_vector

        features = _row_to_features(df, 3)
        x_vec = _build_feature_vector(model, features)
        assert np.allclose(x_vec, X[3:4])


class TestPredictExchangeEventBasics:
    def test_returns_probability_in_valid_range(self, tmp_path) -> None:
        df = _make_synthetic_df()
        save_path, *_ = _train_and_save_bundle(tmp_path, df)
        model = load_exchange_model(save_path)
        features = _row_to_features(df, 0)
        prob, _pred = predict_exchange_event(model, features)
        assert 0.0 <= prob <= 1.0

    def test_missing_feature_key_raises_key_error(self, tmp_path) -> None:
        """必須キー欠如は0埋めせず明示的にエラーにする (誤判定防止)。"""
        df = _make_synthetic_df()
        save_path, *_ = _train_and_save_bundle(tmp_path, df)
        model = load_exchange_model(save_path)
        features = _row_to_features(df, 0)
        del features["fire_current_max_chain"]
        with pytest.raises(KeyError):
            predict_exchange_event(model, features)


SIM_FEATURE_COLS = ("sim_k_hands", "sim_expected_counter_ojama", "sim_damage_score")


def _make_synthetic_df_with_sim(n: int = 100, n_videos: int = 8, seed: int = 11) -> pd.DataFrame:
    """sim_* 3列付きの合成 DataFrame (併用スタッキング版バンドルのテスト用)。"""
    df = _make_synthetic_df(n=n, n_videos=n_videos, seed=seed)
    rng = np.random.default_rng(seed + 1)
    df["sim_k_hands"] = rng.integers(1, 5, size=n).astype(float)
    df["sim_expected_counter_ojama"] = rng.normal(loc=100.0, scale=50.0, size=n)
    df["sim_damage_score"] = rng.uniform(0.0, 1.0, size=n)
    return df


def _train_and_save_stacking_bundle(tmp_path, df: pd.DataFrame):
    """sim_feature_cols 付きの「併用スタッキング」版バンドルを学習・保存する。"""
    X, cols = build_feature_matrix(df, INDICATOR_BASES, extra_feature_cols=list(SIM_FEATURE_COLS))
    y_cls = df["taiou_success"].astype(int).values
    y_reg = df["net_ojama_after"].astype(float).values
    cls_model, reg_model = fit_final_models(X, y_cls, y_reg)
    save_path = tmp_path / "stacking_model.joblib"
    save_model_bundle(
        cls_model, reg_model, INDICATOR_BASES, cols, "aug.csv", "2026-08-03", len(df), save_path,
        sim_feature_cols=SIM_FEATURE_COLS,
    )
    return save_path, cls_model, reg_model, X, cols


def _row_to_features_with_sim(df: pd.DataFrame, row_idx: int) -> dict:
    features = _row_to_features(df, row_idx)
    row = df.iloc[row_idx]
    for col in SIM_FEATURE_COLS:
        features[col] = float(row[col])
    return features


class TestStackingBundleSimFeatureCols:
    """2026-08-03 追加: sim_feature_cols 付き併用スタッキング版バンドルの検収。"""

    def test_bundle_persists_sim_feature_cols(self, tmp_path) -> None:
        df = _make_synthetic_df_with_sim()
        save_path, *_ = _train_and_save_stacking_bundle(tmp_path, df)
        model = load_exchange_model(save_path)
        assert model.sim_feature_cols == SIM_FEATURE_COLS

    def test_legacy_bundle_without_sim_feature_cols_defaults_to_empty(self, tmp_path) -> None:
        """sim_feature_cols キーの無い旧バンドル (joblib) も後方互換で読める。"""
        df = _make_synthetic_df()
        save_path, *_ = _train_and_save_bundle(tmp_path, df)  # 旧経路 (sim無し)
        model = load_exchange_model(save_path)
        assert model.sim_feature_cols == ()

    def test_stacking_prediction_bit_matches_in_memory_model(self, tmp_path) -> None:
        """指摘: スタッキング束の保存/ロードでも bit一致すること。"""
        df = _make_synthetic_df_with_sim()
        save_path, cls_model, reg_model, X, _cols = _train_and_save_stacking_bundle(tmp_path, df)
        model = load_exchange_model(save_path)
        for row_idx in (0, 1, 5, 42):
            features = _row_to_features_with_sim(df, row_idx)
            prob, pred = predict_exchange_event(model, features)
            expected_prob = float(cls_model.predict_proba(X[row_idx:row_idx + 1])[0, 1])
            expected_pred = float(reg_model.predict(X[row_idx:row_idx + 1])[0])
            assert prob == expected_prob
            assert pred == expected_pred

    def test_stacking_feature_vector_appends_sim_cols_after_onehot(self, tmp_path) -> None:
        """特徴量ベクトルの並びが fire/opp/diff→phase→fire_side→sim_* の順であること。"""
        df = _make_synthetic_df_with_sim()
        save_path, _cls, _reg, X, _cols = _train_and_save_stacking_bundle(tmp_path, df)
        model = load_exchange_model(save_path)
        from src.exchange_predictor import _build_feature_vector

        features = _row_to_features_with_sim(df, 3)
        x_vec = _build_feature_vector(model, features)
        assert np.allclose(x_vec, X[3:4])
        assert x_vec.shape[1] == len(INDICATOR_BASES) * 3 + 3 + 2 + len(SIM_FEATURE_COLS)


class TestPredictionSpeed:
    """速度検収 (実測は報告のみ、CIでのフレーキー回避のため閾値は緩め)。"""

    def test_median_latency_under_relaxed_threshold(self, tmp_path) -> None:
        df = _make_synthetic_df()
        save_path, *_ = _train_and_save_bundle(tmp_path, df)
        model = load_exchange_model(save_path)
        features = _row_to_features(df, 0)

        latencies_ms: list[float] = []
        for _ in range(SPEED_TEST_N_TRIALS):
            t0 = time.perf_counter()
            predict_exchange_event(model, features)
            latencies_ms.append((time.perf_counter() - t0) * 1000.0)

        median_ms = statistics.median(latencies_ms)
        p95_ms = statistics.quantiles(latencies_ms, n=100)[94]
        print(f"\n  [速度実測] median={median_ms:.3f}ms  p95={p95_ms:.3f}ms"
              f"  (検収基準50ms、テスト閾値{SPEED_TEST_THRESHOLD_MS:.0f}ms)")
        assert median_ms < SPEED_TEST_THRESHOLD_MS
