"""W2.1 WinPredictorMLP のテスト。"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.old.state_features import TOTAL_FEATURE_DIM
from src.old.win_predictor import WinPredictorMLP


def test_predictor_construction() -> None:
    model = WinPredictorMLP()
    assert model._input_dim == TOTAL_FEATURE_DIM


def test_predict_outputs_in_range() -> None:
    """predict は 0..1 の確率値を返す。"""
    model = WinPredictorMLP()
    X = np.random.randn(5, TOTAL_FEATURE_DIM).astype(np.float32)
    probs = model.predict(X)
    assert probs.shape == (5,)
    assert (probs >= 0).all() and (probs <= 1).all()


def test_predict_single_returns_scalar() -> None:
    model = WinPredictorMLP()
    x = np.random.randn(TOTAL_FEATURE_DIM).astype(np.float32)
    p = model.predict(x)
    assert isinstance(p, float)
    assert 0 <= p <= 1


def test_fit_reduces_loss() -> None:
    """訓練すると損失が減る。"""
    model = WinPredictorMLP()
    # 単純なバランスデータ
    rng = np.random.default_rng(42)
    X = rng.standard_normal((200, TOTAL_FEATURE_DIM)).astype(np.float32)
    # ラベルは入力の最初の値の符号で決定 (学習可能なシグナル)
    y = (X[:, 0] > 0).astype(np.float32)
    losses = model.fit(X, y, epochs=10, batch_size=32, verbose=False)
    assert losses[0] > losses[-1]
    # 学習後、最初の特徴が大きい例で確率が高くなる
    pos = np.zeros((1, TOTAL_FEATURE_DIM), dtype=np.float32)
    pos[0, 0] = 5.0
    neg = np.zeros((1, TOTAL_FEATURE_DIM), dtype=np.float32)
    neg[0, 0] = -5.0
    p_pos = model.predict(pos)[0]
    p_neg = model.predict(neg)[0]
    assert p_pos > p_neg


def test_save_and_load_roundtrip() -> None:
    """save → load で同じ予測が再現される。"""
    model = WinPredictorMLP()
    X = np.random.randn(3, TOTAL_FEATURE_DIM).astype(np.float32)
    probs1 = model.predict(X)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "test.pt"
        model.save(path)
        assert path.exists()
        model2 = WinPredictorMLP()
        # 初期化直後は probs1 と異なる予測 (ランダム重み)
        model2.load(path)
        probs2 = model2.predict(X)
    np.testing.assert_allclose(probs1, probs2, rtol=1e-5)
