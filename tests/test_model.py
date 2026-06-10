"""
model.py のテスト

ML モデルが Scorer 互換 API を提供し、合成データで収束することを検証する。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.old.indicators import (
    ALL_INDICATOR_NAMES,
    IndicatorResult,
    IndicatorSet,
)
from src.old.scorer import (
    DEFAULT_WEIGHTS,
    SCORE_RANGE_MAX,
    SCORE_RANGE_MIN,
    ScoreResult,
    Scorer,
)
from src.old.model import (
    DEFAULT_EPOCHS,
    FEATURE_DIM,
    LinearScorerModel,
    MLPScorerModel,
    ScoreSample,
    TrainingReport,
    generate_synthetic_dataset,
    indicator_vector,
)


# ============================
# ヘルパー
# ============================


def make_set(scores: dict[str, float]) -> IndicatorSet:
    results = {}
    for name in ALL_INDICATOR_NAMES:
        s = scores.get(name, 0.0)
        results[name] = IndicatorResult(name=name, score=s, raw_value=s)
    return IndicatorSet(results=results)


def zeros() -> IndicatorSet:
    return make_set({})


def ones() -> IndicatorSet:
    return make_set({n: 1.0 for n in ALL_INDICATOR_NAMES})


# ============================
# indicator_vector
# ============================


class TestIndicatorVector:
    def test_shape(self):
        v = indicator_vector(zeros())
        assert v.shape == (FEATURE_DIM,)

    def test_order_matches_all_indicator_names(self):
        scores = {n: float(i) / 10 for i, n in enumerate(ALL_INDICATOR_NAMES)}
        v = indicator_vector(make_set(scores))
        for i, name in enumerate(ALL_INDICATOR_NAMES):
            assert v[i] == pytest.approx(scores[name])

    def test_missing_indicators_zero(self):
        partial = IndicatorSet(results={})
        v = indicator_vector(partial)
        assert np.all(v == 0.0)


# ============================
# ScoreSample
# ============================


class TestScoreSample:
    def test_basic_fields(self):
        s = ScoreSample(p1=zeros(), p2=zeros(), label=10.0)
        assert s.label == 10.0


# ============================
# LinearScorerModel - predict
# ============================


class TestLinearPredict:
    def test_default_zero_weights(self):
        m = LinearScorerModel()
        r = m.predict(ones(), zeros())
        assert r.total_score == 0.0

    def test_predict_returns_score_result(self):
        m = LinearScorerModel()
        r = m.predict(ones(), zeros())
        assert isinstance(r, ScoreResult)

    def test_from_scorer_weights_matches_direction(self):
        """Scorer と同じ重みを持てば、方向性は同じ。"""
        scorer = Scorer()
        model = LinearScorerModel.from_scorer_weights()
        # 1P 側のみ本線を持つケース
        p1 = make_set({"main_chain_maturity": 1.0})
        p2 = zeros()
        rb = scorer.score(p1, p2).total_score
        rm = model.predict(p1, p2).total_score
        # 符号が一致
        assert (rb > 0) == (rm > 0)

    def test_symmetry(self):
        m = LinearScorerModel.from_scorer_weights()
        r1 = m.predict(ones(), zeros())
        r2 = m.predict(zeros(), ones())
        # 線形モデルは対称
        assert r1.total_score == pytest.approx(-r2.total_score, abs=1e-6)

    def test_score_in_range(self):
        m = LinearScorerModel(weights=np.array([100.0] * FEATURE_DIM))
        r = m.predict(ones(), zeros())
        assert SCORE_RANGE_MIN <= r.total_score <= SCORE_RANGE_MAX


# ============================
# LinearScorerModel - fit
# ============================


class TestLinearFit:
    def test_empty_samples_raises(self):
        m = LinearScorerModel()
        with pytest.raises(ValueError):
            m.fit([])

    def test_loss_decreases(self):
        # Scorer をオラクルにして合成データを生成
        samples = generate_synthetic_dataset(n_samples=128, seed=0)
        m = LinearScorerModel()
        report = m.fit(samples, epochs=50, learning_rate=0.02)
        assert isinstance(report, TrainingReport)
        assert report.loss_history[0] > report.loss_history[-1]

    def test_learned_model_approximates_oracle(self):
        """Scorer と同じ規則で生成したデータで学習すれば近似する。"""
        samples = generate_synthetic_dataset(n_samples=512, seed=1)
        m = LinearScorerModel()
        m.fit(samples, epochs=DEFAULT_EPOCHS, learning_rate=0.02)

        # テストセット (別シード) で MSE を評価
        test = generate_synthetic_dataset(n_samples=100, seed=99)
        preds = [
            m.predict(s.p1, s.p2).total_score for s in test
        ]
        labels = [s.label for s in test]
        mse = float(np.mean((np.array(preds) - np.array(labels)) ** 2))
        # 極端に大きくなければ OK (スケール 100^2 = 10000 が理論上限)
        assert mse < 500.0


# ============================
# LinearScorerModel - save/load
# ============================


class TestLinearPersistence:
    def test_save_load_roundtrip(self, tmp_path):
        w = np.array([0.1, 0.2, -0.1, 0.3, -0.5, 0.4, 0.2, 0.1])
        m = LinearScorerModel(weights=w, bias=0.5)
        path = tmp_path / "linear.json"
        m.save(path)

        loaded = LinearScorerModel.load(path)
        np.testing.assert_allclose(loaded._w, w)
        assert loaded._b == 0.5

    def test_load_wrong_format_raises(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text(json.dumps({"format": "wrong"}), encoding="utf-8")
        with pytest.raises(ValueError):
            LinearScorerModel.load(path)

    def test_predictions_consistent_after_load(self, tmp_path):
        m = LinearScorerModel.from_scorer_weights()
        path = tmp_path / "m.json"
        m.save(path)
        loaded = LinearScorerModel.load(path)

        p1 = make_set({"main_chain_maturity": 0.5, "offset_power": 0.3})
        p2 = make_set({"death_risk": 0.8})
        assert m.predict(p1, p2).total_score == pytest.approx(
            loaded.predict(p1, p2).total_score
        )


# ============================
# LinearScorerModel - weights_dict
# ============================


class TestLinearWeightsDict:
    def test_keys_match_indicator_names(self):
        m = LinearScorerModel()
        d = m.weights_dict()
        assert set(d.keys()) == set(ALL_INDICATOR_NAMES)


# ============================
# MLPScorerModel - predict
# ============================


class TestMLPPredict:
    def test_predict_returns_score_result(self):
        m = MLPScorerModel()
        r = m.predict(ones(), zeros())
        assert isinstance(r, ScoreResult)

    def test_score_in_range(self):
        m = MLPScorerModel()
        for _ in range(10):
            r = m.predict(ones(), zeros())
            assert SCORE_RANGE_MIN <= r.total_score <= SCORE_RANGE_MAX

    def test_different_inputs_give_different_predictions(self):
        m = MLPScorerModel()
        s1 = m.predict(ones(), zeros()).total_score
        s2 = m.predict(zeros(), ones()).total_score
        # 初期化された MLP でも対称入力で予測は変化するはず
        assert s1 != s2 or m.predict(zeros(), zeros()).total_score != s1


# ============================
# MLPScorerModel - fit
# ============================


class TestMLPFit:
    def test_empty_samples_raises(self):
        m = MLPScorerModel()
        with pytest.raises(ValueError):
            m.fit([])

    def test_loss_decreases(self):
        samples = generate_synthetic_dataset(n_samples=256, seed=2)
        m = MLPScorerModel(hidden_size=12)
        report = m.fit(samples, epochs=30, learning_rate=0.005)
        # 学習初期 vs 最終
        assert report.loss_history[-1] < report.loss_history[0]


# ============================
# MLPScorerModel - save/load
# ============================


class TestMLPPersistence:
    def test_save_load_roundtrip(self, tmp_path):
        m = MLPScorerModel(hidden_size=8, seed=7)
        # 学習して非自明な重みに
        samples = generate_synthetic_dataset(n_samples=64, seed=3)
        m.fit(samples, epochs=5, learning_rate=0.01)

        path = tmp_path / "mlp"
        m.save(path)

        loaded = MLPScorerModel.load(path)
        # 同じ予測を返す
        p1 = make_set({"offset_power": 0.7})
        p2 = make_set({"death_risk": 0.5})
        assert m.predict(p1, p2).total_score == pytest.approx(
            loaded.predict(p1, p2).total_score, abs=1e-6
        )


# ============================
# generate_synthetic_dataset
# ============================


class TestSyntheticDataset:
    def test_correct_length(self):
        samples = generate_synthetic_dataset(n_samples=10)
        assert len(samples) == 10

    def test_labels_match_oracle(self):
        oracle = Scorer()
        samples = generate_synthetic_dataset(n_samples=20, oracle=oracle)
        for s in samples:
            expected = oracle.score(s.p1, s.p2).total_score
            assert s.label == pytest.approx(expected)

    def test_labels_in_range(self):
        samples = generate_synthetic_dataset(n_samples=50)
        for s in samples:
            assert SCORE_RANGE_MIN <= s.label <= SCORE_RANGE_MAX

    def test_reproducible_with_same_seed(self):
        a = generate_synthetic_dataset(n_samples=10, seed=42)
        b = generate_synthetic_dataset(n_samples=10, seed=42)
        for sa, sb in zip(a, b):
            assert sa.label == sb.label


# ============================
# 差し替え可能性: Analyzer との互換
# ============================


class TestAnalyzerCompat:
    def test_predict_has_same_signature_as_scorer_score(self):
        """Model.predict(p1, p2) と Scorer.score(p1, p2) が同じ戻り値型。"""
        scorer_result = Scorer().score(ones(), zeros())
        model_result = LinearScorerModel().predict(ones(), zeros())
        assert type(scorer_result) is type(model_result)
