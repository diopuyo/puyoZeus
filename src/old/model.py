"""
総合判定 ML モデルモジュール (Phase 4)

Scorer (ルールベース) を差し替え可能な学習ベースモデル。numpy のみで実装し、
将来的に torch 実装への差し替えも想定する。

提供モデル:
    - LinearScorerModel:  8 指標差分の線形結合 (Scorer とほぼ同型)
    - MLPScorerModel:     2 層 MLP で非線形相互作用を学習
    - Trainer:            教師信号 (label) から学習する共通ループ

インタフェース:
    .predict(player1, player2) -> ScoreResult   (Scorer 互換)
    .fit(samples, epochs, lr)                   (学習)
    .save(path) / .load(path)                   (永続化)
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from src.old.indicators import ALL_INDICATOR_NAMES, IndicatorSet
from src.old.scorer import (
    DEFAULT_WEIGHTS,
    SCORE_RANGE_MAX,
    SCORE_RANGE_MIN,
    ScoreResult,
    Scorer,
)

# ============================
# 定数定義
# ============================

# 指標ベクトル次元 (= 8)
FEATURE_DIM: int = len(ALL_INDICATOR_NAMES)

# MLP 構成
MLP_HIDDEN_SIZE: int = 16

# 学習ハイパーパラメータ
DEFAULT_LEARNING_RATE: float = 0.01
DEFAULT_EPOCHS: int = 200
DEFAULT_BATCH_SIZE: int = 32

# ランダム初期化シード固定
DEFAULT_SEED: int = 42

# 保存フォーマット識別子
FORMAT_LINEAR: str = "linear_v1"
FORMAT_MLP: str = "mlp_v1"


# ============================
# データクラス
# ============================


@dataclass
class ScoreSample:
    """
    学習サンプル 1 件。

    Attributes:
        p1: 1P 指標セット。
        p2: 2P 指標セット。
        label: 教師スコア (-100〜+100)。
    """
    p1: IndicatorSet
    p2: IndicatorSet
    label: float


@dataclass
class TrainingReport:
    """学習結果サマリ。"""
    epochs: int
    final_loss: float
    loss_history: list[float] = field(default_factory=list)


# ============================
# 特徴量変換
# ============================


def indicator_vector(indicator_set: IndicatorSet) -> np.ndarray:
    """
    IndicatorSet を ALL_INDICATOR_NAMES 順に並べた 8 次元ベクトルに変換する。

    欠けている指標は 0 とみなす。

    Args:
        indicator_set: 指標セット。

    Returns:
        np.ndarray: shape=(FEATURE_DIM,) の float64 配列。
    """
    vec = np.zeros(FEATURE_DIM, dtype=np.float64)
    for i, name in enumerate(ALL_INDICATOR_NAMES):
        if name in indicator_set.results:
            vec[i] = indicator_set.score_of(name)
    return vec


# ============================
# BaseScorerModel
# ============================


class BaseScorerModel(ABC):
    """
    学習可能なスコアラーの抽象基底クラス。
    Scorer と同じ predict() インタフェースを提供する。
    """

    @abstractmethod
    def predict_raw(self, p1_vec: np.ndarray, p2_vec: np.ndarray) -> float:
        """2 つの指標ベクトルから -100〜+100 のスコアを計算する。"""
        ...

    @abstractmethod
    def fit(
        self,
        samples: Sequence[ScoreSample],
        epochs: int = DEFAULT_EPOCHS,
        learning_rate: float = DEFAULT_LEARNING_RATE,
    ) -> TrainingReport:
        """サンプルから学習する。"""
        ...

    @abstractmethod
    def save(self, path: Path) -> None:
        """モデルを永続化する。"""
        ...

    # ============================
    # Scorer 互換 API
    # ============================

    def predict(
        self, player1: IndicatorSet, player2: IndicatorSet,
    ) -> ScoreResult:
        """Scorer.score() 互換: IndicatorSet から ScoreResult を返す。"""
        p1_vec = indicator_vector(player1)
        p2_vec = indicator_vector(player2)
        total = self._clamp_score(self.predict_raw(p1_vec, p2_vec))
        return ScoreResult(
            total_score=total,
            player1_raw=float(np.sum(p1_vec)),
            player2_raw=float(np.sum(p2_vec)),
            player1_breakdown=self._to_breakdown(p1_vec),
            player2_breakdown=self._to_breakdown(p2_vec),
            weights={},
        )

    # ============================
    # ユーティリティ
    # ============================

    @staticmethod
    def _clamp_score(value: float) -> float:
        """スコアを定義域にクランプ。"""
        return max(SCORE_RANGE_MIN, min(SCORE_RANGE_MAX, float(value)))

    @staticmethod
    def _to_breakdown(vec: np.ndarray) -> dict[str, float]:
        """ベクトルを指標名→値の辞書にする (可視化用)。"""
        return {
            name: float(vec[i])
            for i, name in enumerate(ALL_INDICATOR_NAMES)
        }


# ============================
# LinearScorerModel
# ============================


class LinearScorerModel(BaseScorerModel):
    """
    線形モデル: score = SCORE_RANGE_MAX * tanh( w · (p1 - p2) + b )

    Scorer とほぼ同形だが重みを学習で調整できる。初期値は Scorer の
    DEFAULT_WEIGHTS から warm-start することも可能。
    """

    def __init__(
        self,
        weights: np.ndarray | None = None,
        bias: float = 0.0,
    ) -> None:
        """
        Args:
            weights: shape=(FEATURE_DIM,) の重みベクトル (None ならゼロ初期化)。
            bias: バイアス項。
        """
        if weights is None:
            weights = np.zeros(FEATURE_DIM, dtype=np.float64)
        if weights.shape != (FEATURE_DIM,):
            raise ValueError(
                f"weights の形状が不正: {weights.shape} (期待: ({FEATURE_DIM},))"
            )
        self._w = weights.astype(np.float64).copy()
        self._b = float(bias)

    # ============================
    # ファクトリ
    # ============================

    @classmethod
    def from_scorer_weights(
        cls, weights: dict[str, float] | None = None,
    ) -> "LinearScorerModel":
        """Scorer.DEFAULT_WEIGHTS などから初期化する。"""
        src = weights if weights is not None else DEFAULT_WEIGHTS
        w = np.zeros(FEATURE_DIM, dtype=np.float64)
        for i, name in enumerate(ALL_INDICATOR_NAMES):
            w[i] = src.get(name, 0.0)
        return cls(weights=w)

    # ============================
    # 予測
    # ============================

    def predict_raw(self, p1_vec: np.ndarray, p2_vec: np.ndarray) -> float:
        diff = p1_vec - p2_vec
        logit = float(np.dot(self._w, diff) + self._b)
        return SCORE_RANGE_MAX * np.tanh(logit)

    # ============================
    # 学習 (MSE + SGD)
    # ============================

    def fit(
        self,
        samples: Sequence[ScoreSample],
        epochs: int = DEFAULT_EPOCHS,
        learning_rate: float = DEFAULT_LEARNING_RATE,
    ) -> TrainingReport:
        if not samples:
            raise ValueError("学習サンプルが空です")

        X_diff, y = self._prepare_linear_batch(samples)
        losses: list[float] = []
        for _ in range(epochs):
            loss = self._step_linear(X_diff, y, learning_rate)
            losses.append(loss)
        return TrainingReport(
            epochs=epochs, final_loss=losses[-1], loss_history=losses,
        )

    def _step_linear(
        self, X_diff: np.ndarray, y: np.ndarray, lr: float,
    ) -> float:
        """
        1 epoch 勾配降下 (フルバッチ)。
        数値安定のため内部では y を [-1,1] に正規化して学習する。
        戻り値は元スケールの MSE。
        """
        logits = X_diff @ self._w + self._b
        tanh_logits = np.tanh(logits)
        preds = SCORE_RANGE_MAX * tanh_logits

        # 内部残差は正規化空間 [-1, 1] で計算
        y_norm = y / SCORE_RANGE_MAX
        residual_norm = tanh_logits - y_norm

        tanh_grad = 1.0 - tanh_logits ** 2
        dloss_dlogit = (2.0 * residual_norm / len(y)) * tanh_grad

        grad_w = X_diff.T @ dloss_dlogit
        grad_b = float(np.sum(dloss_dlogit))

        self._w -= lr * grad_w
        self._b -= lr * grad_b

        return float(np.mean((preds - y) ** 2))

    @staticmethod
    def _prepare_linear_batch(
        samples: Sequence[ScoreSample],
    ) -> tuple[np.ndarray, np.ndarray]:
        """サンプル列を行列化する。"""
        X_p1 = np.stack([indicator_vector(s.p1) for s in samples])
        X_p2 = np.stack([indicator_vector(s.p2) for s in samples])
        y = np.array([s.label for s in samples], dtype=np.float64)
        return X_p1 - X_p2, y

    # ============================
    # 永続化
    # ============================

    def save(self, path: Path) -> None:
        """JSON で永続化する。"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format": FORMAT_LINEAR,
            "weights": self._w.tolist(),
            "bias": self._b,
        }
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path) -> "LinearScorerModel":
        """JSON から復元する。"""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if data.get("format") != FORMAT_LINEAR:
            raise ValueError(f"フォーマット不一致: {data.get('format')}")
        return cls(
            weights=np.array(data["weights"], dtype=np.float64),
            bias=float(data["bias"]),
        )

    # ============================
    # 参照 API
    # ============================

    def weights_dict(self) -> dict[str, float]:
        """重みを指標名→値の辞書で返す。"""
        return {
            name: float(self._w[i])
            for i, name in enumerate(ALL_INDICATOR_NAMES)
        }


# ============================
# MLPScorerModel
# ============================


class MLPScorerModel(BaseScorerModel):
    """
    2 層 MLP: concat(p1, p2) -> hidden -> scalar -> tanh * 100

    非線形相互作用を捉えるが、学習には十分なサンプルが必要。
    """

    def __init__(
        self,
        hidden_size: int = MLP_HIDDEN_SIZE,
        seed: int = DEFAULT_SEED,
    ) -> None:
        """
        Args:
            hidden_size: 隠れ層のユニット数。
            seed: 乱数シード。
        """
        self._hidden_size = hidden_size
        rng = np.random.default_rng(seed)
        input_dim = FEATURE_DIM * 2
        # Xavier 初期化
        scale_1 = np.sqrt(2.0 / input_dim)
        scale_2 = np.sqrt(2.0 / hidden_size)
        self._W1 = rng.normal(0, scale_1, size=(input_dim, hidden_size))
        self._b1 = np.zeros(hidden_size)
        self._W2 = rng.normal(0, scale_2, size=(hidden_size,))
        self._b2 = 0.0

    # ============================
    # 予測
    # ============================

    def predict_raw(self, p1_vec: np.ndarray, p2_vec: np.ndarray) -> float:
        x = np.concatenate([p1_vec, p2_vec])
        h = np.tanh(x @ self._W1 + self._b1)
        logit = float(h @ self._W2 + self._b2)
        return SCORE_RANGE_MAX * np.tanh(logit)

    # ============================
    # 学習 (MSE + ミニバッチ SGD)
    # ============================

    def fit(
        self,
        samples: Sequence[ScoreSample],
        epochs: int = DEFAULT_EPOCHS,
        learning_rate: float = DEFAULT_LEARNING_RATE,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> TrainingReport:
        if not samples:
            raise ValueError("学習サンプルが空です")

        X = np.stack([
            np.concatenate([
                indicator_vector(s.p1), indicator_vector(s.p2),
            ])
            for s in samples
        ])
        y = np.array([s.label for s in samples], dtype=np.float64)

        n = len(samples)
        losses: list[float] = []
        rng = np.random.default_rng(0)

        for _ in range(epochs):
            indices = rng.permutation(n)
            total_loss = 0.0
            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                batch_idx = indices[start:end]
                loss = self._step_mlp(X[batch_idx], y[batch_idx], learning_rate)
                total_loss += loss * (end - start)
            losses.append(total_loss / n)

        return TrainingReport(
            epochs=epochs, final_loss=losses[-1], loss_history=losses,
        )

    def _step_mlp(
        self, X: np.ndarray, y: np.ndarray, lr: float,
    ) -> float:
        """
        1 ミニバッチ分の勾配ステップ。内部は [-1, 1] で学習する。
        戻り値は元スケール ([-100, 100]) の MSE。
        """
        n = len(y)
        # 順伝播
        z1 = X @ self._W1 + self._b1
        h = np.tanh(z1)
        z2 = h @ self._W2 + self._b2
        tanh_z2 = np.tanh(z2)
        preds = SCORE_RANGE_MAX * tanh_z2

        y_norm = y / SCORE_RANGE_MAX
        residual_norm = tanh_z2 - y_norm
        loss = float(np.mean((preds - y) ** 2))

        # 逆伝播 (正規化空間)
        dz2 = (2.0 * residual_norm / n) * (1.0 - tanh_z2 ** 2)
        grad_W2 = h.T @ dz2
        grad_b2 = float(np.sum(dz2))

        dh = np.outer(dz2, self._W2)
        dz1 = dh * (1.0 - np.tanh(z1) ** 2)
        grad_W1 = X.T @ dz1
        grad_b1 = np.sum(dz1, axis=0)

        self._W1 -= lr * grad_W1
        self._b1 -= lr * grad_b1
        self._W2 -= lr * grad_W2
        self._b2 -= lr * grad_b2

        return loss

    # ============================
    # 永続化
    # ============================

    def save(self, path: Path) -> None:
        """npz で永続化する (メタデータ JSON サイドカー付き)。"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            W1=self._W1, b1=self._b1,
            W2=self._W2, b2=np.array([self._b2]),
        )
        meta_path = path.with_suffix(".json")
        meta_path.write_text(
            json.dumps({
                "format": FORMAT_MLP,
                "hidden_size": self._hidden_size,
            }, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path) -> "MLPScorerModel":
        """npz + サイドカー JSON から復元する。"""
        path = Path(path)
        npz_path = path if path.suffix else path.with_suffix(".npz")
        if not npz_path.exists():
            npz_path = path.with_suffix(".npz")
        meta_path = npz_path.with_suffix(".json")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("format") != FORMAT_MLP:
            raise ValueError(f"フォーマット不一致: {meta.get('format')}")
        data = np.load(npz_path)
        model = cls(hidden_size=int(meta["hidden_size"]))
        model._W1 = data["W1"]
        model._b1 = data["b1"]
        model._W2 = data["W2"]
        model._b2 = float(data["b2"][0])
        return model


# ============================
# 学習データ生成ヘルパー
# ============================


def generate_synthetic_dataset(
    n_samples: int,
    oracle: Scorer | None = None,
    seed: int = DEFAULT_SEED,
) -> list[ScoreSample]:
    """
    Scorer をオラクルとして擬似的な学習データを生成する。

    ML モデルの学習パイプラインが正しく動くかを確認する目的で、
    各指標をランダムに生成し、ルールベースの Scorer ラベルで教師化する。

    Args:
        n_samples: 生成件数。
        oracle: 教師ラベル生成に使う Scorer (None なら DEFAULT_WEIGHTS)。
        seed: 乱数シード。

    Returns:
        list[ScoreSample]: 生成されたサンプル列。
    """
    oracle = oracle or Scorer()
    rng = np.random.default_rng(seed)
    samples: list[ScoreSample] = []
    for _ in range(n_samples):
        p1 = _random_indicator_set(rng)
        p2 = _random_indicator_set(rng)
        label = oracle.score(p1, p2).total_score
        samples.append(ScoreSample(p1=p1, p2=p2, label=label))
    return samples


def _random_indicator_set(rng: np.random.Generator) -> IndicatorSet:
    """ランダムな指標セットを生成する (0〜1 の一様分布)。"""
    from src.old.indicators import IndicatorResult

    results: dict[str, IndicatorResult] = {}
    for name in ALL_INDICATOR_NAMES:
        s = float(rng.uniform(0.0, 1.0))
        results[name] = IndicatorResult(name=name, score=s, raw_value=s)
    return IndicatorSet(results=results)
