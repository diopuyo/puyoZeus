"""W2.1: 勝率予測モデル (state vector → 1P 勝利確率)。

シンプルな MLP で GameState の feature vector → 1P 勝率 (sigmoid) を学習。
Phase W2.3 で訓練、W3.2 で holdout 評価。

設計:
    - 入力: TOTAL_FEATURE_DIM (1068) の float32 vector
    - アーキ: 1068 → 256 → 128 → 64 → 1, ReLU + Dropout, sigmoid 出力
    - 損失: BCE (binary cross-entropy)
    - 訓練: scripts/phase_w_train_predictor.py で実装予定
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    import torch
    import torch.nn as nn
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

from src.state_features import TOTAL_FEATURE_DIM


# デフォルトハイパーパラメータ
# 過学習しやすいデータ (3000 件 × 1068 次元) のため、隠れ層を小さく + dropout 強め
DEFAULT_HIDDEN_DIMS: tuple[int, ...] = (64, 32)
DEFAULT_DROPOUT: float = 0.5


def _check_torch() -> None:
    if not _TORCH_AVAILABLE:
        raise ImportError(
            "torch が未インストールです。pip install torch で導入してください。"
        )


class WinPredictorMLP:
    """勝率予測 MLP モデル。

    Usage:
        model = WinPredictorMLP()
        model.fit(features, labels, epochs=20)
        prob = model.predict(state_features)  # 0..1
    """

    def __init__(
        self,
        input_dim: int = TOTAL_FEATURE_DIM,
        hidden_dims: tuple[int, ...] = DEFAULT_HIDDEN_DIMS,
        dropout: float = DEFAULT_DROPOUT,
        seed: int = 42,
    ) -> None:
        _check_torch()
        torch.manual_seed(seed)
        self._input_dim = int(input_dim)
        self._hidden_dims = tuple(hidden_dims)
        self._dropout = float(dropout)
        self._model: nn.Module = self._build_model()

    def _build_model(self) -> nn.Module:
        layers: list[nn.Module] = []
        prev = self._input_dim
        for h in self._hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            if self._dropout > 0:
                layers.append(nn.Dropout(self._dropout))
            prev = h
        layers.append(nn.Linear(prev, 1))
        # sigmoid は損失内 (BCEWithLogitsLoss) で適用するためモデルでは省略
        return nn.Sequential(*layers)

    def fit(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        epochs: int = 20,
        lr: float = 1e-3,
        batch_size: int = 64,
        weight_decay: float = 1e-5,
        verbose: bool = True,
    ) -> list[float]:
        """features (N, D) と labels (N,) で訓練。labels は 0/1。"""
        assert features.shape[0] == labels.shape[0]
        assert features.shape[1] == self._input_dim
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model.to(device)
        self._model.train()

        X = torch.from_numpy(features.astype(np.float32))
        y = torch.from_numpy(labels.astype(np.float32))

        opt = torch.optim.Adam(
            self._model.parameters(), lr=lr, weight_decay=weight_decay,
        )
        criterion = nn.BCEWithLogitsLoss()

        n = X.shape[0]
        losses: list[float] = []
        for epoch in range(epochs):
            perm = torch.randperm(n)
            total_loss = 0.0
            n_batch = 0
            for s in range(0, n, batch_size):
                idx = perm[s:s + batch_size]
                xb = X[idx].to(device)
                yb = y[idx].to(device).unsqueeze(1)
                opt.zero_grad()
                logits = self._model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                opt.step()
                total_loss += float(loss.item())
                n_batch += 1
            avg = total_loss / max(1, n_batch)
            losses.append(avg)
            if verbose:
                print(f"  epoch {epoch + 1}/{epochs}: loss={avg:.4f}")
        return losses

    def predict(self, features: np.ndarray) -> np.ndarray:
        """features (N, D) または (D,) → 勝率 (N,) または scalar。"""
        single = features.ndim == 1
        if single:
            features = features.reshape(1, -1)
        device = next(self._model.parameters()).device
        self._model.eval()
        with torch.no_grad():
            x = torch.from_numpy(features.astype(np.float32)).to(device)
            logits = self._model(x)
            probs = torch.sigmoid(logits).cpu().numpy().reshape(-1)
        return float(probs[0]) if single else probs

    def save(self, path: str | Path) -> None:
        """torch state_dict として保存。"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self._model.state_dict(), str(path))

    def load(self, path: str | Path) -> None:
        """torch state_dict をロード。"""
        state = torch.load(str(path), map_location="cpu", weights_only=True)
        self._model.load_state_dict(state)
        self._model.eval()


__all__ = [
    "DEFAULT_DROPOUT",
    "DEFAULT_HIDDEN_DIMS",
    "WinPredictorMLP",
]
