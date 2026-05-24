"""W8-C: 16x16 入力 + ResNet 化 CNN v10 アーキ。

旧 CnnPatchClassifier は 8x8 入力で細部表現が損失していた。
v10 は 16x16 入力 + Residual block × 3 で表現力を強化、
データ拡張 (色相/輝度/ノイズ/ランダムクロップ) との組合せで
multi-video 汎化向上を狙う。

入力: BGR(3) + HSV(3) = 6 channel、16×16
出力: 7 クラス (EMPTY/RED/BLUE/GREEN/YELLOW/PURPLE/OJAMA)
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

from src.patch_classifier import (
    CLASS_INDEX_TO_COLOR,
    COLOR_TO_CLASS_INDEX,
    NUM_CLASSES,
    PatchClassifier,
    PatchSample,
)

PATCH_SIZE: int = 16  # 旧 8 → 16 で細部表現
INPUT_CHANNELS: int = 6  # BGR + HSV


class _ResBlock(nn.Module if _TORCH_AVAILABLE else object):  # type: ignore[misc]
    def __init__(self, ch: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(ch)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(ch)

    def forward(self, x):  # type: ignore[no-untyped-def]
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = torch.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = out + identity
        return torch.relu(out)


class CnnPatchClassifierV2(PatchClassifier):
    """16x16 + ResNet 化された色分類器 (CNN v10 用)。"""

    INPUT_CHANNELS: int = INPUT_CHANNELS

    def __init__(self, seed: int = 42, hidden_ch: int = 32) -> None:
        if not _TORCH_AVAILABLE:
            raise ImportError("torch が未インストール")
        torch.manual_seed(seed)
        self._torch = torch
        self._hidden_ch = hidden_ch
        self._model = nn.Sequential(
            nn.Conv2d(INPUT_CHANNELS, hidden_ch, 3, padding=1),
            nn.BatchNorm2d(hidden_ch),
            nn.ReLU(),
            _ResBlock(hidden_ch),
            _ResBlock(hidden_ch),
            _ResBlock(hidden_ch),
            nn.AdaptiveAvgPool2d((2, 2)),
            nn.Flatten(),
            nn.Linear(hidden_ch * 2 * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, NUM_CLASSES),
        )
        self._model.eval()

    def _patch_to_tensor(self, bgr_patch: np.ndarray):  # type: ignore[no-untyped-def]
        torch = self._torch
        if bgr_patch.size == 0:
            return torch.zeros(1, INPUT_CHANNELS, PATCH_SIZE, PATCH_SIZE)
        resized = cv2.resize(
            bgr_patch, (PATCH_SIZE, PATCH_SIZE),
            interpolation=cv2.INTER_AREA,
        )
        hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
        combined = np.concatenate([resized, hsv], axis=2)
        normalized = combined.astype(np.float32) / 255.0
        return torch.from_numpy(normalized).permute(2, 0, 1).unsqueeze(0)

    def classify(self, bgr_patch: np.ndarray) -> int:
        tensor = self._patch_to_tensor(bgr_patch)
        device = next(self._model.parameters()).device
        tensor = tensor.to(device)
        with self._torch.no_grad():
            logits = self._model(tensor)
        idx = int(self._torch.argmax(logits, dim=1).item())
        return CLASS_INDEX_TO_COLOR[idx]

    def predict_proba(self, bgr_patch: np.ndarray) -> np.ndarray:
        tensor = self._patch_to_tensor(bgr_patch)
        device = next(self._model.parameters()).device
        tensor = tensor.to(device)
        with self._torch.no_grad():
            logits = self._model(tensor)
            probs = self._torch.nn.functional.softmax(logits, dim=1)
        return probs[0].cpu().numpy()

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._torch.save(self._model.state_dict(), str(path))

    def load(self, path: str | Path) -> None:
        state = self._torch.load(
            str(path), map_location="cpu", weights_only=True,
        )
        self._model.load_state_dict(state)
        self._model.eval()


def _augment_patch(
    bgr: np.ndarray, rng: np.random.Generator,
) -> np.ndarray:
    """W8-A: 色相 ±5°、輝度 ±10%、ガウシアンノイズ、ランダムクロップ。"""
    out = bgr.copy()
    # 色相シフト
    hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV).astype(np.int16)
    h_shift = int(rng.integers(-5, 6))  # ±5°
    hsv[..., 0] = (hsv[..., 0] + h_shift) % 180
    # 彩度・明度シフト
    s_scale = float(rng.uniform(0.9, 1.1))
    v_scale = float(rng.uniform(0.9, 1.1))
    hsv[..., 1] = np.clip(hsv[..., 1] * s_scale, 0, 255)
    hsv[..., 2] = np.clip(hsv[..., 2] * v_scale, 0, 255)
    out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    # ガウシアンノイズ
    noise = rng.normal(0, 4.0, out.shape).astype(np.int16)
    out = np.clip(out.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return out


class CnnTrainerV2:
    """データ拡張込み訓練 (CNN v10 用)。"""

    def __init__(
        self, classifier: CnnPatchClassifierV2, augment: bool = True,
        seed: int = 42,
    ) -> None:
        self._cls = classifier
        self._augment = augment
        self._rng = np.random.default_rng(seed)

    def fit(
        self,
        samples: Sequence[PatchSample],
        epochs: int = 20,
        lr: float = 1e-3,
        batch_size: int = 64,
        class_weighted: bool = True,
        weight_decay: float = 1e-4,
    ) -> list[float]:
        if not samples:
            raise ValueError("learning samples empty")
        torch = self._cls._torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self._cls._model.to(device)
        self._cls._model.train()

        # 各サンプルのテンソル化 (拡張は epoch ごとに変える)
        n = len(samples)
        labels_idx = np.array([
            COLOR_TO_CLASS_INDEX[int(s.color)] for s in samples
        ], dtype=np.int64)

        # クラス重み (逆頻度)
        if class_weighted:
            counts = np.bincount(labels_idx, minlength=NUM_CLASSES)
            inv = 1.0 / np.clip(counts, 1, None)
            inv = inv / inv.sum() * NUM_CLASSES
            weight_t = torch.tensor(inv, dtype=torch.float32, device=device)
            criterion = torch.nn.CrossEntropyLoss(weight=weight_t)
        else:
            criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(
            self._cls._model.parameters(), lr=lr, weight_decay=weight_decay,
        )

        losses: list[float] = []
        for epoch in range(epochs):
            perm = self._rng.permutation(n)
            total_loss = 0.0
            n_batch = 0
            for s in range(0, n, batch_size):
                idx = perm[s:s + batch_size]
                batch_patches = []
                for i in idx:
                    p = samples[i].patch
                    if self._augment:
                        p = _augment_patch(p, self._rng)
                    resized = cv2.resize(
                        p, (PATCH_SIZE, PATCH_SIZE),
                        interpolation=cv2.INTER_AREA,
                    )
                    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
                    combined = np.concatenate([resized, hsv], axis=2)
                    batch_patches.append(combined)
                X = np.stack(batch_patches).astype(np.float32) / 255.0
                X_t = torch.from_numpy(X).permute(0, 3, 1, 2).to(device)
                y_t = torch.from_numpy(labels_idx[idx]).to(device)
                optimizer.zero_grad()
                logits = self._cls._model(X_t)
                loss = criterion(logits, y_t)
                loss.backward()
                optimizer.step()
                total_loss += float(loss.item())
                n_batch += 1
            avg = total_loss / max(1, n_batch)
            losses.append(avg)
            print(f"  epoch {epoch + 1}/{epochs}: loss={avg:.4f}")
        self._cls._model.eval()
        return losses


__all__ = [
    "CnnPatchClassifierV2",
    "CnnTrainerV2",
    "PATCH_SIZE",
]
