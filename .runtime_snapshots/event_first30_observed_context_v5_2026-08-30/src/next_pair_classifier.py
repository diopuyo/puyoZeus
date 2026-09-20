"""W8-E: NextDetector 専用 CNN (32x32 入力、5 色分類)。

NextDetector は現状、汎用 CnnPatchClassifier (8x8 board cell 用) を使用。
Next pair パッチ (75x75) は board cell より大きく、背景・ぷよ形状も異なる
ため専用モデルが有利。

W8-D で 19 動画から 28576 件 (5色均衡) のラベル付き next pair patches を
StableNextDetector + 1P/2P 一致条件で収集済。これで dedicated CNN を訓練する。

入力: 32x32 BGR + HSV 6ch (NextDetector からの patch を resize)
出力: 5 クラス (RED, BLUE, GREEN, YELLOW, PURPLE)
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

try:
    import torch
    import torch.nn as nn
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

from src.board import (
    COLOR_BLUE, COLOR_GREEN, COLOR_PURPLE, COLOR_RED, COLOR_YELLOW,
)

PATCH_SIZE: int = 32
INPUT_CHANNELS: int = 6  # BGR + HSV
NUM_CLASSES: int = 5

CLASS_INDEX_TO_COLOR: tuple[int, ...] = (
    COLOR_RED, COLOR_BLUE, COLOR_GREEN, COLOR_YELLOW, COLOR_PURPLE,
)
COLOR_TO_CLASS_INDEX: dict[int, int] = {
    c: i for i, c in enumerate(CLASS_INDEX_TO_COLOR)
}


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


class NextPairClassifier:
    """NextDetector 専用 5 色分類器 (32x32 入力)。"""

    def __init__(self, seed: int = 42, hidden_ch: int = 32) -> None:
        if not _TORCH_AVAILABLE:
            raise ImportError("torch が未インストール")
        torch.manual_seed(seed)
        self._torch = torch
        self._model = nn.Sequential(
            nn.Conv2d(INPUT_CHANNELS, hidden_ch, 3, padding=1),
            nn.BatchNorm2d(hidden_ch),
            nn.ReLU(),
            _ResBlock(hidden_ch),
            nn.MaxPool2d(2),  # 32 → 16
            _ResBlock(hidden_ch),
            nn.MaxPool2d(2),  # 16 → 8
            _ResBlock(hidden_ch),
            nn.AdaptiveAvgPool2d((2, 2)),
            nn.Flatten(),
            nn.Linear(hidden_ch * 2 * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
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
        """1 patch → 色コード (RED/BLUE/GRN/YEL/PUR)。"""
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


def _augment_next_patch(
    bgr: np.ndarray, rng: np.random.Generator,
) -> np.ndarray:
    """色相 ±5°、彩度/輝度 ±10%、ガウシアンノイズ。"""
    out = bgr.copy()
    hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV).astype(np.int16)
    h_shift = int(rng.integers(-5, 6))
    hsv[..., 0] = (hsv[..., 0] + h_shift) % 180
    s_scale = float(rng.uniform(0.9, 1.1))
    v_scale = float(rng.uniform(0.9, 1.1))
    hsv[..., 1] = np.clip(hsv[..., 1] * s_scale, 0, 255)
    hsv[..., 2] = np.clip(hsv[..., 2] * v_scale, 0, 255)
    out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    noise = rng.normal(0, 4.0, out.shape).astype(np.int16)
    out = np.clip(out.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return out


__all__ = [
    "CLASS_INDEX_TO_COLOR",
    "COLOR_TO_CLASS_INDEX",
    "INPUT_CHANNELS",
    "NUM_CLASSES",
    "NextPairClassifier",
    "PATCH_SIZE",
    "_augment_next_patch",
]
