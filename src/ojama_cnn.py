"""予告お邪魔ぷよアイコン分類用の軽量 CNN。

入力: 36×36 BGR パッチ (中央切出)
出力: 7 クラス (empty, small, line, rock, moon, crown, big_crown)

軽量設計:
    - Conv2D(16) → Pool → Conv2D(32) → Pool → Conv2D(64) → GlobalAvgPool
    - FC(64) → FC(7)
    - パラメータ ~30K (CPU でも訓練・推論高速)
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.ojama_warning import (
    ICON_BIG_CROWN,
    ICON_CROWN,
    ICON_EMPTY,
    ICON_LINE,
    ICON_MOON,
    ICON_ROCK,
    ICON_SMALL,
)

# クラスインデックス順 (固定)
OJAMA_CLASSES: tuple[str, ...] = (
    ICON_EMPTY,
    ICON_SMALL,
    ICON_LINE,
    ICON_ROCK,
    ICON_MOON,
    ICON_CROWN,
    ICON_BIG_CROWN,
)
N_CLASSES: int = len(OJAMA_CLASSES)

# 入力サイズ (中央 36×36)
INPUT_HEIGHT: int = 36
INPUT_WIDTH: int = 36

# モデルファイルパス (デフォルト)
DEFAULT_MODEL_PATH: Path = Path("models/ojama_cnn.pt")


class _ResBlock(nn.Module):
    """2 Conv + skip connection の小規模 ResNet ブロック。"""

    def __init__(self, ch_in: int, ch_out: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(ch_in, ch_out, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(ch_out)
        self.conv2 = nn.Conv2d(ch_out, ch_out, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(ch_out)
        self.shortcut: nn.Module
        if ch_in != ch_out:
            self.shortcut = nn.Sequential(
                nn.Conv2d(ch_in, ch_out, 1, bias=False),
                nn.BatchNorm2d(ch_out),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x = F.relu(x + residual)
        return x


class OjamaCnn(nn.Module):
    """ResNet 風の軽量分類器 (3 ブロック + GAP + FC)。

    2026-04-27: Phase I で深化。チャンネル数 16/32/64 → 32/64/128、
    各ブロック 2 Conv + skip connection、計約 220K params。
    """

    def __init__(self, n_classes: int = N_CLASSES) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.block1 = _ResBlock(32, 64)
        self.block2 = _ResBlock(64, 128)
        self.block3 = _ResBlock(128, 128)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, n_classes)
        self.dropout = nn.Dropout(0.4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 3, 36, 36)
        x = self.stem(x)
        x = self.block1(x)
        x = F.max_pool2d(x, 2)  # 18x18
        x = self.block2(x)
        x = F.max_pool2d(x, 2)  # 9x9
        x = self.block3(x)
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)  # (B, 128)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

    @torch.no_grad()
    def predict_class(self, patch_bgr: np.ndarray) -> tuple[str, float]:
        """36×36 BGR パッチからクラス名と confidence を返す。"""
        self.eval()
        if patch_bgr.shape[:2] != (INPUT_HEIGHT, INPUT_WIDTH):
            raise ValueError(
                f"入力サイズ不一致: {patch_bgr.shape[:2]}, "
                f"期待={INPUT_HEIGHT}x{INPUT_WIDTH}"
            )
        x = torch.from_numpy(patch_bgr).float() / 255.0
        x = x.permute(2, 0, 1).unsqueeze(0)  # (1,3,36,36)
        logits = self.forward(x)
        probs = F.softmax(logits, dim=1)[0]
        idx = int(torch.argmax(probs).item())
        return OJAMA_CLASSES[idx], float(probs[idx].item())


def load_cnn(
    path: Path = DEFAULT_MODEL_PATH,
    device: str = "cpu",
) -> OjamaCnn | None:
    """訓練済みモデルを読み込む (なければ None)。"""
    if not path.exists():
        return None
    model = OjamaCnn()
    state = torch.load(str(path), map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


def save_cnn(model: OjamaCnn, path: Path = DEFAULT_MODEL_PATH) -> None:
    """訓練済みモデルを保存する。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), str(path))


__all__ = [
    "DEFAULT_MODEL_PATH",
    "INPUT_HEIGHT",
    "INPUT_WIDTH",
    "N_CLASSES",
    "OJAMA_CLASSES",
    "OjamaCnn",
    "load_cnn",
    "save_cnn",
]
