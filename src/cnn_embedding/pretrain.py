"""SimCLR ベースの BoardCNN 自己教師あり事前学習モジュール.

設計方針:
    - 既存 `BoardCNN` (8ch×13×6 → embed_dim) を backbone として再利用.
    - SimCLR の標準構成: backbone → projection MLP → NT-Xent loss.
    - augmentation は board (one-hot 8ch) に対する 4 種:
        * 色 permutation (戦略色は交換可換)
        * 左右反転 (col 軸)
        * partial mask (1〜3 cell を EMPTY に)
        * 行 padding/crop (上端 1〜2 row シフト)
    - 各 augmentation は独立確率で適用、view A / view B を生成し contrastive learning.

注意:
    - 1 関数 50 行以内.
    - マジックナンバーは module 上部の定数で集約.
    - 既存 `src.cnn_embedding.board_cnn` の API は壊さない.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.cnn_embedding.board_cnn import (
    BOARD_COLS,
    BOARD_ROWS,
    BoardCNN,
    EMBED_DIM,
    N_COLOR_CHANNELS,
)

# ============================
# 定数 (マジックナンバー回避)
# ============================
DEFAULT_PROJECTION_DIM: int = 64
DEFAULT_HIDDEN_DIM: int = 128
DEFAULT_TEMPERATURE: float = 0.5

# augmentation 各処理の適用確率 (独立)
AUG_PROB_COLOR_PERM: float = 0.5
AUG_PROB_HFLIP: float = 0.5
AUG_PROB_MASK: float = 0.5
AUG_PROB_SHIFT: float = 0.5

# partial mask: 1〜3 cell を EMPTY に
MASK_MIN_CELLS: int = 1
MASK_MAX_CELLS: int = 3

# 行 shift: -2〜+2 (-=上にずらして上端 row を欠落、+=下にずらして上端 row 追加)
SHIFT_MAX_ABS: int = 2

# 色 permutation 対象 channel (色 ch1..ch5、空/おじゃま/UNKNOWN は不変)
COLOR_CHANNELS: tuple[int, ...] = (1, 2, 3, 4, 5)

# board の channel index
CH_EMPTY: int = 0


# ============================
# Encoder + Projection head
# ============================
class SimCLRBoardEncoder(nn.Module):
    """BoardCNN backbone + 2 層 projection head (SimCLR 標準)."""

    def __init__(
        self,
        embed_dim: int = EMBED_DIM,
        projection_dim: int = DEFAULT_PROJECTION_DIM,
        hidden_dim: int = DEFAULT_HIDDEN_DIM,
        in_channels: int = N_COLOR_CHANNELS,
    ) -> None:
        super().__init__()
        self.encoder = BoardCNN(in_channels=in_channels, embed_dim=embed_dim)
        self.projection = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, projection_dim),
        )

    def forward(
        self, x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """(B, C, H, W) → (h: backbone embedding, z: projection)."""
        h = self.encoder(x)
        z = self.projection(h)
        return h, z


# ============================
# NT-Xent loss
# ============================
def nt_xent_loss(
    z1: torch.Tensor, z2: torch.Tensor,
    temperature: float = DEFAULT_TEMPERATURE,
) -> torch.Tensor:
    """SimCLR の NT-Xent (normalized temperature-scaled cross entropy) loss.

    z1, z2: (B, D). 同じ index 同士が positive pair.
    L2 normalize → 2B×2B cosine 類似度行列 → 自己除外で対称な CE を計算.
    """
    if z1.shape != z2.shape:
        raise ValueError(f"z1/z2 shape 不一致: {z1.shape} vs {z2.shape}")
    batch_size = z1.size(0)
    z = torch.cat([z1, z2], dim=0)              # (2B, D)
    z = F.normalize(z, dim=-1)
    sim = z @ z.t() / temperature                # (2B, 2B)
    # 自己類似度を -inf に (対角要素 = 自分自身)
    mask_self = torch.eye(2 * batch_size, dtype=torch.bool, device=sim.device)
    sim = sim.masked_fill(mask_self, float("-inf"))
    # positive index: i ↔ i+B (循環)
    targets = torch.arange(2 * batch_size, device=sim.device)
    targets = (targets + batch_size) % (2 * batch_size)
    return F.cross_entropy(sim, targets)


# ============================
# Augmentation (numpy ベース)
# ============================
@dataclass
class AugmentConfig:
    """augmentation 確率のオーバーライド用 dataclass."""

    p_color_perm: float = AUG_PROB_COLOR_PERM
    p_hflip: float = AUG_PROB_HFLIP
    p_mask: float = AUG_PROB_MASK
    p_shift: float = AUG_PROB_SHIFT


def _aug_color_permute(
    onehot: np.ndarray, rng: np.random.Generator,
) -> np.ndarray:
    """色 channel (1..5) を random permutation で入れ替える."""
    perm = rng.permutation(len(COLOR_CHANNELS))
    out = onehot.copy()
    src = onehot[list(COLOR_CHANNELS)]            # (5, H, W)
    for i, ch in enumerate(COLOR_CHANNELS):
        out[ch] = src[perm[i]]
    return out


def _aug_hflip(onehot: np.ndarray) -> np.ndarray:
    """col 軸を左右反転."""
    return onehot[:, :, ::-1].copy()


def _aug_partial_mask(
    onehot: np.ndarray, rng: np.random.Generator,
) -> np.ndarray:
    """ランダム 1..3 cell を EMPTY (ch0) に置き換える."""
    n_cells = int(rng.integers(MASK_MIN_CELLS, MASK_MAX_CELLS + 1))
    out = onehot.copy()
    rows = rng.integers(0, BOARD_ROWS, size=n_cells)
    cols = rng.integers(0, BOARD_COLS, size=n_cells)
    for r, c in zip(rows, cols):
        out[:, r, c] = 0.0
        out[CH_EMPTY, r, c] = 1.0
    return out


def _aug_row_shift(
    onehot: np.ndarray, rng: np.random.Generator,
) -> np.ndarray:
    """行を ±SHIFT_MAX_ABS 範囲で ±k シフト. はみ出し側は EMPTY で埋める."""
    k = int(rng.integers(-SHIFT_MAX_ABS, SHIFT_MAX_ABS + 1))
    if k == 0:
        return onehot.copy()
    out = np.zeros_like(onehot)
    out[CH_EMPTY] = 1.0
    if k > 0:                                    # 下にずらす (上が空く)
        out[:, k:, :] = onehot[:, :BOARD_ROWS - k, :]
        out[CH_EMPTY, k:, :] = onehot[CH_EMPTY, :BOARD_ROWS - k, :]
    else:                                        # 上にずらす (下が空く)
        kk = -k
        out[:, :BOARD_ROWS - kk, :] = onehot[:, kk:, :]
        out[CH_EMPTY, :BOARD_ROWS - kk, :] = onehot[CH_EMPTY, kk:, :]
    # EMPTY one-hot を維持するため非 EMPTY ch が立っている cell は EMPTY ch=0
    nonempty_mask = out[1:].sum(axis=0) > 0
    out[CH_EMPTY] = np.where(nonempty_mask, 0.0, 1.0)
    return out


def augment_board(
    onehot: np.ndarray, rng: np.random.Generator,
    cfg: AugmentConfig | None = None,
) -> np.ndarray:
    """1 枚の one-hot board に augmentation を確率的に適用.

    入力 shape: (N_COLOR_CHANNELS, BOARD_ROWS, BOARD_COLS) float32.
    出力 shape: 同上 (新しい配列).
    """
    if cfg is None:
        cfg = AugmentConfig()
    out = onehot
    if rng.random() < cfg.p_color_perm:
        out = _aug_color_permute(out, rng)
    if rng.random() < cfg.p_hflip:
        out = _aug_hflip(out)
    if rng.random() < cfg.p_mask:
        out = _aug_partial_mask(out, rng)
    if rng.random() < cfg.p_shift:
        out = _aug_row_shift(out, rng)
    return out.astype(np.float32, copy=False)


def make_two_views(
    onehot: np.ndarray, rng: np.random.Generator,
    cfg: AugmentConfig | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """同じ board から独立な 2 view を作成 (SimCLR 用)."""
    return augment_board(onehot, rng, cfg), augment_board(onehot, rng, cfg)


__all__ = [
    "SimCLRBoardEncoder",
    "AugmentConfig",
    "nt_xent_loss",
    "augment_board",
    "make_two_views",
    "DEFAULT_PROJECTION_DIM",
    "DEFAULT_HIDDEN_DIM",
    "DEFAULT_TEMPERATURE",
]
