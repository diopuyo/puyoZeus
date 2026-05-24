"""
CutMix augmentation for puyo patch classification.

2 つのパッチを部分的に貼り合わせて、混同しやすい色対の決定境界を強化する。
本プロジェクトでは「赤と青」「赤と紫」「青と紫」など hue 距離が近い色対の
誤認が多いため、これらを意図的に混合してモデルに「中間サンプル」を経験させる。

公式アルゴリズム:
    1. 元パッチ A と別パッチ B をランダムに選ぶ
    2. ランダムな矩形領域 R をサンプリング
    3. A の R 領域を B の同領域で置き換える
    4. ラベルは A: (1-r), B: r で混合（r = R 面積 / 全面積）
       学習側でソフトラベル損失を使うか、ラベル A or B を r 確率で確率的選択

本モジュールでは「ソフトラベルなし、確率的ハードラベル」方式を採用:
    - r >= 0.5 なら label = B、 r < 0.5 なら label = A
    - 既存 cross_entropy 損失と互換

使い方:
    from src.cutmix import cutmix_pair, generate_cutmix_samples

    # 単発: 2 パッチを混ぜる
    mixed_patch, mixed_label = cutmix_pair(patch_a, label_a, patch_b, label_b)

    # バッチ生成: 既存サンプル群から N 個合成
    extra_samples = generate_cutmix_samples(
        samples,
        n_extra=100,
        focus_pairs={(COLOR_RED, COLOR_BLUE), (COLOR_RED, COLOR_PURPLE)},
    )
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

# ============================
# 設定
# ============================

# CutMix 矩形のサイズ範囲（全体に対する比率）
DEFAULT_RATIO_RANGE: tuple[float, float] = (0.25, 0.6)

# ハードラベル切替閾値: 矩形が画素全体の何割を占めたら label を B に切替えるか
HARD_LABEL_SWAP_THRESHOLD: float = 0.5


@dataclass(frozen=True)
class _Rect:
    y0: int
    x0: int
    y1: int
    x1: int

    @property
    def area(self) -> int:
        return max(0, self.y1 - self.y0) * max(0, self.x1 - self.x0)


def _random_rect(h: int, w: int, ratio_range: tuple[float, float]) -> _Rect:
    r_lo, r_hi = ratio_range
    ratio = random.uniform(r_lo, r_hi)
    side = int((h * w * ratio) ** 0.5)
    side = max(1, min(side, min(h, w) - 1))
    y0 = random.randint(0, h - side)
    x0 = random.randint(0, w - side)
    return _Rect(y0=y0, x0=x0, y1=y0 + side, x1=x0 + side)


def cutmix_pair(
    patch_a: np.ndarray,
    label_a: int,
    patch_b: np.ndarray,
    label_b: int,
    ratio_range: tuple[float, float] = DEFAULT_RATIO_RANGE,
    swap_threshold: float = HARD_LABEL_SWAP_THRESHOLD,
) -> tuple[np.ndarray, int]:
    """
    2 パッチを CutMix し、ハードラベル付きの合成パッチを返す。

    Args:
        patch_a, patch_b: BGR uint8 パッチ（同一 shape 必須）。
        label_a, label_b: 元ラベル。
        ratio_range: 矩形面積比のサンプリング範囲。
        swap_threshold: 矩形が全面積のこの比率以上なら label を B に切替。

    Returns:
        (mixed_patch, mixed_label)
    """
    if patch_a.shape != patch_b.shape:
        raise ValueError(f"shape 不一致: {patch_a.shape} vs {patch_b.shape}")
    h, w = patch_a.shape[:2]
    rect = _random_rect(h, w, ratio_range)
    mixed = patch_a.copy()
    mixed[rect.y0:rect.y1, rect.x0:rect.x1] = patch_b[rect.y0:rect.y1, rect.x0:rect.x1]
    area_ratio = rect.area / (h * w)
    label = label_b if area_ratio >= swap_threshold else label_a
    return mixed, label


def generate_cutmix_arrays(
    patches: np.ndarray,
    labels: np.ndarray,
    n_extra: int,
    focus_pairs: set[tuple[int, int]] | None = None,
    ratio_range: tuple[float, float] = DEFAULT_RATIO_RANGE,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    numpy 配列ベースの CutMix サンプル生成（学習スクリプトから直接使う用）。

    Args:
        patches: (N, H, W, 3) uint8 BGR
        labels: (N,) int
        n_extra: 生成する合成サンプル数
        focus_pairs: 重点的に混ぜたい色対の集合（None なら全色対）
        ratio_range: cutmix 矩形比率
        seed: 再現性用

    Returns:
        (mixed_patches (n_extra, H, W, 3), mixed_labels (n_extra,))
        生成失敗時は (空配列, 空配列)
    """
    if len(patches) == 0:
        return (
            np.zeros((0,) + patches.shape[1:], dtype=patches.dtype),
            np.zeros((0,), dtype=labels.dtype),
        )
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    # ラベル → インデックス配列
    by_color: dict[int, np.ndarray] = {}
    for c in np.unique(labels):
        by_color[int(c)] = np.where(labels == c)[0]

    available_colors = list(by_color.keys())
    if len(available_colors) < 2:
        return (
            np.zeros((0,) + patches.shape[1:], dtype=patches.dtype),
            np.zeros((0,), dtype=labels.dtype),
        )

    if focus_pairs is None:
        pairs = []
        for i, c1 in enumerate(available_colors):
            for c2 in available_colors[i + 1:]:
                pairs.append((c1, c2))
    else:
        pairs = [
            (a, b) for (a, b) in focus_pairs
            if a in by_color and b in by_color
        ]
        if not pairs:
            return (
                np.zeros((0,) + patches.shape[1:], dtype=patches.dtype),
                np.zeros((0,), dtype=labels.dtype),
            )

    out_patches = np.empty((n_extra,) + patches.shape[1:], dtype=patches.dtype)
    out_labels = np.empty((n_extra,), dtype=labels.dtype)
    for k in range(n_extra):
        c_a, c_b = random.choice(pairs)
        if random.random() < 0.5:
            c_a, c_b = c_b, c_a
        i_a = int(np.random.choice(by_color[c_a]))
        i_b = int(np.random.choice(by_color[c_b]))
        mixed_patch, mixed_label = cutmix_pair(
            patches[i_a], int(labels[i_a]),
            patches[i_b], int(labels[i_b]),
            ratio_range=ratio_range,
        )
        out_patches[k] = mixed_patch
        out_labels[k] = mixed_label
    return out_patches, out_labels


def generate_cutmix_samples(
    samples: Sequence,  # PatchSample 互換: .patch (np.ndarray), .color (int)
    n_extra: int,
    focus_pairs: set[tuple[int, int]] | None = None,
    ratio_range: tuple[float, float] = DEFAULT_RATIO_RANGE,
    seed: int | None = None,
) -> list:
    """
    既存サンプル群から CutMix 合成サンプルを生成する。

    Args:
        samples: 元サンプル群（.patch, .color プロパティが必要）。
        n_extra: 生成する合成サンプル数。
        focus_pairs: 重点的に混ぜたい色対の集合。None なら全色対をランダムに。
            例: {(COLOR_RED, COLOR_BLUE), (COLOR_RED, COLOR_PURPLE)}
        ratio_range: cutmix 矩形比率。
        seed: 再現性確保用。

    Returns:
        list[PatchSample]: 元サンプルと同じ class を持つ合成サンプル。
            （PatchSample のクラスは samples の最初の要素から推定）
    """
    if not samples:
        return []
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    sample_cls = type(samples[0])

    by_color: dict[int, list] = {}
    for s in samples:
        by_color.setdefault(s.color, []).append(s)

    available_colors = list(by_color.keys())
    if len(available_colors) < 2:
        return []

    # focus_pairs が指定されていればそれだけ、なければ全色対
    if focus_pairs is None:
        pairs = []
        for i, c1 in enumerate(available_colors):
            for c2 in available_colors[i + 1:]:
                pairs.append((c1, c2))
    else:
        pairs = [
            (a, b) for (a, b) in focus_pairs
            if a in by_color and b in by_color
        ]
        if not pairs:
            return []

    out: list = []
    for _ in range(n_extra):
        c_a, c_b = random.choice(pairs)
        if random.random() < 0.5:
            c_a, c_b = c_b, c_a
        s_a = random.choice(by_color[c_a])
        s_b = random.choice(by_color[c_b])
        mixed_patch, mixed_label = cutmix_pair(
            s_a.patch, s_a.color, s_b.patch, s_b.color, ratio_range=ratio_range,
        )
        out.append(sample_cls(patch=mixed_patch, color=mixed_label))
    return out
