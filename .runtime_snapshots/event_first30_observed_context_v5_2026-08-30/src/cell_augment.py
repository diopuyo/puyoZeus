"""4 色 permutation augmentation ヘルパー.

ぷよぷよでは「赤+青で連鎖」と「黄+紫で連鎖」は色置換すると等価であるため、
学習データに対して 4! = 24 通りの色 permutation を runtime augment として
適用することでデータを実質 24 倍に水増しできる。

出典: 添島・山口 2019 「深層学習を用いたぷよぷよ AI の開発」(FSS35 paper)

実装方針:
    - cell BGR patch + 既知の元色 (label) を入力に取る
    - HSV 空間で hue を「元色の代表 hue → 新色の代表 hue」へ shift
    - 元色判定不要のため pixel-wise mask は使わず、patch 全体に hue shift を適用
      (ぷよ patch は単色なので簡潔に動く)
    - EMPTY/OJAMA/UNKNOWN cell は augment 対象外: そのまま patch を返す

主要 API:
    permute_colors_in_patch(bgr_patch, src_color, color_map) -> bgr_patch_new
    generate_color_permutations(rng, include_identity=True) -> list[dict[int,int]]
    apply_color_permutation_to_label(label, color_map) -> new_label
"""
from __future__ import annotations

import itertools
from typing import Iterable

import cv2
import numpy as np

from src.board import (
    COLOR_BLUE,
    COLOR_EMPTY,
    COLOR_GREEN,
    COLOR_OJAMA,
    COLOR_PURPLE,
    COLOR_RED,
    COLOR_UNKNOWN,
    COLOR_YELLOW,
)


# ============================
# 定数: 各色の代表 hue (OpenCV HSV: 0-179)
# ============================
# `src/image_reader.py:DEFAULT_COLOR_RANGES` の中心値から導出。
# RED は折り返し領域 (h=0/180) なので 0 を採用、permutation 計算時は
# uint8 mod 180 演算で wraparound する。

REPRESENTATIVE_HUE: dict[int, int] = {
    COLOR_RED: 0,        # 0-18 と 166-180 の合成 (代表は 0)
    COLOR_YELLOW: 26,    # 14-38 の中央付近
    COLOR_GREEN: 67,     # 50-85 の中央
    COLOR_BLUE: 115,     # 100-130 の中央
    COLOR_PURPLE: 147,   # 130-165 の中央
}

# OpenCV HSV の hue 周期 (uint8: 0-179)
HUE_PERIOD: int = 180

# augment 対象 (4 色 augment 標準セット)
DEFAULT_PERMUTABLE_COLORS: tuple[int, ...] = (
    COLOR_RED,
    COLOR_BLUE,
    COLOR_GREEN,
    COLOR_YELLOW,
)

# augment 対象外 (固定): EMPTY/OJAMA/UNKNOWN
NON_PERMUTABLE_COLORS: frozenset[int] = frozenset({
    COLOR_EMPTY,
    COLOR_OJAMA,
    COLOR_UNKNOWN,
})


# ============================
# 公開 API
# ============================


def permute_colors_in_patch(
    bgr_patch: np.ndarray,
    src_color: int,
    color_map: dict[int, int],
) -> np.ndarray:
    """patch を color_map に従って色 permutation した新 patch を返す.

    Args:
        bgr_patch: shape=(H, W, 3) uint8 BGR patch。
        src_color: patch の元色 (label, COLOR_*)。
        color_map: 元色 → 新色 の写像。例: {1:2, 2:1, 3:4, 4:3}。

    Returns:
        np.ndarray: shape 同一の新 BGR patch。

    Notes:
        - src_color が augment 対象外 (EMPTY/OJAMA/UNKNOWN) ならコピーを返す。
        - color_map に src_color のエントリがなければ identity (コピー)。
        - HSV 空間で hue だけ shift し、saturation/value は不変。
    """
    if bgr_patch.size == 0:
        return bgr_patch.copy()
    if src_color in NON_PERMUTABLE_COLORS:
        return bgr_patch.copy()
    dst_color = color_map.get(src_color, src_color)
    if dst_color == src_color:
        return bgr_patch.copy()
    return _shift_hue(bgr_patch, src_color, dst_color)


def generate_color_permutations(
    rng: np.random.Generator | None = None,
    permutable_colors: Iterable[int] = DEFAULT_PERMUTABLE_COLORS,
    include_identity: bool = True,
) -> list[dict[int, int]]:
    """permutable_colors の全順列を color_map dict 列として返す.

    4 色なら 4! = 24 通り。

    Args:
        rng: 与えられれば順列順をシャッフルする (再現性目的)。
        permutable_colors: 順列対象の色集合。
        include_identity: identity (恒等写像) を含めるか。

    Returns:
        list[dict[int,int]]: color_map のリスト。
    """
    base = list(permutable_colors)
    perms = list(itertools.permutations(base))
    out: list[dict[int, int]] = []
    for perm in perms:
        cmap = {src: dst for src, dst in zip(base, perm)}
        if not include_identity and all(s == d for s, d in cmap.items()):
            continue
        out.append(cmap)
    if rng is not None:
        idx = rng.permutation(len(out))
        out = [out[int(i)] for i in idx]
    return out


def apply_color_permutation_to_label(
    label: int,
    color_map: dict[int, int],
) -> int:
    """label (元色) を color_map で写像して新 label を返す.

    NON_PERMUTABLE_COLORS (EMPTY/OJAMA/UNKNOWN) は不変。

    Args:
        label: 元色 (COLOR_*)。
        color_map: 色置換 dict。

    Returns:
        int: 新 label。
    """
    if label in NON_PERMUTABLE_COLORS:
        return int(label)
    return int(color_map.get(int(label), int(label)))


def random_color_permutation(
    rng: np.random.Generator,
    permutable_colors: Iterable[int] = DEFAULT_PERMUTABLE_COLORS,
) -> dict[int, int]:
    """rng から 1 つランダムに color_map を取得.

    Args:
        rng: numpy.random.Generator (要 seed 制御)。
        permutable_colors: 順列対象色。

    Returns:
        dict[int,int]: color_map。
    """
    base = list(permutable_colors)
    perm = rng.permutation(len(base))
    return {base[i]: base[int(perm[i])] for i in range(len(base))}


# ============================
# 内部: HSV hue shift
# ============================


def _shift_hue(
    bgr_patch: np.ndarray,
    src_color: int,
    dst_color: int,
) -> np.ndarray:
    """patch の hue を REPRESENTATIVE_HUE[src] → REPRESENTATIVE_HUE[dst] へ shift.

    OpenCV uint8 HSV (0-179) の wraparound を mod で扱う。
    saturation/value は不変。
    """
    src_hue = REPRESENTATIVE_HUE.get(src_color)
    dst_hue = REPRESENTATIVE_HUE.get(dst_color)
    if src_hue is None or dst_hue is None:
        return bgr_patch.copy()
    delta = (dst_hue - src_hue) % HUE_PERIOD
    if delta == 0:
        return bgr_patch.copy()
    hsv = cv2.cvtColor(bgr_patch, cv2.COLOR_BGR2HSV)
    h_int = hsv[:, :, 0].astype(np.int32)
    h_new = (h_int + delta) % HUE_PERIOD
    hsv[:, :, 0] = h_new.astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


__all__ = [
    "DEFAULT_PERMUTABLE_COLORS",
    "HUE_PERIOD",
    "NON_PERMUTABLE_COLORS",
    "REPRESENTATIVE_HUE",
    "apply_color_permutation_to_label",
    "generate_color_permutations",
    "permute_colors_in_patch",
    "random_color_permutation",
]
