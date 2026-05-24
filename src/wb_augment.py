"""WB (White Balance) color augmentation for illumination invariance.

Afifi 2019 (ICCV) WB color augmenter (https://github.com/mahmoudnafifi/WB_color_augmenter, MIT)
の効果を、LAB 色空間の a/b channel への random shift で簡易再現する。

配信プラットフォーム (Twitch / YouTube) ごとに異なる auto WB のドリフトに対する
illumination 不変性を CNN に学ばせる目的で、学習時 augment chain に挟む。

設計方針:
    - pure function (stateless) — augment chain 内で別 augment と独立に呼べる
    - shift=0 で恒等変換 (regression check で活用)
    - LAB 空間の a (green-red) と b (blue-yellow) に additive shift
      ※ OpenCV LAB の a/b は uint8 で 0-255 にスケール、L は L*0..100 を 0..255
      ※ a 軸 (緑-赤): +で赤寄り, -で緑寄り
      ※ b 軸 (青-黄): +で黄寄り, -で青寄り
    - max_temp / max_tint は控えめに開始 (色境界保護)
    - 推論時は呼ばない (default off は呼出側責任)

参考: https://github.com/mahmoudnafifi/WB_color_augmenter (MIT, Afifi 2019 ICCV)
"""
from __future__ import annotations

from typing import Optional

import cv2
import numpy as np


# ============================
# 定数 (controlled magic numbers)
# ============================

# uint8 値域 [0, 255] clipping
_UINT8_MIN: int = 0
_UINT8_MAX: int = 255

# 控えめな default 範囲 (色境界が崩れて精度悪化するリスクを抑制)
DEFAULT_MAX_TEMP: float = 15.0
"""色温度 (b channel: 青-黄) の random shift 最大絶対値 (LAB uint8 単位)."""

DEFAULT_MAX_TINT: float = 10.0
"""色 tint (a channel: 緑-赤) の random shift 最大絶対値 (LAB uint8 単位)."""

# float ゼロ判定のしきい値 (恒等変換 fast path)
_EPS_SHIFT: float = 1e-6


# ============================
# public API
# ============================


def apply_wb_shift(
    bgr_patch: np.ndarray,
    temp_shift: float = 0.0,
    tint_shift: float = 0.0,
) -> np.ndarray:
    """LAB 空間の a/b channel に additive shift を加えて WB ドリフトを擬似する.

    Args:
        bgr_patch: BGR uint8 ndarray (H, W, 3).
        temp_shift: b channel (青-黄) shift, +で黄寄り / -で青寄り.
        tint_shift: a channel (緑-赤) shift, +で赤寄り / -で緑寄り.

    Returns:
        BGR uint8 ndarray (H, W, 3) with shift applied (clip to [0, 255]).
        shift が両方とも ~0 のときは入力 array の copy を返す (恒等変換).
    """
    _validate_bgr(bgr_patch)
    if (
        abs(float(temp_shift)) < _EPS_SHIFT
        and abs(float(tint_shift)) < _EPS_SHIFT
    ):
        # 恒等変換 fast path (LAB 往復の量子化誤差も避ける)
        return bgr_patch.copy()
    lab = cv2.cvtColor(bgr_patch, cv2.COLOR_BGR2LAB).astype(np.int16)
    # LAB channel: 0=L, 1=a (green-red), 2=b (blue-yellow)
    lab[:, :, 1] = np.clip(
        lab[:, :, 1] + int(round(float(tint_shift))),
        _UINT8_MIN, _UINT8_MAX,
    )
    lab[:, :, 2] = np.clip(
        lab[:, :, 2] + int(round(float(temp_shift))),
        _UINT8_MIN, _UINT8_MAX,
    )
    lab_u8 = lab.astype(np.uint8)
    bgr_out = cv2.cvtColor(lab_u8, cv2.COLOR_LAB2BGR)
    return bgr_out


def random_wb_shift(
    bgr_patch: np.ndarray,
    max_temp: float = DEFAULT_MAX_TEMP,
    max_tint: float = DEFAULT_MAX_TINT,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """[-max_temp, max_temp] / [-max_tint, max_tint] uniform から shift を引いて適用.

    Args:
        bgr_patch: BGR uint8 ndarray (H, W, 3).
        max_temp: b channel shift 絶対値上限 (>=0).
        max_tint: a channel shift 絶対値上限 (>=0).
        rng: numpy Generator (None なら default_rng() を毎回作成).

    Returns:
        BGR uint8 ndarray (H, W, 3) with random shift applied.
    """
    if max_temp < 0.0 or max_tint < 0.0:
        raise ValueError("max_temp / max_tint must be >= 0")
    g = rng if rng is not None else np.random.default_rng()
    temp_shift = float(g.uniform(-max_temp, max_temp)) if max_temp > 0.0 else 0.0
    tint_shift = float(g.uniform(-max_tint, max_tint)) if max_tint > 0.0 else 0.0
    return apply_wb_shift(
        bgr_patch, temp_shift=temp_shift, tint_shift=tint_shift,
    )


# ============================
# internal
# ============================


def _validate_bgr(patch: np.ndarray) -> None:
    """BGR uint8 (H, W, 3) のフォーマットを検証."""
    if not isinstance(patch, np.ndarray):
        raise TypeError("bgr_patch must be a numpy ndarray")
    if patch.dtype != np.uint8:
        raise TypeError(f"bgr_patch dtype must be uint8, got {patch.dtype}")
    if patch.ndim != 3 or patch.shape[2] != 3:
        raise ValueError(
            f"bgr_patch shape must be (H, W, 3), got {patch.shape}",
        )


__all__ = [
    "DEFAULT_MAX_TEMP",
    "DEFAULT_MAX_TINT",
    "apply_wb_shift",
    "random_wb_shift",
]
