"""連鎖アニメーション/エフェクト中のフレーム検出 (Phase T サイクル 2)。

連鎖発火中は画面に閃光・色エフェクト・粒子が大量に発生し、盤面読取が
不安定になる。これらを検出して該当フレームの読取を「直前盤面のまま保持」
することでノイズを除去する。

検出方式:
    1. フレーム間差分: 直前フレームとの平均絶対差が閾値超 → アニメ中
    2. 盤面領域 V 急上昇: 閃光発生時は V 平均が突然上がる
    3. 盤面領域 V 標準偏差急上昇: エフェクトでムラが激しくなる

利用例:
    af = AnimationFilter()
    is_anim = af.is_animation(prev_frame, cur_frame, region)
    if is_anim:
        # 直前 board を保持
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

# フレーム間差分 (BGR) の平均絶対差。これ超でアニメ中
DEFAULT_FRAME_DIFF_THRESHOLD: float = 18.0
# 盤面領域 V 平均の急上昇 (前回との差)
DEFAULT_V_MEAN_DELTA_THRESHOLD: float = 25.0
# 盤面領域 V 標準偏差の急上昇 (前回との差)
DEFAULT_V_STD_DELTA_THRESHOLD: float = 15.0


@dataclass(frozen=True)
class FrameStats:
    """1 フレームの簡易統計。"""
    v_mean: float
    v_std: float


@dataclass(frozen=True)
class AnimationDetectionResult:
    """検出結果 + 詳細。"""
    is_animation: bool
    frame_diff: float
    v_mean_delta: float
    v_std_delta: float
    reason: str = ""


def compute_region_stats(
    frame: np.ndarray, region: tuple[int, int, int, int],
) -> FrameStats:
    """指定領域の V 平均/標準偏差を計算。"""
    x, y, w, h = region
    if frame is None or frame.size == 0:
        return FrameStats(0.0, 0.0)
    h_full, w_full = frame.shape[:2]
    x1 = max(0, min(x, w_full))
    y1 = max(0, min(y, h_full))
    x2 = max(x1 + 1, min(x + w, w_full))
    y2 = max(y1 + 1, min(y + h, h_full))
    sub = frame[y1:y2, x1:x2]
    if sub.size == 0:
        return FrameStats(0.0, 0.0)
    hsv = cv2.cvtColor(sub, cv2.COLOR_BGR2HSV)
    v = hsv[..., 2]
    return FrameStats(float(v.mean()), float(v.std()))


def compute_frame_diff(
    prev_frame: np.ndarray, cur_frame: np.ndarray,
    region: tuple[int, int, int, int] | None = None,
) -> float:
    """2 フレーム間の平均絶対差 (0-255)。region 指定で特定領域のみ。"""
    if prev_frame is None or cur_frame is None:
        return 0.0
    if prev_frame.shape != cur_frame.shape:
        return 0.0
    if region is None:
        a, b = prev_frame, cur_frame
    else:
        x, y, w, h = region
        H, W = prev_frame.shape[:2]
        x1 = max(0, min(x, W))
        y1 = max(0, min(y, H))
        x2 = max(x1 + 1, min(x + w, W))
        y2 = max(y1 + 1, min(y + h, H))
        a = prev_frame[y1:y2, x1:x2]
        b = cur_frame[y1:y2, x1:x2]
    return float(np.mean(np.abs(a.astype(np.int16) - b.astype(np.int16))))


class AnimationFilter:
    """連鎖アニメ中フレームを検出するフィルタ。"""

    def __init__(
        self,
        frame_diff_threshold: float = DEFAULT_FRAME_DIFF_THRESHOLD,
        v_mean_delta_threshold: float = DEFAULT_V_MEAN_DELTA_THRESHOLD,
        v_std_delta_threshold: float = DEFAULT_V_STD_DELTA_THRESHOLD,
    ) -> None:
        self._frame_diff_th = float(frame_diff_threshold)
        self._v_mean_delta_th = float(v_mean_delta_threshold)
        self._v_std_delta_th = float(v_std_delta_threshold)
        self._prev_stats: dict[tuple[int, int, int, int], FrameStats] = {}
        self._prev_frame: np.ndarray | None = None

    def is_animation(
        self,
        cur_frame: np.ndarray,
        region: tuple[int, int, int, int],
        prev_frame: np.ndarray | None = None,
    ) -> AnimationDetectionResult:
        """指定領域でアニメーション中か判定。

        前フレームは内部状態として保持されているため通常は省略可能。
        明示的に prev_frame を渡したい場合のみ指定。
        """
        if prev_frame is None:
            prev_frame = self._prev_frame

        cur_stats = compute_region_stats(cur_frame, region)
        prev_stats = self._prev_stats.get(region)
        v_mean_delta = (
            abs(cur_stats.v_mean - prev_stats.v_mean)
            if prev_stats is not None else 0.0
        )
        v_std_delta = (
            abs(cur_stats.v_std - prev_stats.v_std)
            if prev_stats is not None else 0.0
        )
        frame_diff = compute_frame_diff(prev_frame, cur_frame, region)

        is_anim = False
        reasons: list[str] = []
        if frame_diff >= self._frame_diff_th:
            is_anim = True
            reasons.append(f"frame_diff={frame_diff:.1f}")
        if v_mean_delta >= self._v_mean_delta_th:
            is_anim = True
            reasons.append(f"v_mean_delta={v_mean_delta:.1f}")
        if v_std_delta >= self._v_std_delta_th:
            is_anim = True
            reasons.append(f"v_std_delta={v_std_delta:.1f}")

        # 状態更新 (前フレーム/前ステートを記録)
        self._prev_stats[region] = cur_stats
        self._prev_frame = cur_frame

        return AnimationDetectionResult(
            is_animation=is_anim,
            frame_diff=frame_diff,
            v_mean_delta=v_mean_delta,
            v_std_delta=v_std_delta,
            reason=", ".join(reasons),
        )

    def reset(self) -> None:
        """状態をクリア。"""
        self._prev_stats.clear()
        self._prev_frame = None


__all__ = [
    "AnimationDetectionResult",
    "AnimationFilter",
    "DEFAULT_FRAME_DIFF_THRESHOLD",
    "DEFAULT_V_MEAN_DELTA_THRESHOLD",
    "DEFAULT_V_STD_DELTA_THRESHOLD",
    "FrameStats",
    "compute_frame_diff",
    "compute_region_stats",
]
