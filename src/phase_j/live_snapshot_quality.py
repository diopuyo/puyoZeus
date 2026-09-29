"""発火前窓の画素判定。画素を間引かず、OpenCVの集計で一時配列を減らす。"""
from __future__ import annotations

from typing import Any
import cv2
import numpy as np

from src.animation_filter import (DEFAULT_FRAME_DIFF_THRESHOLD, DEFAULT_V_MEAN_DELTA_THRESHOLD,
                                  DEFAULT_V_STD_DELTA_THRESHOLD, compute_region_stats)
from src.board import BOARD_ROWS, BOARD_COLS, HIDDEN_ROWS
from src.effect_glow_detector import BRIGHT_V_THRESHOLD, EFFECT_BRIGHT_RATIO_MAX_THRESHOLD

ROUNDING_TOLERANCE = 1e-9
BGR_CHANNELS = 3


class SnapshotQuality:
    """直前画像は所有権を受け取ったcapture画像のviewで保持する。"""

    def __init__(self) -> None:
        self.geometry: tuple | None = None
        self.rects: tuple | None = None
        self.shape: tuple | None = None
        self.reset()

    def reset(self) -> None:
        self.previous: np.ndarray | None = None
        self.stats: tuple[float, float] | None = None

    def observe(self, frame: np.ndarray, region: Any, hsv: np.ndarray | None = None,
                check_glow: bool = True) -> tuple[bool, bool]:
        crop = frame[region.y:region.y+region.height, region.x:region.x+region.width]
        self.prepare(crop.shape)
        value = self.value(crop, hsv)
        mean, std = cv2.meanStdDev(value)
        current = float(mean[0, 0]), float(std[0, 0])
        deltas = (0., 0.) if self.stats is None else tuple(abs(a-b) for a, b in zip(current, self.stats))
        thresholds = (DEFAULT_V_MEAN_DELTA_THRESHOLD, DEFAULT_V_STD_DELTA_THRESHOLD)
        # NumPyとOpenCVの丸めだけで閾値の上下が変わらないよう、境界だけ元の式へ戻す。
        if self.stats is not None and any(abs(d-t) < ROUNDING_TOLERANCE for d, t in zip(deltas, thresholds)):
            before = compute_region_stats(self.previous, (0, 0, region.width, region.height))
            after = compute_region_stats(crop, (0, 0, region.width, region.height))
            deltas = abs(after.v_mean-before.v_mean), abs(after.v_std-before.v_std)
        difference = (cv2.norm(self.previous, crop, cv2.NORM_L1)/crop.size
                      if self.previous is not None and self.previous.shape == crop.shape else 0.)
        quality = difference >= DEFAULT_FRAME_DIFF_THRESHOLD or any(d >= t for d, t in zip(deltas, thresholds))
        self.previous, self.stats = crop, current
        return quality, self.glow(value, region) if check_glow else False

    def value(self, crop: np.ndarray, hsv: np.ndarray | None) -> np.ndarray:
        if hsv is not None:
            return cv2.extractChannel(hsv, BGR_CHANNELS-1, self.channels[0])
        blue, green, red = cv2.split(crop, self.channels)
        # HSVのVはBGRの最大値。参照しないH/Sの算出と3ch HSV配列を省く。
        cv2.max(blue, green, blue)
        return cv2.max(blue, red, blue)

    def prepare(self, shape: tuple) -> None:
        """画素数が変わった時だけ作業領域を確保する。次フレームで全域を書き換える。"""
        if self.shape == shape:
            return
        self.shape = shape
        self.geometry = None
        self.channels = tuple(np.empty(shape[:2], dtype=np.uint8) for _ in range(BGR_CHANNELS))
        self.bright = np.empty(shape[:2], dtype=np.uint8)
        self.integral = np.empty((shape[0]+1, shape[1]+1), dtype=np.int32)

    def glow(self, value: np.ndarray, region: Any) -> bool:
        geometry = region.x, region.y, region.width, region.height
        if geometry != self.geometry:
            rects = np.array([region.cell_sample_rect(r, c)
                for r in range(HIDDEN_ROWS, BOARD_ROWS) for c in range(BOARD_COLS)])
            rects -= np.array([region.x, region.y, region.x, region.y])
            self.rects, self.geometry = tuple(rects.T), geometry
            self.prepare_samples(value)
        x1, y1, x2, y2 = self.rects
        if self.samples is not None:
            np.greater_equal(self.samples, BRIGHT_V_THRESHOLD, out=self.sample_mask)
            counts = np.count_nonzero(self.sample_mask, axis=(2, 3))
            return bool(np.any(counts/self.sample_area > EFFECT_BRIGHT_RATIO_MAX_THRESHOLD))
        bright = cv2.threshold(value, BRIGHT_V_THRESHOLD-1, 1, cv2.THRESH_BINARY, self.bright)[1]
        integral = cv2.integral(bright, self.integral)
        counts = integral[y2, x2]-integral[y1, x2]-integral[y2, x1]+integral[y1, x1]
        return bool(np.any(counts/((x2-x1)*(y2-y1)) > EFFECT_BRIGHT_RATIO_MAX_THRESHOLD))

    def prepare_samples(self, value: np.ndarray) -> None:
        """等間隔セルは標本領域のviewを作る。不等間隔ROIでは積分画像を使う。"""
        x1, y1, x2, y2 = self.rects
        rows = BOARD_ROWS-HIDDEN_ROWS
        dx, dy = x1[1]-x1[0], y1[BOARD_COLS]-y1[0]
        pw, ph = x2[0]-x1[0], y2[0]-y1[0]
        regular = (min(x1) >= 0 and min(y1) >= 0 and max(x2) <= value.shape[1] and max(y2) <= value.shape[0] and
            np.all(x2-x1 == pw) and np.all(y2-y1 == ph) and
            np.array_equal(x1, np.tile(x1[0]+np.arange(BOARD_COLS)*dx, rows)) and
            np.array_equal(y1, np.repeat(y1[0]+np.arange(rows)*dy, BOARD_COLS)))
        self.samples = None
        if regular:
            sy, sx = value.strides
            self.samples = np.lib.stride_tricks.as_strided(value[y1[0]:, x1[0]:],
                shape=(rows, BOARD_COLS, ph, pw), strides=(dy*sy, dx*sx, sy, sx), writeable=False)
            self.sample_mask = np.empty(self.samples.shape, dtype=np.bool_)
            self.sample_area = int(pw*ph)
