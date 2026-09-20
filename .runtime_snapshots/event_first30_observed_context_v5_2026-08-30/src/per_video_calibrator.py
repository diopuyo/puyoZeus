"""W11-C: Per-video 色キャリブレーション。

戦略:
    1. 試合前盤面 (全 cell EMPTY) のフレームを N 枚採取
    2. 各 cell の中心 80% 平均色 = 「この動画での EM 表現」
    3. 訓練データ (manual_labels の EM cell) の平均色 = 「訓練 EM 表現」
    4. (target - this_video) を BGR shift として保存
    5. 推論時: 各 cell パッチに shift を加算してから CNN へ

期待効果:
    動画ごとの色温度・コントラスト差を吸収、訓練分布に近づける。
    特に v18 のような暗めの照明で v7 が EM→色 hallucination する場面で
    背景の dot pattern を訓練 EM に近づけることで誤検出抑制。

シンプル実装: BGR の各 channel mean だけを揃える (additive shift)。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from src.board import (
    BOARD_COLS, BOARD_ROWS, HIDDEN_ROWS,
)
from src.image_reader import (
    BoardRegion, DEFAULT_P1_REGION, DEFAULT_P2_REGION,
)


# 訓練データの EM cell 平均 BGR (manual_plus_strict.npz の code=0、5000 cells 実測)
TRAIN_EM_BGR_DEFAULT: tuple[float, float, float] = (71.36, 73.47, 69.00)


def _patch_inner_mean(bgr: np.ndarray) -> np.ndarray:
    """中心 80% パッチの BGR 平均。"""
    h, w = bgr.shape[:2]
    crop = bgr[
        int(h * 0.1):int(h * 0.9),
        int(w * 0.1):int(w * 0.9),
    ] if bgr.size else bgr
    if crop.size == 0:
        return np.zeros(3, dtype=np.float32)
    return crop.reshape(-1, 3).mean(axis=0).astype(np.float32)


@dataclass
class PerVideoCalibrator:
    """動画ごとの BGR shift をキャリブレーションして適用する。"""

    # (B, G, R) shift: + で加算
    bgr_shift: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=np.float32),
    )
    # 訓練データの EM 平均 BGR
    train_em_bgr: np.ndarray = field(
        default_factory=lambda: np.array(
            TRAIN_EM_BGR_DEFAULT, dtype=np.float32,
        ),
    )
    n_calib_frames: int = 0

    def reset(self) -> None:
        self.bgr_shift = np.zeros(3, dtype=np.float32)
        self.n_calib_frames = 0

    def calibrate_from_frames(
        self,
        frames: list[np.ndarray],
        p1_region: BoardRegion = DEFAULT_P1_REGION,
        p2_region: BoardRegion = DEFAULT_P2_REGION,
    ) -> None:
        """試合前盤面フレーム複数枚から EM 平均色を算出 → shift。"""
        if not frames:
            return
        observed_bgrs: list[np.ndarray] = []
        for frame in frames:
            if frame.shape[:2] != (1080, 1920):
                frame = cv2.resize(
                    frame, (1920, 1080), interpolation=cv2.INTER_AREA,
                )
            for region in (p1_region, p2_region):
                for vrow in range(12):
                    row = vrow + HIDDEN_ROWS
                    for col in range(BOARD_COLS):
                        h, w = frame.shape[:2]
                        x1, y1, x2, y2 = region.cell_sample_rect(row, col)
                        x1 = max(0, min(x1, w - 1))
                        x2 = max(x1 + 1, min(x2, w))
                        y1 = max(0, min(y1, h - 1))
                        y2 = max(y1 + 1, min(y2, h))
                        patch = frame[y1:y2, x1:x2]
                        if patch.size == 0:
                            continue
                        observed_bgrs.append(_patch_inner_mean(patch))
        if not observed_bgrs:
            return
        video_em_bgr = np.stack(observed_bgrs).mean(axis=0)
        self.bgr_shift = (
            self.train_em_bgr - video_em_bgr
        ).astype(np.float32)
        self.n_calib_frames = len(frames)

    def calibrate_from_video(
        self,
        cap,  # cv2.VideoCapture
        anchor_sec: float,
        offsets_sec: tuple[float, ...] = (-0.4, -0.2, 0.0, 0.2, 0.4),
        p1_region: BoardRegion = DEFAULT_P1_REGION,
        p2_region: BoardRegion = DEFAULT_P2_REGION,
    ) -> int:
        """指定秒の前後フレームでキャリブレーション。"""
        frames: list[np.ndarray] = []
        for off in offsets_sec:
            t = max(0.0, anchor_sec + off)
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
            ok, fb = cap.read()
            if not ok or fb is None:
                continue
            frames.append(fb)
        self.calibrate_from_frames(frames, p1_region, p2_region)
        return len(frames)

    def apply(self, bgr_patch: np.ndarray) -> np.ndarray:
        """patch に shift を加算 (clip 0-255)。"""
        if bgr_patch.size == 0 or np.all(self.bgr_shift == 0):
            return bgr_patch
        out = bgr_patch.astype(np.int16) + self.bgr_shift.astype(np.int16)
        return np.clip(out, 0, 255).astype(np.uint8)


def compute_train_em_bgr(npz_path: Path) -> np.ndarray:
    """manual_labels.npz の code=0 cell の平均 BGR を計算 (一度だけ事前計算用)。"""
    d = np.load(npz_path)
    patches = d["patches"]
    labels = d["labels"]
    em_mask = labels == 0
    if em_mask.sum() == 0:
        return np.array(TRAIN_EM_BGR_DEFAULT, dtype=np.float32)
    em_patches = patches[em_mask]
    means = []
    for p in em_patches[:5000]:  # 5000 cell までで十分
        means.append(_patch_inner_mean(p))
    return np.stack(means).mean(axis=0).astype(np.float32)


__all__ = [
    "PerVideoCalibrator",
    "TRAIN_EM_BGR_DEFAULT",
    "compute_train_em_bgr",
]
