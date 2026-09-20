"""Cell perceptual hash anomaly 検出 (Z-3J)。

各 cell の patch を時系列で perceptual hash 化し、直前 N frame と比較。
ハッシュ距離が大きい cell は「不安定」マークし、認識結果を前 stable
frame の値で置き換えることで連鎖アニメ・落下中の評価値 swing を抑制。

「ぷよぷよは 1 cell 違いで評価値が swing する」問題への直接対処。

設計:
    - 各 cell の patch を 8x8 dHash (差分ハッシュ) 化
    - 直前 stable frame との Hamming 距離を計算
    - 距離 > THRESHOLD なら anomaly フラグ
    - anomaly cell は前 stable 値で上書き (現フレーム認識を捨てる)

連鎖中フレームは ChainPhaseDetector の予測を尊重するため対象外。
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import cv2
import numpy as np

from src.board import (
    BOARD_COLS,
    COLOR_EMPTY,
    HIDDEN_ROWS,
    Board,
)

# dHash 8x8 = 64 bit、差分閾値 (0=同一、64=完全に違う)
# 初版 12 は厳しすぎ (puyo 自然変動でも超えて全 cell anomaly 化、-7.7pt 悪化)
# 30 に緩和: puyo の照明変動は許容、連鎖アニメ・落下中の大変化のみ検出
HASH_SIZE: int = 8
HAMMING_THRESHOLD: int = 30
WINDOW: int = 3  # 直近 N frame の hash を保持


def _compute_dhash(patch: np.ndarray, hash_size: int = HASH_SIZE) -> int:
    """8x8 dHash (差分ハッシュ) を 64bit int で返す。

    - グレースケール化 → (hash_size+1) x hash_size に resize
    - 隣接 pixel の差分から 64 bit ハッシュ生成
    - 高速 (cv2.resize + numpy bit shift)
    """
    if patch.size == 0:
        return 0
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(
        gray, (hash_size + 1, hash_size),
        interpolation=cv2.INTER_AREA,
    )
    diff = resized[:, 1:] > resized[:, :-1]
    flat = diff.flatten()
    h = 0
    for i, b in enumerate(flat):
        if b:
            h |= (1 << i)
    return int(h)


def _hamming(a: int, b: int) -> int:
    """2 つの 64bit ハッシュの Hamming 距離。"""
    return int(bin(a ^ b).count("1"))


@dataclass
class CellAnomalyDetector:
    """cell の patch hash 距離で連鎖アニメ・落下中の不安定 cell を検出。

    Args:
        threshold: Hamming 距離この値超で anomaly
        window: 直近 N frame の hash 保持
    """
    threshold: int = HAMMING_THRESHOLD
    window: int = WINDOW
    # (side, vrow, col) → deque of (hash, color)
    history: dict[tuple[str, int, int], "deque[tuple[int, int]]"] = field(
        default_factory=dict,
    )

    def reset(self) -> None:
        self.history = {}

    def _get_deque(
        self, key: tuple[str, int, int],
    ) -> "deque[tuple[int, int]]":
        if key not in self.history:
            self.history[key] = deque(maxlen=self.window)
        return self.history[key]

    def refine(
        self,
        frame: np.ndarray,
        region,
        board: Board,
        side: str,
        is_chain: bool = False,
    ) -> tuple[Board, np.ndarray]:
        """frame と region から各 cell の patch hash を取得、anomaly 検出。

        Args:
            frame: 1080p BGR
            region: cell sample rect 計算用
            board: 現フレーム認識結果
            side: "1P" or "2P"
            is_chain: 連鎖中なら anomaly チェック skip (ChainSimulator 予測尊重)

        Returns:
            (refined_board, anomaly_mask) - shape=(12, BOARD_COLS) bool
        """
        out = board.copy()
        anomaly_mask = np.zeros((12, BOARD_COLS), dtype=bool)
        if is_chain:
            return out, anomaly_mask
        h, w = frame.shape[:2]
        for vrow in range(12):
            row = vrow + HIDDEN_ROWS
            for col in range(BOARD_COLS):
                key = (side, vrow, col)
                x1, y1, x2, y2 = region.cell_sample_rect(row, col)
                x1 = max(0, min(x1, w - 1))
                x2 = max(x1 + 1, min(x2, w))
                y1 = max(0, min(y1, h - 1))
                y2 = max(y1 + 1, min(y2, h))
                patch = frame[y1:y2, x1:x2]
                if patch.size == 0:
                    continue
                cur_hash = _compute_dhash(patch)
                cur_color = int(out.get(row, col))
                dq = self._get_deque(key)
                # 直近 stable hash と比較
                if len(dq) >= self.window:
                    distances = [
                        _hamming(cur_hash, h_old)
                        for h_old, _ in dq
                    ]
                    min_dist = min(distances)
                    if min_dist > self.threshold:
                        # anomaly: 直前で stable な color に戻す
                        # 直近で出現頻度最大の color を選ぶ
                        color_counts: dict[int, int] = {}
                        for _, c in dq:
                            color_counts[c] = color_counts.get(c, 0) + 1
                        stable_color, _ = max(
                            color_counts.items(), key=lambda kv: kv[1],
                        )
                        if stable_color != cur_color:
                            out.set(row, col, stable_color)
                            anomaly_mask[vrow, col] = True
                # history 更新 (anomaly でない場合のみ stable history に追加)
                if not anomaly_mask[vrow, col]:
                    dq.append((cur_hash, cur_color))
        return out, anomaly_mask


__all__ = [
    "CellAnomalyDetector",
    "HAMMING_THRESHOLD",
    "HASH_SIZE",
    "WINDOW",
]
