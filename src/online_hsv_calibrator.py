"""未知動画でのリアルタイム HSV 範囲自動学習 (Phase Z-3I / Phase X 候補)。

試合進行中に「信頼できる puyo cell」の HSV 統計を蓄積し、動画別の
puyo 色 HSV 範囲を自動算出する。training に無い動画でも動画再生中に
HSV パラメータが調整されて 99.9% 認識率を維持する仕組み。

設計:
    - 各 frame で「信頼できる cell」を抽出して色別に HSV を蓄積
    - 信頼条件: CNN 確信度 ≥ HIGH_CONF + HSV 単独判定一致
    - 色別に EMA (指数移動平均) で HSV 平均を更新
    - サンプル数 ≥ MIN_SAMPLES に達した色は動画別 ranges に反映
    - 動画別 ranges は ColorClassifier に注入して以降の判定に使用

計算量:
    - 1 frame あたり cell 数 = 144、信頼サンプル抽出は O(N)
    - 統計更新は O(1) (EMA)、色別ranges 更新は O(色数 × 範囲計算) で軽量

使用例:
    calib = OnlineHsvCalibrator()
    for frame in frames:
        # CNN/HSV 結果を渡す
        calib.update(frame, board, cnn_proba_grid, hsv_color_grid)
    if calib.is_ready():
        # 動画別 ranges を取得して ColorClassifier に注入
        ranges = calib.get_per_video_ranges()
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from src.board import (
    BOARD_COLS,
    BOARD_ROWS,
    COLOR_BLUE,
    COLOR_EMPTY,
    COLOR_GREEN,
    COLOR_OJAMA,
    COLOR_PURPLE,
    COLOR_RED,
    COLOR_UNKNOWN,
    COLOR_YELLOW,
    HIDDEN_ROWS,
)

# 信頼サンプル条件
# Z-3I 改: cnn_proba_grid=None で呼ばれた場合の誤学習を防ぐため厳格化
HIGH_CONF: float = 0.99           # CNN 確信度 (0.95→0.99)
MIN_SAMPLES: int = 200            # 動画別 ranges 採用閾値 (50→200、初期値の安定性向上)
EMA_ALPHA: float = 0.05           # EMA 重み (0.1→0.05、ノイズ耐性向上)
RANGE_STD_MULT: float = 1.5       # 範囲幅 (2.0→1.5、過度な拡大を防ぐ)

# E (cycle 56): 赤色 H 循環バグ対策。
# 赤の H は 0-13 と 166-180 に分布するため、 samples_h が H=90 を跨いで
# 分布する場合は単純算術 mean が「H=90 (= 緑!)」 と崩壊する。
# 主要 cluster (= H<90 / H>=90 で多い方) を選択して range 計算する。
# 既存 image_reader.py:176-177 の COLOR_RED 用 2 つの HsvRange (低 H 側 + 高 H 側)
# が default で wrap-around を補完するため、 主要 cluster 単一 range の返却で十分。
H_CIRCULAR_SPLIT_THRESHOLD: int = 90


def _circular_h_range(
    h_samples: list[float], mult: float,
) -> tuple[int, int]:
    """H の循環 (= 0/180 折り返し) を考慮した (h_min, h_max) 計算.

    samples_h が H=90 を跨いで分布する場合 (= 赤系):
    - 多数派 cluster (= H<90 か H>=90 のどちらか多い方) を選択
    - 少数派は無視 (= 既存 image_reader.py:176-177 の wrap-around 範囲が補完)

    span < H_CIRCULAR_SPLIT_THRESHOLD なら通常の mean ± std × mult 計算.

    Args:
        h_samples: 直近 H サンプル (= 最大 200 件保持の samples_h)
        mult: range 幅倍率 (= RANGE_STD_MULT)

    Returns:
        (h_min, h_max) 0..180 にクリップ済
    """
    h_arr = np.array(h_samples)
    span = float(h_arr.max() - h_arr.min())
    if span < H_CIRCULAR_SPLIT_THRESHOLD:
        # 折り返しなし: 通常計算
        mean = float(h_arr.mean())
        std = float(h_arr.std()) if len(h_arr) > 1 else 5.0
        h_min = max(0, int(mean - std * mult))
        h_max = min(180, int(mean + std * mult))
        return h_min, h_max
    # 折り返しあり: 多数派 cluster を選択
    low = h_arr[h_arr < H_CIRCULAR_SPLIT_THRESHOLD]
    high = h_arr[h_arr >= H_CIRCULAR_SPLIT_THRESHOLD]
    if len(low) >= len(high):
        mean = float(low.mean())
        std = float(low.std()) if len(low) > 1 else 5.0
    else:
        mean = float(high.mean())
        std = float(high.std()) if len(high) > 1 else 5.0
    h_min = max(0, int(mean - std * mult))
    h_max = min(180, int(mean + std * mult))
    return h_min, h_max

# 学習対象色 (EM/UNKNOWN は学習しない)
TRAINABLE_COLORS: tuple[int, ...] = (
    COLOR_RED, COLOR_BLUE, COLOR_GREEN, COLOR_YELLOW,
    COLOR_PURPLE, COLOR_OJAMA,
)


@dataclass
class _ColorStats:
    """色別 HSV 統計 (EMA)。"""
    h_mean: float = 0.0
    h_var: float = 0.0
    s_mean: float = 0.0
    s_var: float = 0.0
    v_mean: float = 0.0
    v_var: float = 0.0
    n: int = 0
    samples_h: list[float] = field(default_factory=list)
    samples_s: list[float] = field(default_factory=list)
    samples_v: list[float] = field(default_factory=list)

    def update(self, h: float, s: float, v: float) -> None:
        """EMA で平均/分散を更新。直近サンプルも保持 (range 算出用)。"""
        self.n += 1
        if self.n == 1:
            self.h_mean = h
            self.s_mean = s
            self.v_mean = v
            self.h_var = 0.0
            self.s_var = 0.0
            self.v_var = 0.0
        else:
            # EMA (welford-like online)
            alpha = EMA_ALPHA
            self.h_mean = (1 - alpha) * self.h_mean + alpha * h
            self.s_mean = (1 - alpha) * self.s_mean + alpha * s
            self.v_mean = (1 - alpha) * self.v_mean + alpha * v
            # 分散も EMA (近似)
            dh = h - self.h_mean
            ds = s - self.s_mean
            dv = v - self.v_mean
            self.h_var = (1 - alpha) * self.h_var + alpha * dh * dh
            self.s_var = (1 - alpha) * self.s_var + alpha * ds * ds
            self.v_var = (1 - alpha) * self.v_var + alpha * dv * dv
        # サンプル保持 (上限 200 件で sliding window)
        self.samples_h.append(h)
        self.samples_s.append(s)
        self.samples_v.append(v)
        if len(self.samples_h) > 200:
            self.samples_h = self.samples_h[-200:]
            self.samples_s = self.samples_s[-200:]
            self.samples_v = self.samples_v[-200:]

    def hsv_range(self) -> tuple[int, int, int, int, int, int]:
        """(h_min, h_max, s_min, s_max, v_min, v_max) 動画別範囲。

        mean ± std × RANGE_STD_MULT、ただし整数 + 0..255/180 にクリップ。

        E (cycle 56): H は循環構造 (= 0/180 折り返し) のため、 samples_h が
        充分な数 (>= 10) ある場合は _circular_h_range で多数派 cluster を
        選択して赤系 H の EMA 崩壊バグを回避する。
        """
        if len(self.samples_h) >= 10:
            h_min, h_max = _circular_h_range(
                self.samples_h, RANGE_STD_MULT,
            )
        else:
            h_std = float(np.sqrt(self.h_var))
            h_min = max(0, int(self.h_mean - h_std * RANGE_STD_MULT))
            h_max = min(180, int(self.h_mean + h_std * RANGE_STD_MULT))
        s_std = float(np.sqrt(self.s_var))
        v_std = float(np.sqrt(self.v_var))
        s_min = max(0, int(self.s_mean - s_std * RANGE_STD_MULT))
        s_max = min(255, int(self.s_mean + s_std * RANGE_STD_MULT))
        v_min = max(0, int(self.v_mean - v_std * RANGE_STD_MULT))
        v_max = min(255, int(self.v_mean + v_std * RANGE_STD_MULT))
        return h_min, h_max, s_min, s_max, v_min, v_max


class OnlineHsvCalibrator:
    """試合進行中に色別 HSV 範囲を自動学習する。

    update() を毎 frame 呼ぶ → 信頼サンプル抽出 → 色別統計 EMA 更新。
    is_ready() で動画別 ranges 採用可否を判定。
    """

    def __init__(
        self,
        high_conf: float = HIGH_CONF,
        min_samples: int = MIN_SAMPLES,
        require_cnn_proba: bool = True,
    ) -> None:
        self._high_conf = float(high_conf)
        self._min_samples = int(min_samples)
        # CNN proba 不要モード (mode collapse model でも機能させる用途)。
        # False の場合は HSV 一致のみで信頼サンプル判定 (2026-05-09 追加)。
        self._require_cnn_proba = bool(require_cnn_proba)
        self._stats: dict[int, _ColorStats] = {
            c: _ColorStats() for c in TRAINABLE_COLORS
        }
        self._prev_board_p1: np.ndarray | None = None
        self._prev_board_p2: np.ndarray | None = None

    def reset(self) -> None:
        for c in TRAINABLE_COLORS:
            self._stats[c] = _ColorStats()
        self._prev_board_p1 = None
        self._prev_board_p2 = None

    def update(
        self,
        frame: np.ndarray,
        region,
        board,
        cnn_proba_grid: np.ndarray | None,
        hsv_color_grid: np.ndarray | None,
        is_chain: bool = False,
    ) -> None:
        """1 frame 分の信頼サンプルを抽出して色別統計を更新。

        Args:
            frame: 1080p BGR
            region: P1 or P2 region
            board: 補正後 board
            cnn_proba_grid: shape=(12,6) CNN max softmax (なければ None)
            hsv_color_grid: shape=(12,6) HSV 単独判定 color (なければ None)
            is_chain: 連鎖中なら学習対象外 (puyo の HSV が変動するため)
        """
        if is_chain:
            return
        # hsv_color_grid は必須 (HSV 一致が信頼条件のコア)
        if hsv_color_grid is None:
            return
        # require_cnn_proba=True の場合のみ CNN proba grid 必須
        if self._require_cnn_proba and cnn_proba_grid is None:
            return
        h, w = frame.shape[:2]
        for vrow in range(12):
            row = vrow + HIDDEN_ROWS
            for col in range(BOARD_COLS):
                color = int(board.get(row, col))
                if color not in TRAINABLE_COLORS:
                    continue
                # CNN 確信度チェック (require_cnn_proba=True のみ)
                if (
                    self._require_cnn_proba
                    and cnn_proba_grid is not None
                    and float(cnn_proba_grid[vrow, col]) < self._high_conf
                ):
                    continue
                # HSV 単独判定との一致チェック
                if int(hsv_color_grid[vrow, col]) != color:
                    continue
                # patch 抽出
                x1, y1, x2, y2 = region.cell_sample_rect(row, col)
                x1 = max(0, min(x1, w - 1))
                x2 = max(x1 + 1, min(x2, w))
                y1 = max(0, min(y1, h - 1))
                y2 = max(y1 + 1, min(y2, h))
                patch = frame[y1:y2, x1:x2]
                if patch.size == 0:
                    continue
                hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
                h_med = float(np.median(hsv[:, :, 0]))
                s_med = float(np.median(hsv[:, :, 1]))
                v_med = float(np.median(hsv[:, :, 2]))
                self._stats[color].update(h_med, s_med, v_med)

    def is_ready(self, color: int | None = None) -> bool:
        """動画別 ranges を採用できる準備が整ったか。

        color=None なら全色チェック、指定色ならその色のみ。
        """
        if color is not None:
            return self._stats[color].n >= self._min_samples
        return all(
            s.n >= self._min_samples for s in self._stats.values()
        )

    def get_per_video_ranges(
        self,
    ) -> dict[int, tuple[int, int, int, int, int, int]]:
        """動画別 HSV 範囲を取得。

        Returns:
            color → (h_min, h_max, s_min, s_max, v_min, v_max)
            サンプル不足の色は除外。
        """
        out = {}
        for color, stats in self._stats.items():
            if stats.n >= self._min_samples:
                out[color] = stats.hsv_range()
        return out

    def get_sample_counts(self) -> dict[int, int]:
        """色別サンプル数 (デバッグ用)。"""
        return {c: s.n for c, s in self._stats.items()}

    def export_state(self) -> dict[str, Any]:
        """学習済 stats を JSON serializable dict として export (段階 3)。

        次回起動時に load_state() で再現可能 (動画別 ranges DB)。
        """
        return {
            "high_conf": float(self._high_conf),
            "min_samples": int(self._min_samples),
            "require_cnn_proba": bool(self._require_cnn_proba),
            "stats": {
                str(c): {
                    "h_mean": float(s.h_mean), "h_var": float(s.h_var),
                    "s_mean": float(s.s_mean), "s_var": float(s.s_var),
                    "v_mean": float(s.v_mean), "v_var": float(s.v_var),
                    "n": int(s.n),
                }
                for c, s in self._stats.items()
            },
        }

    def load_state(self, state: dict[str, Any]) -> None:
        """export_state() の逆変換。学習済 stats を復元。

        既に蓄積されたサンプル list (samples_h/s/v) は復元しない (容量節約)。
        EMA 平均/分散と n だけ復元すれば hsv_range() は計算可能。
        """
        if "stats" not in state:
            return
        for color_str, st in state["stats"].items():
            try:
                color = int(color_str)
            except (TypeError, ValueError):
                continue
            if color not in self._stats:
                continue
            cs = self._stats[color]
            cs.h_mean = float(st.get("h_mean", 0.0))
            cs.h_var = float(st.get("h_var", 0.0))
            cs.s_mean = float(st.get("s_mean", 0.0))
            cs.s_var = float(st.get("s_var", 0.0))
            cs.v_mean = float(st.get("v_mean", 0.0))
            cs.v_var = float(st.get("v_var", 0.0))
            cs.n = int(st.get("n", 0))


__all__ = [
    "OnlineHsvCalibrator",
    "HIGH_CONF",
    "MIN_SAMPLES",
    "TRAINABLE_COLORS",
]
