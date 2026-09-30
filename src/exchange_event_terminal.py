"""E16専用: ばたんきゅーを観測した試合では新しい発火を受け付けない。"""
from __future__ import annotations

import os
from pathlib import Path
import cv2
import numpy as np
from src.match_end_detector import DEFAULT_NCC_THRESHOLD, SEARCH_P1, SEARCH_P2

CONFIRM_FRAMES = 2
SIDES = ("1P", "2P")
TEMPLATE = Path("models/ui_templates/match_end_batan.png")
FAST_ENV = "PUYO_FAST_TERMINAL"
# 低解像度の予備判定 (B20)。縮小率と、これ未満なら全解像度でも閾値未満とみなす床。
# 床は scripts/scan_terminal_prefilter_b20.py の全フレーム実測で、閾値以上の全フレームの
# 低解像度スコア最小値より十分低く置く (根拠は logs/live_b20/RESULT.md)。
PREFILTER_SCALE = 0.25
PREFILTER_FLOOR = 0.40
# 予備判定を通った後の全解像度照合は、粗いピーク位置の近傍だけを先に見る。近傍で閾値以上なら
# 全域の最大値も閾値以上なので決定は同一。未満の時だけ従来どおり全域を照合する。
LOCAL_MARGIN_PX = 3 * round(1 / PREFILTER_SCALE)


def confirmed_winner_probability(dead_sides: set[str]) -> float | None:
    """既存episodeの片側死亡確定方向を、勝者の確定確率へ変換する。"""
    from src.exchange_ledger import PhysicalContext
    from src.live_exchange_episode_tracker import LiveExchangeEpisodeTracker
    context = PhysicalContext(p1_dead="1P" in dead_sides, p2_dead="2P" in dead_sides)
    target = LiveExchangeEpisodeTracker._active_only_death_target(context)
    return None if target is None else float(target > 0)


class ObservedDeathDetector:
    """既存ばたんきゅーテンプレートを両側へ適用し、連続観測だけを公開する。"""
    def __init__(self, fast: bool | None = None) -> None:
        """fast=None は環境変数 PUYO_FAST_TERMINAL=1 の時だけ高速経路 (既定OFF)。"""
        self.template = cv2.imread(str(TEMPLATE), cv2.IMREAD_GRAYSCALE)
        if self.template is None:
            raise FileNotFoundError(TEMPLATE)
        self.fast = os.environ.get(FAST_ENV) == "1" if fast is None else fast
        self.small_template = _shrink(self.template) if self.fast else None
        self.streak = [0, 0]
        self.scores = [0., 0.]
        self.full_matches = [0, 0]  # 高速経路で全解像度照合まで進んだ回数 (計装)
        self.local_hits = [0, 0]    # うち近傍照合だけで閾値以上と確定した回数 (計装)
        self.frames = 0

    def _score(self, frame: np.ndarray, gray: np.ndarray | None, idx: int) -> float:
        """1側の全解像度スコア。高速経路では低解像度が床未満なら全解像度を省く。"""
        x, y, width, height = (SEARCH_P1, SEARCH_P2)[idx]
        if gray is not None:
            roi = gray[y:y+height, x:x+width]
        else:  # 高速経路は探索領域だけ灰色化する (画素ごとの演算なので値は同一)
            roi = cv2.cvtColor(frame[y:y+height, x:x+width], cv2.COLOR_BGR2GRAY)
        if self.small_template is not None:
            coarse, where = _peak_at(_shrink(roi), self.small_template)
            if coarse < PREFILTER_FLOOR:
                return coarse  # 閾値未満と確定扱い。scores は近似値になる
            self.full_matches[idx] += 1
            local = _local_peak(roi, self.template, where)
            if local >= DEFAULT_NCC_THRESHOLD:
                self.local_hits[idx] += 1
                return local   # 全域の最大値はこれ以上。scores は下界になる
        return _peak(roi, self.template)

    def update(self, frame: np.ndarray) -> tuple[str, ...]:
        """検出閾値は既存値のまま、勝敗ラベルや未来フレームを参照しない。"""
        gray = None if self.fast else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        confirmed = []
        self.frames += 1
        for idx in range(len(SIDES)):
            score = self._score(frame, gray, idx)
            self.scores[idx] = score
            self.streak[idx] = self.streak[idx]+1 if score >= DEFAULT_NCC_THRESHOLD else 0
            if self.streak[idx] >= CONFIRM_FRAMES:
                confirmed.append(SIDES[idx])
        return tuple(confirmed)


def _shrink(image: np.ndarray) -> np.ndarray:
    return cv2.resize(image, None, fx=PREFILTER_SCALE, fy=PREFILTER_SCALE,
                      interpolation=cv2.INTER_AREA)


def _peak(roi: np.ndarray, template: np.ndarray) -> float:
    return float(cv2.minMaxLoc(cv2.matchTemplate(roi, template, cv2.TM_CCOEFF_NORMED))[1])


def _peak_at(roi: np.ndarray, template: np.ndarray) -> tuple[float, tuple[int, int]]:
    """最大スコアとその位置 (x, y)。"""
    _, peak, _, where = cv2.minMaxLoc(cv2.matchTemplate(roi, template, cv2.TM_CCOEFF_NORMED))
    return float(peak), where


def _local_peak(roi: np.ndarray, template: np.ndarray, coarse_where: tuple[int, int]) -> float:
    """粗いピーク位置を全解像度へ戻した近傍だけの最大スコア (全域最大の下界)。"""
    scale = round(1 / PREFILTER_SCALE)
    height, width = template.shape
    x0 = max(0, coarse_where[0] * scale - LOCAL_MARGIN_PX)
    y0 = max(0, coarse_where[1] * scale - LOCAL_MARGIN_PX)
    x1 = min(roi.shape[1], coarse_where[0] * scale + width + LOCAL_MARGIN_PX)
    y1 = min(roi.shape[0], coarse_where[1] * scale + height + LOCAL_MARGIN_PX)
    if x1 - x0 < width or y1 - y0 < height:
        return -1.0
    return _peak(roi[y0:y1, x0:x1], template)
