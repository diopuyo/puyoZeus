"""起動時だけ式の有効コマを集め、得点・式の共通枠を固定する。"""
from __future__ import annotations

import cv2
import numpy as np

from src.score_ocr import (
    ScoreOcr, Side, SCORE_1P_REGION, SCORE_2P_REGION, DIGIT_TOP,
    DIGIT_HEIGHT, DIGIT_WIDTH, DIGIT_LEFTS_1P, DIGIT_LEFTS_2P,
    FORMULA_MULT_CELL_INDEX, FORMULA_MULT_NCC_MIN, _ensure_1080p,
    _to_gray, parse_formula_cells,
)

SEARCH_RADIUS = 6
SAMPLE_HZ = 5
FORMULA_SAMPLES = 12
# ×の固定窓での相関: 読取可能域0.88～0.92と欠損域約0.70の間。
# 式の有効判定0.70とは別の移動許可境界。回帰結果で調整しない。
FIXED_ROI_NCC_MIN = 0.80
TIE_ATOL = 1e-6
SIDES: tuple[Side, ...] = ('1P', '2P')
CANDIDATES = tuple((dx, dy) for dy in range(-SEARCH_RADIUS, SEARCH_RADIUS+1)
                   for dx in range(-SEARCH_RADIUS, SEARCH_RADIUS+1))
ZERO_OFFSET_INDEX = CANDIDATES.index((0, 0))


def select_offset(scores: np.ndarray) -> tuple[int, int]:
    """平均NCC最大、同点は移動量・横移動量が最小の候補を選ぶ。"""
    best = float(np.max(scores))
    tied = [p for p, score in zip(CANDIDATES, scores) if best-score <= TIE_ATOL]
    return min(tied, key=lambda p: (abs(p[0])+abs(p[1]), abs(p[0]), abs(p[1]), p))


def select_final_offset(mean_scores: np.ndarray) -> tuple[int, int]:
    """固定枠で十分読める場合は、より高い相関の候補があっても動かさない。"""
    if mean_scores[ZERO_OFFSET_INDEX] >= FIXED_ROI_NCC_MIN:
        return (0, 0)
    return select_offset(mean_scores)


def candidate_scores(frame: np.ndarray, side: Side, ocr: ScoreOcr) -> np.ndarray:
    """各候補と同じ照合窓を、広げた×セルのNCCマップ一枚から得る。
    画像なし方式の OCR は切り抜きの × を持たないので、× らしさの判定器で各位置の点数を出す (11/14)。"""
    by_model = getattr(ocr, 'multiply_scores_at_offsets', None)
    if by_model is not None:
        return np.asarray(by_model(frame, side, CANDIDATES), dtype=float)
    template = ocr._mult_template_gray
    if template is None:
        return np.zeros(len(CANDIDATES))
    region = SCORE_1P_REGION if side == '1P' else SCORE_2P_REGION
    lefts = DIGIT_LEFTS_1P if side == '1P' else DIGIT_LEFTS_2P
    y, x = region[0]+DIGIT_TOP, region[2]+lefts[FORMULA_MULT_CELL_INDEX]
    radius = SEARCH_RADIUS
    cell = _to_gray(frame[y-radius:y+DIGIT_HEIGHT+radius,
                          x-radius:x+DIGIT_WIDTH+radius])
    height, width = DIGIT_HEIGHT-template.shape[0]+1, DIGIT_WIDTH-template.shape[1]+1
    if height <= 0 or width <= 0:
        return np.zeros(len(CANDIDATES))
    scores = cv2.matchTemplate(cell, template, cv2.TM_CCOEFF_NORMED)
    return np.array([scores[dy+radius:dy+radius+height,
                            dx+radius:dx+radius+width].max() for dx, dy in CANDIDATES])


class ScoreRoiAlignment:
    """試合を跨いで保持する入力単位の較正。証拠不足では永久に固定枠。"""

    def __init__(self, ocr: ScoreOcr) -> None:
        self.ocr = ocr
        self.counts = {side: 0 for side in SIDES}
        self.sums = {side: np.zeros(len(CANDIDATES)) for side in SIDES}
        self.offsets = {side: (0, 0) for side in SIDES}
        self.last_sample = float('-inf')

    @property
    def ready(self) -> bool:
        return all(n == FORMULA_SAMPLES for n in self.counts.values())

    def observe(self, frame: np.ndarray, time_sec: float) -> None:
        """得点読取りより前に呼ぶ。確定した側は以後探索しない。"""
        if self.ready or time_sec-self.last_sample < 1/SAMPLE_HZ-TIE_ATOL:
            return
        self.last_sample = time_sec
        frame = _ensure_1080p(frame)
        if frame is None:
            return
        for side in SIDES:
            if self.counts[side] < FORMULA_SAMPLES:
                self._sample(frame, side)

    def _sample(self, frame: np.ndarray, side: Side) -> None:
        scores = candidate_scores(frame, side, self.ocr)
        if float(scores.max()) < FORMULA_MULT_NCC_MIN:
            return
        dx, dy = select_offset(scores)
        _, _, labels, confs = self.ocr.read_side_detail_at_offset(frame, side, dx, dy)
        if not parse_formula_cells(labels, confs, float(scores.max())).valid:
            return
        self.sums[side] += scores
        self.counts[side] += 1
        if self.counts[side] == FORMULA_SAMPLES:
            self.offsets[side] = select_final_offset(self.sums[side]/FORMULA_SAMPLES)
            self.ocr._region_offsets[side] = self.offsets[side]

    def status(self) -> dict:
        """表示専用の状態。認識や評価の完了待ち条件にはしない。"""
        return {side: dict(ready=self.counts[side] == FORMULA_SAMPLES,
                           samples=self.counts[side], offset=list(self.offsets[side]))
                for side in SIDES}
