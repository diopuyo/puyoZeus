"""画像を保存しない決定木による得点 OCR の実験経路（既定 OFF）。"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Callable

import cv2
import joblib
import numpy as np

from src.score_ocr import (ScoreOcr, NCC_MIN_CONFIDENCE, NCC_MARGIN_MIN,
                           NCC_AVG_MIN_CONFIDENCE, _ensure_1080p, _crop_score_roi,
                           _crop_digit_cell, _formula_invalid, parse_formula_cells,
                           FORMULA_MULT_CELL_INDEX, FormulaReadResult, DIGIT_COUNT)

FEATURE_SIZE = (20, 25)
UNKNOWN_LABEL = 10
MODEL_ENV = 'PUYO_IMAGE_FREE_DIGIT_MODEL'
DEFAULT_MODEL = Path('models/image_free/digit_student.joblib')
MODEL_SCHEMA = 2
SIDES = ('1P', '2P')
DigitDetail = tuple[int | None, float, tuple[int | None, ...], tuple[float, ...]]


def wide_multiply_features(frame: np.ndarray) -> np.ndarray:
    """R1と同じ余白付き領域を特徴化し、通常式セルと混同しない。"""
    from src.placement_signal_images import PAD_Y, PAD_X, MULT_LEFT, MULT_RIGHT
    from src.score_ocr import SCORE_1P_REGION, SCORE_2P_REGION
    image = _ensure_1080p(frame)
    if image is None:
        raise ValueError('R1の入力フレームが不正です')
    features = []
    for y1, y2, x1, _ in (SCORE_1P_REGION, SCORE_2P_REGION):
        roi = image[y1-PAD_Y:y2+PAD_Y, x1+MULT_LEFT-PAD_X:x1+MULT_RIGHT+PAD_X]
        features.append(digit_features(roi))
    return np.stack(features)


def digit_features(cell: np.ndarray) -> np.ndarray:
    """低解像度の濃淡と勾配を抽出する。画像自体はモデルに保持しない。"""
    gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY) if cell.ndim == 3 else cell
    small = cv2.resize(gray, FEATURE_SIZE, interpolation=cv2.INTER_AREA).astype(np.float32)
    # NCC と同じく明るさ・コントラストへの依存を減らす（候補 v2）。
    deviation = max(float(small.std()), 1.0)
    small = (small - small.mean()) / np.float32(deviation)
    dx = cv2.Sobel(small, cv2.CV_32F, 1, 0)
    dy = cv2.Sobel(small, cv2.CV_32F, 0, 1)
    return np.concatenate((small.ravel(), dx.ravel(), dy.ravel()))


class ImageFreeScoreOcr(ScoreOcr):
    """旧 OCR のラベルと信頼度を蒸留した候補。画像への fallback は持たない。"""

    def __init__(self, model_path: Path | None = None,
                 min_confidence: float = NCC_MIN_CONFIDENCE,
                 margin_min: float = NCC_MARGIN_MIN,
                 avg_min_confidence: float = NCC_AVG_MIN_CONFIDENCE,
                 region_offsets: dict | None = None) -> None:
        path = model_path or Path(os.environ.get(MODEL_ENV, str(DEFAULT_MODEL)))
        if min_confidence != NCC_MIN_CONFIDENCE or margin_min != NCC_MARGIN_MIN:
            raise ValueError('候補モデルは既定の教師閾値だけに対応しています')
        model = joblib.load(path)
        if model.get('schema') != MODEL_SCHEMA or model.get('feature_size') != FEATURE_SIZE:
            raise ValueError('画像非同梱 OCR モデルの形式が違います')
        self._student = model
        self._min_confidence = min_confidence
        self._margin_min = margin_min
        self._avg_min_confidence = avg_min_confidence
        self._region_offsets = {'1P': (0, 0), '2P': (0, 0)}
        self._region_offsets.update(region_offsets or {})

    def _classify_digit(self, cell: np.ndarray) -> tuple[int | None, float]:
        features = digit_features(cell).reshape(1, -1)
        label = int(self._student['labels'].predict(features)[0])
        confidence = float(self._student['confidence'].predict(features)[0])
        return (None if label == UNKNOWN_LABEL else label), confidence

    def _read_one_side_detail(
        self, frame: np.ndarray, side: str,
        offset: tuple[int, int] | None = None,
    ) -> tuple[int | None, float, tuple[int | None, ...], tuple[float, ...]]:
        """8桁を一括推論する。旧方式の欠測・平均信頼度の判定順序を保つ。"""
        effective_offset = self._region_offsets[side] if offset is None else offset
        roi = _crop_score_roi(frame, side, effective_offset)
        if roi is None or roi.size == 0:
            return None, 0.0, (None,) * DIGIT_COUNT, (0.0,) * DIGIT_COUNT
        features = np.stack([digit_features(_crop_digit_cell(roi, i, side))
                             for i in range(DIGIT_COUNT)])
        labels = self._student['labels'].predict(features)
        confidences = self._student['confidence'].predict(features)
        return self._assemble_digits(labels, confidences)

    def _assemble_digits(self, labels: np.ndarray, confidences: np.ndarray) -> DigitDetail:
        """単フレーム・連続比較の双方で同じ欠測判定を利用する。"""
        digits = tuple(None if int(v) == UNKNOWN_LABEL else int(v) for v in labels)
        confs = tuple(float(v) for v in confidences)
        minimum = min(confs)
        if any(v is None for v in digits):
            return None, minimum, digits, confs
        average = sum(confs) / DIGIT_COUNT
        if average < self._avg_min_confidence:
            return None, average, digits, confs
        score = 0
        for digit in digits:
            assert digit is not None
            score = score * 10 + digit
        return score, minimum, digits, confs

    def read_batch(self, frames: list[np.ndarray]) -> list[list[tuple[DigitDetail, FormulaReadResult]]]:
        """全フレームを保持したまま木の推論をまとめる。時系列状態は持たない。"""
        if not frames:
            return []
        features = []
        for frame in frames:
            image = _ensure_1080p(frame)
            if image is None:
                raise ValueError('連続比較の入力フレームが不正です')
            for side in SIDES:
                roi = _crop_score_roi(image, side, self._region_offsets[side])
                features.extend(digit_features(_crop_digit_cell(roi, i, side))
                                for i in range(DIGIT_COUNT))
        data = np.stack(features)
        labels = self._student['labels'].predict(data).reshape(-1, DIGIT_COUNT)
        confidence = self._student['confidence'].predict(data).reshape(-1, DIGIT_COUNT)
        multiply = self._student['multiply'].predict(data[FORMULA_MULT_CELL_INDEX::DIGIT_COUNT])
        results = []
        for i in range(len(frames)):
            sides = []
            for side in range(len(SIDES)):
                index = i * len(SIDES) + side
                detail = self._assemble_digits(labels[index], confidence[index])
                formula = parse_formula_cells(detail[2], detail[3], float(multiply[index]))
                sides.append((detail, formula))
            results.append(sides)
        return results

    @property
    def _mult_template_gray(self) -> None:
        """広い探索ROIを使うR1経路へ候補を誤接続しない。"""
        raise RuntimeError('画像非同梱OCR候補はR1置き照合の広域×検出に未対応です')

    def _match_mult_cell(self, roi: np.ndarray, side: str) -> float:
        cell = _crop_digit_cell(roi, FORMULA_MULT_CELL_INDEX, side)
        return float(self._student['multiply'].predict(digit_features(cell).reshape(1, -1))[0])

    def placement_multiply_reader(self) -> Callable[[np.ndarray], list[float]]:
        """R1用学習器の存在を初期化時に検査し、未学習を隠して続行しない。"""
        if 'wide_multiply' not in self._student:
            raise RuntimeError('画像非同梱OCRモデルにR1広域×検出器がありません')
        return self._read_wide_multiply

    def _read_wide_multiply(self, frame: np.ndarray) -> list[float]:
        values = self._student['wide_multiply'].predict(wide_multiply_features(frame))
        return [float(value) for value in values]

    def read_formula_side(self, frame: np.ndarray, side: str,
                          digit_labels: tuple | None = None,
                          digit_confs: tuple | None = None) -> FormulaReadResult:
        image = _ensure_1080p(frame)
        if image is None:
            return _formula_invalid('bad_frame')
        roi = _crop_score_roi(image, side, self._region_offsets[side])
        if roi is None or roi.size == 0:
            return _formula_invalid('bad_roi')
        if digit_labels is None or digit_confs is None:
            _, _, digit_labels, digit_confs = self._read_one_side_detail(image, side)
        return parse_formula_cells(digit_labels, digit_confs, self._match_mult_cell(roi, side))
