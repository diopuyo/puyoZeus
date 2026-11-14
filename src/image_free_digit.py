"""画像を保存しない決定木による得点 OCR の実験経路（既定 OFF）。"""
from __future__ import annotations

import os
from collections import OrderedDict
from dataclasses import dataclass
import threading
from pathlib import Path
from typing import Callable

import cv2
import joblib
import numpy as np

from src.score_ocr import (ScoreOcr, NCC_MIN_CONFIDENCE, NCC_MARGIN_MIN,
                           NCC_AVG_MIN_CONFIDENCE, _ensure_1080p, _crop_score_roi,
                           _crop_digit_cell, _formula_invalid, parse_formula_cells,
                           FORMULA_MULT_CELL_INDEX, FORMULA_DIGIT_NCC_MIN,
                           FORMULA_MULT_NCC_MIN, FormulaReadResult, DIGIT_COUNT)

FEATURE_SIZE = (20, 25)
DIGIT_MEMO_SIZE = 512  # 同じ画素の桁セルの判定を使い回す件数 (1 コマに両側の得点・式を何度も読み直し、変わらない桁も多い、11/14)
UNKNOWN_LABEL = 10
MODEL_ENV = 'PUYO_IMAGE_FREE_DIGIT_MODEL'
DEFAULT_MODEL = Path('models/image_free/digit_student.joblib')
MODEL_SCHEMA = 2
SIDES = ('1P', '2P')
DigitDetail = tuple[int | None, float, tuple[int | None, ...], tuple[float, ...]]
DARK_DIGIT_MAX = 64
DARK_DIGIT_PERCENTILE = 80
RECOVERY_MIN_CONFIDENCE = NCC_AVG_MIN_CONFIDENCE
PIXEL_MAX = 255.0
FOREGROUND_GAMMAS = (2, 3)
LOCAL_CONTRAST_CLIP = 2.0
LOCAL_CONTRAST_GRID = (4, 5)
RECOVERY_PAIRS = ((0, 1, RECOVERY_MIN_CONFIDENCE), (2, 3, NCC_MIN_CONFIDENCE),
                  (4, 5, RECOVERY_MIN_CONFIDENCE), (4, 6, RECOVERY_MIN_CONFIDENCE))
FORMULA_FOREGROUND_GAMMA = 4
FORMULA_MIN_SUPPORT = 2
MULTIPLY_RECHECK_MIN = 0.60


def _normalize_dark_digit(cell: np.ndarray) -> np.ndarray:
    """暗転時の細い輪郭を、上位の明るさを基準に見える範囲へ戻す。"""
    gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY) if cell.ndim == 3 else cell
    if not 0 < gray.max() < DARK_DIGIT_MAX:
        return cell
    bright = max(float(np.percentile(gray, DARK_DIGIT_PERCENTILE)), 1.0)
    return np.uint8(np.clip(gray.astype(float) * PIXEL_MAX / bright, 0, PIXEL_MAX))


def _occlusion_views(cell: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """演出の色を抑えた像と、その暗い輪郭を持ち上げた像を作る。"""
    if cell.ndim == 3:
        hsv = cv2.cvtColor(cell, cv2.COLOR_BGR2HSV).astype(float)
        white = np.uint8(hsv[:, :, 2] * (1 - hsv[:, :, 1] / PIXEL_MAX))
    else:
        white = cell
    gamma = np.uint8(PIXEL_MAX * np.sqrt(white.astype(float) / PIXEL_MAX))
    return white, gamma


def _foreground_views(cell: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """明るい文字を残して、文字の背後にある中間の明るさを抑える。"""
    gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY) if cell.ndim == 3 else cell
    normalized = gray.astype(float) / PIXEL_MAX
    lower, upper = FOREGROUND_GAMMAS
    return np.uint8(PIXEL_MAX * normalized**lower), np.uint8(PIXEL_MAX * normalized**upper)


def _alignment_views(cell: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """局所的な明暗補正と左右一画素の位置ずれを独立に試す。"""
    gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY) if cell.ndim == 3 else cell
    enhanced = cv2.createCLAHE(clipLimit=LOCAL_CONTRAST_CLIP,
                             tileGridSize=LOCAL_CONTRAST_GRID).apply(gray)
    left = np.pad(gray[:, 1:], ((0, 0), (0, 1)), mode='edge')
    right = np.pad(gray[:, :-1], ((0, 0), (1, 0)), mode='edge')
    return enhanced, left, right


def _formula_views(cell: np.ndarray) -> tuple[np.ndarray, ...]:
    """式の既知の数字と×を、色・背景・位置を変えて再確認する。"""
    gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY) if cell.ndim == 3 else cell
    foreground = np.uint8(PIXEL_MAX * (gray.astype(float) / PIXEL_MAX)**FORMULA_FOREGROUND_GAMMA)
    return (*_occlusion_views(cell), *_foreground_views(cell), foreground, *_alignment_views(cell))


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


FEATURE_ITEMS = np.intp(FEATURE_SIZE[0] * FEATURE_SIZE[1])   # numpy の mean・std が割る個数 (intp)


def digit_features(cell: np.ndarray) -> np.ndarray:
    """低解像度の濃淡と勾配を抽出する。画像自体はモデルに保持しない。"""
    gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY) if cell.ndim == 3 else cell
    small = cv2.resize(gray, FEATURE_SIZE, interpolation=cv2.INTER_AREA).astype(np.float32)
    # NCC と同じく明るさ・コントラストへの依存を減らす（候補 v2）。
    # 平均は 1 回だけ計算する。手順と型は numpy の small.mean()・small.std() と同じ
    # (float32 の和を intp の個数で割って float32 に戻す・差の 2 乗の和も同じ・平方根は float32) なので値は同じ (11/14)。
    mean = np.float32(np.add.reduce(small, axis=None) / FEATURE_ITEMS)
    centered = small - mean
    variance = np.float32(np.add.reduce(centered * centered, axis=None) / FEATURE_ITEMS)
    deviation = max(float(np.sqrt(variance)), 1.0)
    small = centered / np.float32(deviation)
    dx = cv2.Sobel(small, cv2.CV_32F, 1, 0)
    dy = cv2.Sobel(small, cv2.CV_32F, 0, 1)
    return np.concatenate((small.ravel(), dx.ravel(), dy.ravel()))


def _cell_key(cell: np.ndarray) -> tuple:
    """セルの使い回しの鍵 (形・型・画素の並びそのもの。要約しないので取り違えがない)。"""
    return cell.shape, cell.dtype.str, cell.tobytes()


@dataclass(frozen=True)
class _SharedStudent:
    """読み込んだ判定器と、その判定器の桁の使い回しの表。"""

    model: dict
    memo: OrderedDict
    lock: threading.Lock


_SHARED_LOCK = threading.Lock()
_SHARED_STUDENTS: dict[tuple[str, int, int], _SharedStudent] = {}


def _shared_student(path: Path) -> _SharedStudent:
    """同じ判定器のファイル (場所・大きさ・更新時刻が同じ) は 1 回だけ読み、使い回しの表も部品のあいだで共有する
    (11/14: 0 点の判定と B3 の得点の読みが別の部品で、同じコマの同じマスを 2 回計算していた。同じ判定器・
    同じ画素なら答えは同じ)。"""
    stat = path.stat()
    key = (str(path.resolve()), stat.st_size, stat.st_mtime_ns)
    with _SHARED_LOCK:
        shared = _SHARED_STUDENTS.get(key)
        if shared is None:
            model = joblib.load(path)
            if model.get('schema') != MODEL_SCHEMA or model.get('feature_size') != FEATURE_SIZE:
                raise ValueError('画像非同梱 OCR モデルの形式が違います')
            from src.fast_forest import accelerate
            shared = _SHARED_STUDENTS[key] = _SharedStudent(accelerate(model), OrderedDict(), threading.Lock())
        return shared


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
        shared = _shared_student(Path(path))
        self._student = shared.model
        self._digit_memo = shared.memo        # 同じ判定器を使う部品どうしで使い回しの表を共有する
        self._memo_lock = shared.lock
        self._min_confidence = min_confidence
        self._margin_min = margin_min
        self._avg_min_confidence = avg_min_confidence
        self._region_offsets = {'1P': (0, 0), '2P': (0, 0)}
        self._region_offsets.update(region_offsets or {})

    def _labels_and_confidences(self, features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """桁の種類と確信度 (同じ特徴を受ける 2 つの森を 1 回のたどりで評価する。答えは別々と同じ)。"""
        from src.fast_forest import predict_together
        labels, confidences = predict_together([self._student['labels'], self._student['confidence']], features)
        return labels, confidences

    def _classify_digit(self, cell: np.ndarray) -> tuple[int | None, float]:
        labels, confidences = self._predict_digit_cells([cell])
        label, confidence = int(labels[0]), float(confidences[0])
        return (None if label == UNKNOWN_LABEL else label), confidence

    def _predict_digit_cells(self, cells: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        """桁セルの判定。セルごとに同じ画素なら直近の答えを使い回す
        (判定は 1 つのセルの画素だけで決まり、ほかのセルに左右されないので答えは同じ)。"""
        memo = self.__dict__.setdefault('_digit_memo', OrderedDict())
        lock = self.__dict__.setdefault('_memo_lock', threading.Lock())
        keys = [_cell_key(cell) for cell in cells]
        with lock:                            # 表の出し入れだけを守る (計算は外で行い、ほかの流れを待たせない)
            known = {key: memo[key] for key in keys if key in memo}
            for key in known:
                memo.move_to_end(key)
        missing = {key: cell for key, cell in zip(keys, cells) if key not in known}
        if missing:
            labels, confidences = self._predict_digit_cells_uncached(list(missing.values()))
            fresh = dict(zip(missing, zip(labels, confidences)))
            with lock:
                memo.update(fresh)
                while len(memo) > DIGIT_MEMO_SIZE:
                    memo.popitem(last=False)
            known.update(fresh)
        found = [known[key] for key in keys]
        return (np.array([label for label, _ in found]),   # 判定器の答えと同じ型 (numpy の値のまま並べる)
                np.array([confidence for _, confidence in found]))

    def _predict_digit_cells_uncached(self, cells: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        """暗転を補正し、欠測だけは異なる二つの補正の一致で回復する。"""
        features = np.stack([digit_features(c) for c in cells])
        labels, confidences = (array.copy() for array in self._labels_and_confidences(features))
        self._refine_dark_cells(cells, labels, confidences)
        unknown = np.flatnonzero(labels == UNKNOWN_LABEL)
        if not len(unknown):
            return labels, confidences
        views = [(*_occlusion_views(cells[i]), *_foreground_views(cells[i]),
                  *_alignment_views(cells[i])) for i in unknown]
        recovery = np.stack([digit_features(view) for group in views for view in group])
        recovered, certainty = (array.reshape(len(unknown), -1)
                                for array in self._labels_and_confidences(recovery))
        for i, values, scores in zip(unknown, recovered, certainty):
            restored = self._agree_recovery(values, scores)
            if restored is not None:
                labels[i], confidences[i] = restored
        return labels, confidences

    def _refine_dark_cells(self, cells: list[np.ndarray], labels: np.ndarray,
                           confidences: np.ndarray) -> None:
        """補正で既知の桁を欠測にせず、確信度が上がる読替えだけを使う。"""
        dark = [(i, _normalize_dark_digit(cell)) for i, cell in enumerate(cells)]
        dark = [(i, cell) for i, cell in dark if cell is not cells[i]]
        if not dark:
            return
        data = np.stack([digit_features(cell) for _, cell in dark])
        proposed, scores = self._labels_and_confidences(data)
        for (i, _), value, score in zip(dark, proposed, scores):
            if (value != UNKNOWN_LABEL and value != labels[i] and score >= NCC_MIN_CONFIDENCE
                    and (labels[i] == UNKNOWN_LABEL or score > confidences[i])):
                labels[i], confidences[i] = value, score

    @staticmethod
    def _agree_recovery(values: np.ndarray, scores: np.ndarray) -> tuple[int, float] | None:
        """色・背景・位置の各組で一致し、組どうしに数字の矛盾がない時だけ採る。"""
        candidates = [(int(values[a]), float(min(scores[a], scores[b])))
                      for a, b, minimum in RECOVERY_PAIRS
                      if values[a] != UNKNOWN_LABEL and values[a] == values[b]
                      and min(scores[a], scores[b]) >= minimum]
        if not candidates or len({value for value, _ in candidates}) != 1:
            return None
        return min(candidates, key=lambda candidate: candidate[1])

    def _formula_confidences(self, cells: list[np.ndarray], labels: tuple,
                             confidences: tuple) -> tuple[float, ...]:
        """数字自体は変えず、二通り以上で再確認できた弱い桁の確信度を補う。"""
        result = list(confidences)
        weak = [i for i, value in enumerate(labels)
                if value is not None and confidences[i] < FORMULA_DIGIT_NCC_MIN]
        if not weak:
            return tuple(result)
        groups = [_formula_views(cells[i]) for i in weak]
        features = np.stack([digit_features(view) for group in groups for view in group])
        predicted, certainty = (array.reshape(len(weak), -1)
                                for array in self._labels_and_confidences(features))
        for i, values, scores in zip(weak, predicted, certainty):
            known = [(int(value), float(score)) for value, score in zip(values, scores)
                     if value != UNKNOWN_LABEL and score >= FORMULA_DIGIT_NCC_MIN]
            if len(known) >= FORMULA_MIN_SUPPORT and {value for value, _ in known} == {labels[i]}:
                result[i] = sorted((score for _, score in known), reverse=True)[FORMULA_MIN_SUPPORT-1]
        return tuple(result)

    def _recover_multiply(self, cell: np.ndarray, score: float) -> float:
        """×の可能性がある低い値だけを再確認し、閾値そのものは変えない。"""
        if not MULTIPLY_RECHECK_MIN <= score < FORMULA_MULT_NCC_MIN:
            return score
        features = digit_features(cell).reshape(1, -1)
        if int(self._student['labels'].predict(features)[0]) != UNKNOWN_LABEL:
            return score
        data = np.stack([digit_features(view) for view in _formula_views(cell)])
        alternatives = sorted(self._student['multiply'].predict(data), reverse=True)
        supported = float(alternatives[FORMULA_MIN_SUPPORT-1])
        return max(score, supported) if supported >= FORMULA_MULT_NCC_MIN else score

    def _parse_formula(self, cells: list[np.ndarray], detail: DigitDetail,
                       multiply: float) -> FormulaReadResult:
        score = self._recover_multiply(cells[FORMULA_MULT_CELL_INDEX], multiply)
        if score < FORMULA_MULT_NCC_MIN:
            return _formula_invalid('mult', score)
        confidences = self._formula_confidences(cells, detail[2], detail[3])
        return parse_formula_cells(detail[2], confidences, score)

    def _read_one_side_detail(
        self, frame: np.ndarray, side: str,
        offset: tuple[int, int] | None = None,
    ) -> tuple[int | None, float, tuple[int | None, ...], tuple[float, ...]]:
        """8桁を一括推論する。旧方式の欠測・平均信頼度の判定順序を保つ。"""
        effective_offset = self._region_offsets[side] if offset is None else offset
        roi = _crop_score_roi(frame, side, effective_offset)
        if roi is None or roi.size == 0:
            return None, 0.0, (None,) * DIGIT_COUNT, (0.0,) * DIGIT_COUNT
        cells = [_crop_digit_cell(roi, i, side) for i in range(DIGIT_COUNT)]
        labels, confidences = self._predict_digit_cells(cells)
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
        cells = []
        for frame in frames:
            image = _ensure_1080p(frame)
            if image is None:
                raise ValueError('連続比較の入力フレームが不正です')
            for side in SIDES:
                roi = _crop_score_roi(image, side, self._region_offsets[side])
                cells.extend(_crop_digit_cell(roi, i, side) for i in range(DIGIT_COUNT))
        data = np.stack([digit_features(cell) for cell in cells])
        flat_labels, flat_confidence = self._predict_digit_cells(cells)
        labels = flat_labels.reshape(-1, DIGIT_COUNT)
        confidence = flat_confidence.reshape(-1, DIGIT_COUNT)
        multiply = self._student['multiply'].predict(data[FORMULA_MULT_CELL_INDEX::DIGIT_COUNT])
        results = []
        for i in range(len(frames)):
            sides = []
            for side in range(len(SIDES)):
                index = i * len(SIDES) + side
                detail = self._assemble_digits(labels[index], confidence[index])
                side_cells = cells[index*DIGIT_COUNT:(index+1)*DIGIT_COUNT]
                formula = self._parse_formula(side_cells, detail, float(multiply[index]))
                sides.append((detail, formula))
            results.append(sides)
        return results

    @property
    def _mult_template_gray(self) -> None:
        """広い探索ROIを使うR1経路へ候補を誤接続しない。"""
        raise RuntimeError('画像非同梱OCR候補はR1置き照合の広域×検出に未対応です')

    def multiply_scores_at_offsets(self, frame: np.ndarray, side: str,
                                   offsets: tuple[tuple[int, int], ...]) -> np.ndarray:
        """得点の欄の位置合わせ (score_roi_alignment) 用: 欄を offsets だけずらした時の × のセルの × らしさ。
        切り抜きの × との見比べ (相関の最大) を学んだ判定器 multiply で、同じ量を画像なしで出す。
        フレームの外にはみ出す位置は 0。"""
        image = _ensure_1080p(frame)
        scores = np.zeros(len(offsets))
        if image is None:
            return scores
        cells, kept = [], []
        for index, offset in enumerate(offsets):
            roi = _crop_score_roi(image, side, offset)
            if roi is not None and roi.size:
                cells.append(_crop_digit_cell(roi, FORMULA_MULT_CELL_INDEX, side))
                kept.append(index)
        if cells:
            scores[kept] = self._student['multiply'].predict(np.stack([digit_features(c) for c in cells]))
        return scores

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
        cells = [_crop_digit_cell(roi, i, side) for i in range(DIGIT_COUNT)]
        detail = (None, 0.0, digit_labels, digit_confs)
        return self._parse_formula(cells, detail, self._match_mult_cell(roi, side))
