"""ゲーム画像を保持しない UI 蒸留器。採否は旧方式との連続比較で決める。"""
from __future__ import annotations

import json
import os
from functools import lru_cache
from typing import Any

import joblib
import numpy as np

from src.image_free_digit import digit_features

MODEL_ENV = 'PUYO_IMAGE_FREE_UI_MODEL'
MODEL_SCHEMA = 1
FRAME_SHAPE = (1080, 1920)
FRAME_REGIONS = ((100, 100, 800, 700), (1150, 100, 1850, 700),
                 (600, 300, 1320, 700), (200, 890, 1740, 1020))


def enabled() -> bool:
    from src.production_config import IMAGE_FREE_ALL_DEFAULT
    return os.environ.get('PUYO_IMAGE_FREE_ALL', str(int(IMAGE_FREE_ALL_DEFAULT))) == '1'


def features(image: np.ndarray, full: bool = True) -> np.ndarray:
    """全画面では独立領域を連結し、左右の証拠を混ぜない。"""
    patches = [image[y:Y, x:X] for x, y, X, Y in FRAME_REGIONS] if full else [image]
    return np.concatenate([digit_features(p) for p in patches])


@lru_cache(maxsize=4)
def load(path: str) -> dict:
    model = joblib.load(path)
    if model['schema'] != MODEL_SCHEMA:
        raise ValueError('UI 蒸留器の schema が不一致です')
    from src.fast_forest import accelerate
    return accelerate(model)     # 答えは sklearn と同じまま、1 回の呼び出しを速くする (11/14)


def predict(key: str, image: np.ndarray, full: bool = True) -> Any:
    path = os.environ.get(MODEL_ENV, 'models/image_free/ui.joblib')
    model = load(path)['tasks'][key]
    if hasattr(model, 'predict_image'):
        return model.predict_image(image)
    return model.predict_one(features(image, full).reshape(1, -1))


class StructuredModel:
    """離散出力と信頼度を分離し、回帰誤差で検出採否を反転させない。"""
    def __init__(self, labels: Any, scores: Any, score_keys: list[str], scalar: bool) -> None:
        self.labels, self.scores = labels, scores
        self.score_keys, self.scalar = score_keys, scalar

    def predict_one(self, data: np.ndarray) -> Any:
        result = json.loads(str(self.labels.predict(data)[0]))
        if self.scalar:
            return result
        if self.scores is not None:
            values = np.asarray(self.scores.predict(data)[0]).reshape(-1)
            result.update(zip(self.score_keys, map(float, values)))
        return result


class LearnedTemplates(dict):
    """画像の代わりにモデルの用途だけを保持する識別子。"""
    def __init__(self, key: str) -> None:
        super().__init__()
        self.key = key

    def __bool__(self) -> bool:
        return True


def learned_score(key: str, image: np.ndarray) -> float:
    return float(predict(key, image))


from src.match_end_detector import MatchEndDetector, MatchEndDetectionResult
from src.telop_detector import TelopDetector, TelopResult
from src.score_zero import ScoreZeroDetector, ScoreZeroResult
from src.win_panel import (WinPanelDetector, WinPanelResult, NUMBER_Y,
                          NUMBER_LEFT_X, NUMBER_RIGHT_X)
from src.chain_count_ocr import ChainCountOcr, ChainCountReadResult


def require_default(value: float, default: float) -> None:
    if value != default:
        raise ValueError('画像非同梱候補は教師の既定閾値だけに対応しています')


class LearnedMatchEnd(MatchEndDetector):
    def __init__(self, threshold: float, lockdown_sec: float) -> None:
        from src.match_end_detector import DEFAULT_NCC_THRESHOLD
        require_default(threshold, DEFAULT_NCC_THRESHOLD)
        super().__init__({}, threshold, lockdown_sec)

    def detect(self, frame_bgr: np.ndarray) -> MatchEndDetectionResult:
        if frame_bgr is None or not frame_bgr.size:
            return MatchEndDetectionResult(False, None, 0.)
        return MatchEndDetectionResult(**predict('match_end', frame_bgr))


class LearnedTelop(TelopDetector):
    def __init__(self, threshold: float, fast: bool | None = None) -> None:
        from src.telop_detector import DEFAULT_NCC_THRESHOLD
        require_default(threshold, DEFAULT_NCC_THRESHOLD)
        super().__init__({}, threshold, fast)

    @property
    def template_count(self) -> int:
        return len(load(os.environ.get(MODEL_ENV, 'models/image_free/ui.joblib'))['tasks'])

    def detect(self, frame_bgr: np.ndarray) -> TelopResult:
        if frame_bgr is None or not frame_bgr.size:
            return TelopResult(False, None, 0.)
        result = predict('telop', frame_bgr)
        if result['bbox'] is not None:
            result['bbox'] = tuple(result['bbox'])
        return TelopResult(**result)


class LearnedZero(ScoreZeroDetector):
    def __init__(self, threshold: float) -> None:
        from src.score_zero import ZERO_NCC_THRESHOLD
        require_default(threshold, ZERO_NCC_THRESHOLD)
        self._threshold = threshold

    def detect(self, frame: np.ndarray) -> ScoreZeroResult:
        if frame is None or frame.ndim != 3 or frame.shape[:2] != FRAME_SHAPE:
            return ScoreZeroResult(False, False, 0., 0.)
        return ScoreZeroResult(**predict('zero', frame))


class LearnedPanel(WinPanelDetector):
    def __init__(self, threshold: float) -> None:
        from src.win_panel import PANEL_NCC_THRESHOLD
        require_default(threshold, PANEL_NCC_THRESHOLD)
        self._threshold = threshold

    def detect(self, frame: np.ndarray) -> WinPanelResult:
        if frame is None or frame.ndim != 3 or frame.shape[:2] != FRAME_SHAPE:
            return WinPanelResult(False, 0.)
        result = predict('panel', frame)
        y, Y = NUMBER_Y
        left = frame[y:Y, NUMBER_LEFT_X[0]:NUMBER_LEFT_X[1]]
        right = frame[y:Y, NUMBER_RIGHT_X[0]:NUMBER_RIGHT_X[1]]
        return WinPanelResult(**result, digit_left_roi=left if result['present'] else None,
                              digit_right_roi=right if result['present'] else None)


class LearnedChain(ChainCountOcr):
    def __init__(self, min_confidence: float) -> None:
        from src.chain_count_ocr import CHAIN_NCC_MIN_CONFIDENCE
        require_default(min_confidence, CHAIN_NCC_MIN_CONFIDENCE)
        self._min_confidence = min_confidence

    def _classify(self, gray_roi: np.ndarray) -> ChainCountReadResult:
        result = predict('chain', gray_roi, False)
        if result['location'] is not None:
            result['location'] = tuple(result['location'])
        return ChainCountReadResult(**result)
