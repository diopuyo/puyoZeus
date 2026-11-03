"""追加観測だけの厳密画像キャッシュ。画素が違えば必ず元の認識器を呼ぶ。"""
from __future__ import annotations

from collections import OrderedDict
import copy
from threading import Lock
from typing import Any, Callable
import numpy as np
import torch

ICON_CACHE_SIZE = 512
SCORE_CACHE_SIZE = 32
PIXEL_MAX = 255.0
BORDERLINE_CONFIDENCE_MARGIN = 0.0001


def memoized(function: Callable, limit: int, key_for: Callable,
             thread_safe: bool = False) -> Callable:
    """近似・丸めをせず、同一画素入力の同一出力を再利用する。"""
    if thread_safe:
        return _memoized_parallel(function, limit, key_for)
    cache: OrderedDict = OrderedDict()

    def call(*args: Any, **kwargs: Any) -> Any:
        """上限付きLRU。元関数の不変なスカラー・tuple結果だけを保持する。"""
        key = key_for(*args, **kwargs)
        if key not in cache:
            cache[key] = function(*args, **kwargs)
            if len(cache) > limit:
                cache.popitem(last=False)
        else:
            cache.move_to_end(key)
        return cache[key]

    return call


def _memoized_parallel(function: Callable, limit: int, key_for: Callable) -> Callable:
    """LRU操作だけを排他化し、推論は並列実行する。退避後もローカル値を返す。"""
    cache: OrderedDict = OrderedDict()
    lock = Lock()

    def call(*args: Any, **kwargs: Any) -> Any:
        key = key_for(*args, **kwargs)
        with lock:
            if key in cache:
                cache.move_to_end(key)
                return cache[key]
        value = function(*args, **kwargs)
        with lock:
            cache[key] = value
            cache.move_to_end(key)
            if len(cache) > limit:
                cache.popitem(last=False)
        return value

    return call


def enable(pilot: Any, thread_safe: bool = False) -> None:
    """本番pipelineへは配線せず、独立した試作OCRと予告CNNだけに適用する。"""
    from src.score_ocr import _crop_score_roi

    def score_key(frame: Any, side: str, offset: tuple | None = None) -> tuple:
        """同じオフセット・側・ROI画素が揃った場合だけ再利用する。"""
        effective = pilot.ocr._region_offsets[side] if offset is None else offset
        roi = _crop_score_roi(frame, side, effective)
        return side, effective, None if roi is None else (roi.shape, roi.dtype.str, roi.tobytes())

    pilot.ocr._read_one_side_detail = memoized(pilot.ocr._read_one_side_detail, SCORE_CACHE_SIZE, score_key)
    if pilot.warning is not None and pilot.warning._cnn is not None:
        model = pilot.warning._cnn
        predict = model.predict_class_readonly if thread_safe else model.predict_class
        model.predict_class = memoized(predict, ICON_CACHE_SIZE,
            lambda patch: (patch.shape, patch.dtype.str, patch.tobytes()), thread_safe=thread_safe)


def enable_batch(pilot: Any, device: str = 'cpu') -> None:
    """試作予告CNNだけを同一フレーム内でまとめる。閾値付近は元の単体推論。"""
    from src.ojama_cnn import OJAMA_CLASSES
    from src.ojama_warning import CELL_COUNT, P1_BOARD_X, P2_BOARD_X
    warning = pilot.warning
    model = warning._cnn
    if model is None:
        return
    batch_model = model if device == 'cpu' else copy.deepcopy(model).to(device).eval()
    original_detect = warning.detect
    original_predict = model.predict_class
    values = {}

    def detect(frame: np.ndarray) -> Any:
        """予測値だけを供給し、存在判定・NCC fallbackは既存コードに任せる。"""
        patches = [warning._extract_center_patch(warning._extract_cell(frame, x, i))
                   for x in (P1_BOARD_X, P2_BOARD_X) for i in range(CELL_COUNT)]
        patches = [p for p in patches if p is not None]
        values.clear()
        if patches:
            inputs = torch.from_numpy(np.stack(patches)).float().permute(0, 3, 1, 2)/PIXEL_MAX
            with torch.inference_mode():
                probs = torch.softmax(batch_model(inputs.to(device)), dim=1)
                best, indices = probs.max(dim=1)
            for p, confidence, idx in zip(patches, best.tolist(), indices.tolist()):
                if abs(confidence-warning._cnn_confidence_min) > BORDERLINE_CONFIDENCE_MARGIN:
                    values[p.tobytes()] = (OJAMA_CLASSES[idx], confidence)
        return original_detect(frame)

    def predict(patch: np.ndarray) -> tuple[str, float]:
        """閾値付近と未対応入力は既存関数へ戻す。"""
        key = patch.tobytes()
        return values[key] if key in values else original_predict(patch)

    warning.detect = detect
    model.predict_class = predict
