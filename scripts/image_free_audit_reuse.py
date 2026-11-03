"""監査の呼出と副作用を保ち、同一フレームの同一入力だけを再利用する。"""
from __future__ import annotations

from collections import Counter, OrderedDict
from contextlib import contextmanager, ExitStack
from copy import deepcopy
from functools import wraps
import os
from typing import Any, Iterator
from unittest.mock import patch

import numpy as np

MAX_BYTES = 64 * 1024 * 1024
PURE_CV = {'matchTemplate': 3, 'dft': 1, 'idft': 1, 'integral2': 1}


def key_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return ('array', value.dtype.str, value.shape, value.strides, value.tobytes())
    if isinstance(value, (list, tuple)):
        return tuple(key_value(v) for v in value)
    if isinstance(value, dict):
        return tuple(sorted((k, key_value(v)) for k, v in value.items()))
    return value


def byte_size(value: Any) -> int:
    if isinstance(value, np.ndarray):
        return value.nbytes
    if isinstance(value, (bytes, str)):
        return len(value)
    if isinstance(value, (list, tuple)):
        return sum(byte_size(v) for v in value)
    if isinstance(value, dict):
        return sum(byte_size(k)+byte_size(v) for k, v in value.items())
    return 32


class FrameMemo:
    def __init__(self, limit: int = MAX_BYTES) -> None:
        self.limit, self.bytes = limit, 0
        self.active = self.audit = False
        self.values: OrderedDict = OrderedDict()
        self.counts: Counter = Counter()

    def clear(self) -> None:
        self.values.clear()
        self.bytes = 0

    def call(self, label: str, key: Any, function: Any) -> Any:
        if not self.active:
            return function()
        key = (label, key)
        self.counts[label+':calls'] += 1
        if self.audit and key in self.values:
            self.counts[label+':audit_hits'] += 1
            value, _ = self.values[key]
            self.values.move_to_end(key)
            return deepcopy(value)
        value = function()
        self.counts[label+':computed'] += 1
        size = byte_size(key)+byte_size(value)
        if size <= self.limit:
            if key in self.values:
                self.bytes -= self.values.pop(key)[1]
            while self.values and self.bytes+size > self.limit:
                self.bytes -= self.values.popitem(last=False)[1][1]
            self.values[key] = (deepcopy(value), size)
            self.bytes += size
        return value


def cv_wrapper(memo: FrameMemo, name: str, original: Any) -> Any:
    @wraps(original)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        # 出力先を渡す呼出はバッファ更新の副作用があるので再利用しない。
        if (not memo.active or len(args) > PURE_CV[name]
                or set(kwargs) & {'dst', 'result', 'sum', 'sqsum', 'mask'}):
            return original(*args, **kwargs)
        key = key_value((args, kwargs))
        return memo.call('cv2.'+name, key, lambda: original(*args, **kwargs))
    return wrapped


def ocr_wrapper(memo: FrameMemo, original: Any) -> Any:
    @wraps(original)
    def wrapped(self: Any, frame: np.ndarray, side: str, offset: tuple | None = None) -> Any:
        from src.score_ocr import _crop_score_roi
        if not memo.active:
            return original(self, frame, side, offset)
        effective = self._region_offsets[side] if offset is None else offset
        settings = tuple(getattr(self, k, None) for k in ('_min_confidence', '_margin_min',
            '_avg_min_confidence', '_enable_matmul_ncc'))
        key = (id(self), side, effective, settings, frame.shape,
               key_value(_crop_score_roi(frame, side, effective)))
        return memo.call('ocr.detail', key, lambda: original(self, frame, side, offset))
    return wrapped


@contextmanager
def reuse_audit() -> Iterator[FrameMemo]:
    import cv2
    from scripts.image_free_runtime import Runtime
    from scripts import image_free_runtime_observe as observation
    from src import image_free_ui
    from src.score_ocr import ScoreOcr
    from src.image_free_digit import ImageFreeScoreOcr
    memo = FrameMemo()
    step, observe, predict = Runtime.step, observation.observe, image_free_ui.predict
    @wraps(step)
    def scoped(*args: Any, **kwargs: Any) -> Any:
        memo.clear()
        memo.active = True
        try:
            return step(*args, **kwargs)
        finally:
            memo.active = memo.audit = False
            memo.clear()
    @wraps(observe)
    def audited(*args: Any, **kwargs: Any) -> Any:
        memo.audit = True
        try:
            return observe(*args, **kwargs)
        finally:
            memo.audit = False
    @wraps(predict)
    def predicted(key: str, image: np.ndarray, full: bool = True) -> Any:
        identity = (os.environ.get(image_free_ui.MODEL_ENV), key, full, key_value(image))
        return memo.call('ui.predict', identity, lambda: predict(key, image, full))
    with ExitStack() as stack:
        for name in PURE_CV:
            stack.enter_context(patch.object(cv2, name, cv_wrapper(memo, name, getattr(cv2, name))))
        for owner in (ScoreOcr, ImageFreeScoreOcr):
            stack.enter_context(patch.object(owner, '_read_one_side_detail',
                                            ocr_wrapper(memo, owner._read_one_side_detail)))
        stack.enter_context(patch.object(Runtime, 'step', scoped))
        stack.enter_context(patch.object(observation, 'observe', audited))
        stack.enter_context(patch.object(image_free_ui, 'predict', predicted))
        yield memo
