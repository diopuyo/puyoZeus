"""画素完全一致のCNN確率だけを再利用する実験機構（既定OFF）。

HSV、背景、演出、盤面後処理は毎回実行する。画素差の許容量は0 DNであり、
1チャネル1階調でも変化すれば再推論する。場面別のしきい値は持たない。
"""
from __future__ import annotations

import os
from typing import Any, Callable

import numpy as np

FULL_REFRESH_SEC = 1.0
PIXEL_TOLERANCE_DN = 0
KEY_LIMIT = 312  # 1回の読取の安全上限（通常は可視144マス）。保持する世代は2つ。


class ExactPatchCache:
    """状態はCNNの外側に保持し、推論済み画素列と確率を対応させる。"""

    def __init__(self) -> None:
        self.previous: dict[tuple, np.ndarray] = {}
        self.current: dict[tuple, np.ndarray] = {}
        self.context: Any = None
        self.refreshed_at = float('-inf')
        self.last_time = float('-inf')
        self.requested = self.inferred = self.reused = 0
        self.full_reads = 0

    def begin(self, stamp: float, context: Any, temporal: bool) -> None:
        """試合・較正変更、巻戻し、間隔超過では前回分を捨てる。"""
        force = (context != self.context or stamp <= self.last_time
                 or stamp - self.refreshed_at >= FULL_REFRESH_SEC)
        self.previous = self.current if temporal and not force else {}
        self.current = {}
        if force:
            self.refreshed_at = stamp
            self.full_reads += 1
        self.context, self.last_time = context, stamp

    def predict(self, patches: list[np.ndarray], infer: Callable) -> np.ndarray:
        """同一入力の重複と前回一致を除き、必要分を元の順序でバッチ化する。"""
        if not patches:
            return infer(patches)
        keys = [(p.shape, p.dtype.str, p.tobytes()) for p in patches]
        missing: dict[tuple, np.ndarray] = {}
        for key, patch in zip(keys, patches):
            if key not in self.current and key not in self.previous:
                missing.setdefault(key, patch)
        if missing:
            values = infer(list(missing.values()))
            for key, value in zip(missing, values):
                self.current[key] = value.copy()
        result = np.stack([self.current[k] if k in self.current else self.previous[k] for k in keys])
        for key, value in zip(keys, result):
            self.current[key] = value.copy()
        self.requested += len(patches)
        self.inferred += len(missing)
        self.reused += len(patches) - len(missing)
        if len(self.current) > KEY_LIMIT:
            self.current.clear()
        return result


def read_boards(pipe: Any, frame: np.ndarray, stamp: float, **kwargs: Any) -> Any:
    """盤面CNNだけに適用し、その他の呼出・出力は従来経路を通す。"""
    reader = pipe._reader
    temporal = os.environ.get('PUYO_LIVE_DELTA_CELLS') == '1'
    dedup = os.environ.get('PUYO_LIVE_CNN_DEDUP') == '1'
    if not (temporal or dedup):
        return reader.read_both_boards(frame, **kwargs)
    hybrid = reader._classifier
    cnn = getattr(hybrid, '_cnn', None)
    if cnn is None or not hasattr(cnn, '_predict_proba_batch_uncached'):
        return reader.read_both_boards(frame, **kwargs)
    cache = getattr(pipe, '_live_delta_cache', None)
    if cache is None:
        cache = pipe._live_delta_cache = ExactPatchCache()
    context = (id(cnn), frame.shape, pipe._match_active_started_time,
               repr(getattr(getattr(hybrid, '_hsv', None), '_ranges', None)))
    cache.begin(stamp, context, temporal)
    prior = getattr(cnn, '_live_delta_cache', None)
    cnn._live_delta_cache = cache
    try:
        return reader.read_both_boards(frame, **kwargs)
    except BaseException:
        pipe._live_delta_cache = None
        raise
    finally:
        cnn._live_delta_cache = prior
