"""認識の段を排他的に計測する検証専用ラッパー。"""
from __future__ import annotations

from contextlib import ExitStack, contextmanager
from functools import wraps
import json
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Iterator
from unittest.mock import patch

import numpy as np

STAGES = ('preprocess', 'board', 'next', 'ocr', 'state', 'boundary', 'audit')
MILLISECONDS = 1000.0


class StageMeter:
    def __init__(self) -> None:
        self.rows: list[dict] = []
        self.current: dict | None = None
        self.stack: list[list] = []
        self.calls: dict[str, int] = {}

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        if self.current is None:
            yield
            return
        entry = [perf_counter(), 0.0]
        self.stack.append(entry)
        try:
            yield
        finally:
            elapsed = perf_counter()-entry[0]
            self.stack.pop()
            self.current[name] += (elapsed-entry[1])*MILLISECONDS
            if self.stack:
                self.stack[-1][1] += elapsed

    def wrap(self, function: Callable, name: str) -> Callable:
        @wraps(function)
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            key = function.__qualname__
            self.calls[key] = self.calls.get(key, 0)+1
            with self.stage(name):
                return function(*args, **kwargs)
        return wrapped

    def recognize(self, original: Callable) -> Callable:
        @wraps(original)
        def wrapped(pipe: Any, frame: Any) -> Any:
            self.current = dict(frame=frame.index, t_sec=frame.media_sec,
                                **{key: 0.0 for key in STAGES})
            start = perf_counter()
            with self.stage('audit'):
                result = original(pipe, frame)
            self.current['recognition_ms'] = (perf_counter()-start)*MILLISECONDS
            self.rows.append(self.current)
            return result
        return wrapped

    def save(self, path: Path, start: float) -> None:
        arrays = {key: np.array([row[key] for row in self.rows]) for key in self.rows[0]}
        np.savez_compressed(path/'stages.npz', **arrays)
        mask = arrays['t_sec'] >= start
        summary = {key: np.percentile(arrays[key][mask], [50, 95]).tolist()
                   for key in (*STAGES, 'recognition_ms')}
        summary['frames'] = int(mask.sum())
        (path/'stages.json').write_text(json.dumps(summary, indent=2))
        (path/'calls.json').write_text(json.dumps(self.calls, indent=2))


def install(stack: ExitStack, meter: StageMeter) -> None:
    from src import recognition_pipeline as rp
    from src.image_reader import ImageReader
    from src.next_detector import NextDetector
    from src.score_ocr import ScoreOcr
    from src.match_end_detector import MatchEndDetector
    from src.telop_detector import TelopDetector
    from src.phase_j import live_process
    from src.phase_j.live_device_session import DeviceSession
    from src.phase_j.live_audit import RecognitionAudit
    targets = [(rp.RecognitionPipeline, 'update', 'state'),
        (ImageReader, 'read_both_boards', 'board'),
        (ImageReader, 'read_board_hsv_only', 'board'),
        (NextDetector, 'detect_both', 'next'),
        (ScoreOcr, 'read_side_detail', 'ocr'),
        (ScoreOcr, 'read_side_detail_at_offset', 'ocr'),
        (ScoreOcr, 'read_formula_side', 'ocr'),
        (rp.MatchStateDetector, 'detect', 'boundary'),
        (rp.ScoreZeroDetector, 'detect', 'boundary'),
        (MatchEndDetector, 'detect', 'boundary'),
        (TelopDetector, 'detect', 'boundary'),
        (DeviceSession, 'update', 'preprocess'),
        (RecognitionAudit, 'append', 'audit'),
        (live_process.NoticeDeltaCodec, 'encode', 'audit')]
    for owner, name, stage in targets:
        stack.enter_context(patch.object(owner, name, meter.wrap(getattr(owner, name), stage)))
    stack.enter_context(patch.object(live_process, 'recognize', meter.recognize(live_process.recognize)))
