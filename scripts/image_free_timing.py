"""計算を省略せず、入れ子の処理時間を排他的に測る。"""
from __future__ import annotations

from contextlib import contextmanager, ExitStack
from functools import wraps
import time
from typing import Any, Iterator
from unittest.mock import patch

STAGES = ('decode', 'recognition', 'accounting', 'features', 'inference', 'audit', 'save')


class Timings:
    def __init__(self) -> None:
        self.seconds = dict.fromkeys(STAGES, 0.)
        self.calls = dict.fromkeys(STAGES, 0)
        self.stack: list[list] = []

    @contextmanager
    def measure(self, name: str) -> Iterator[None]:
        # observeの重複読取は、内側の検出器も含めて監査費用として数える。
        effective = 'audit' if self.stack and self.stack[-1][0] == 'audit' else name
        entry = [effective, time.perf_counter(), 0.]
        self.stack.append(entry)
        try:
            yield
        finally:
            elapsed = time.perf_counter()-entry[1]
            self.stack.pop()
            self.seconds[effective] += elapsed-entry[2]
            self.calls[effective] += 1
            if self.stack:
                self.stack[-1][2] += elapsed

    def wrapped(self, function: Any, category: str) -> Any:
        @wraps(function)
        def measured(*args: Any, **kwargs: Any) -> Any:
            with self.measure(category):
                return function(*args, **kwargs)
        return measured

    def snapshot(self) -> dict:
        return dict(seconds=dict(self.seconds), calls=dict(self.calls), exclusive=True,
            definitions='認識はRuntime.stepの残余と時系列接続、会計はcapture/会計駆動/overlay、特徴はevaluateとRealtimeRows、監査はobserve全体')


@contextmanager
def instrument(runtime: Any, timing: Timings) -> Iterator[None]:
    from scripts import image_free_runtime_observe as observation
    from scripts.image_free_runtime import Runtime
    from src.b3_observation import ObservationCapture
    from src.b3_realtime import RealtimeRows
    targets = [(Runtime, 'step', 'recognition'), (Runtime, 'evaluate', 'features'),
        (ObservationCapture, 'update', 'accounting'), (RealtimeRows, 'update', 'features'),
        (type(runtime.model), 'predict_row', 'inference'), (observation, 'observe', 'audit')]
    overlay = runtime.state.get('event_overlay')
    if overlay is not None:
        targets.append((type(overlay), 'update', 'accounting'))
    with ExitStack() as stack:
        for owner, name, category in targets:
            stack.enter_context(patch.object(owner, name, timing.wrapped(getattr(owner, name), category)))
        if '_drive_ojama' in runtime.globals:
            original = runtime.globals['_drive_ojama']
            stack.enter_context(patch.dict(runtime.globals, {'_drive_ojama': timing.wrapped(original, 'accounting')}))
        yield
