"""原anchorの意味を保持し、別方式の開始証拠を同じ実producerから追加採録する。"""
from __future__ import annotations
from copy import deepcopy
from typing import Any
import anchor_v2 as A
import start_trace as T
import start_qualification as Q


class Capture(A.Capture):
    def __init__(self, loop: Any, tail: Any, identity: dict[str, str]) -> None:
        self.start_counter_trace: list[dict] = []
        super().__init__(loop, tail, identity)
        original = self.shared.advance_if_new.__func__
        self.debounce = original.__globals__['GAME_BOUNDARY_DEBOUNCE_SEC']
        visual = self.shared.observe_visual_signal.__func__
        self.visual_persist = visual.__globals__['BOUNDARY_VISUAL_RISE_PERSIST_SEC']
        Q.require(self.visual_persist == Q.VISUAL_PERSIST_SEC, 'original_visual_persistence')

    def completed(self, frame: int) -> None:
        try:
            super().completed(frame)
            self.start_counter_trace.append(T.capture(self.loop.runtime_state, frame))
        except BaseException as error:
            self.error = self.loop.error = error
            raise

    def snapshot(self) -> dict:
        try:
            value = super().snapshot()
            value['start_counter_trace'] = deepcopy(self.start_counter_trace)
            value['start_parameters'] = dict(debounce_sec=self.debounce, visual_persist_sec=self.visual_persist)
            value['start_decision'] = Q.decide(value, debounce_sec=self.debounce)
            return value
        except BaseException as error:
            self.error = self.loop.error = error
            raise


def install(stack: Any, loop: Any, tail: Any, identity: dict[str, str]) -> Capture:
    capture = Capture(loop, tail, identity)
    stack.callback(capture.close)
    return capture
