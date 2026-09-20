"""原4文の完了→実anchor型→追加trace保存/解除。画像とtrace値は明示人工。"""
from __future__ import annotations
from contextlib import ExitStack
import importlib.util
import sys
from types import MethodType
from pathlib import Path
from typing import Any
import pytest
import start_trace as T
import start_qualification as Q
import whole_collector_capture as W
from test_whole_collector_capture import fixture
from test_whole_review121 import anchor, enrich

GAME_BOUNDARY_DEBOUNCE_SEC = Q.DEBOUNCE_SEC
BOUNDARY_VISUAL_RISE_PERSIST_SEC = Q.VISUAL_PERSIST_SEC


def advance(self: Any, seconds: float) -> bool:
    return False


@pytest.mark.parametrize('failure', [False, True])
def test_capture_snapshot_and_original_failure(monkeypatch: Any, tmp_path: Path, failure: bool) -> None:
    collector, state, cls, shared, pipe, tracker, result, consumer, events = fixture(monkeypatch, tmp_path)
    base = anchor(monkeypatch)
    monkeypatch.setitem(sys.modules, 'anchor_v2', base)
    path = Path(__file__).parent / 'start_capture.py'
    spec = importlib.util.spec_from_file_location('_start_capture_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    enrich(state, pipe, result, missing=False)
    shared.advance_if_new = MethodType(advance, shared)
    shared.observe_visual_signal = MethodType(advance, shared)
    def capture(current: dict, frame: int) -> dict:
        assert current['ojama_tracker'] is tracker
        if failure:
            raise LookupError('original_getter_failure')
        return dict(frame=frame, game=0, active=True, gates=[True, True], states=['menu', 'menu'],
            scores=[0, 0], resets=[0, 0], activity={k: [0, 0] for k in Q.ACTIVITY},
            pending=[0, 0], capped=[0, 0], leftover=[0, 0], unsettled=[False, False])
    monkeypatch.setattr(T, 'capture', capture)
    def run() -> Any:
        with ExitStack() as stack:
            bridge = W.Bridge(collector, state, stack, cls, module, (100, 102), consumer)
            collector.collect_lean(shared, pipe, tracker, result)
        return bridge
    if failure:
        with pytest.raises(LookupError, match='original_getter_failure'):
            run()
        saved = __import__('json').loads((tmp_path / 'JOINT_PRODUCER_CAPTURE.failure.json').read_bytes())
        assert saved['error_type'] == 'LookupError'
    else:
        bridge = run()
        saved = state['joint_producer_capture_snapshot']
        assert len(saved['start_counter_trace']) == 2 and bridge.capture.closed
        assert saved['start_decision']['reason'] == 'visual_start_missing'
        assert saved['start_anchor'] is None
