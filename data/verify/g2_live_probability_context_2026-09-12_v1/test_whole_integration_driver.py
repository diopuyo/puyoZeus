"""限定停止のmarker/原例外保持/原collect code保持だけを人工入力で検査する。"""
from __future__ import annotations
import json
from pathlib import Path
from types import SimpleNamespace as N
from typing import Any
import pytest
import probe_whole_integration as P


def test_stop_is_after_four_complete_callbacks(tmp_path: Path) -> None:
    events, kept = [], {}
    bridge = N(completed_frames=[], physical=N(_observed_frame_count=0))
    bridge.capture = N(snapshot=lambda: dict(observed_count=len(bridge.completed_frames)))
    def consumer(value: Any, frame: int) -> None:
        events.append(frame)
    bridge.consumer = consumer
    state = {P.A.W.KEY: bridge, 'output': tmp_path, 'actual_constructor': dict(calls=1)}
    def guard(*args: Any) -> None:
        events.append('guard')
    namespace = dict(constructor_guard=guard, Any=Any)
    exec('def original(state, collector, base, *, option=True):\n constructor_guard(None, collector, state, base)\n', namespace)
    original = namespace['original']
    patched = P.intercepted(original, kept)
    assert patched.__code__ is original.__code__
    assert patched.__kwdefaults__ is original.__kwdefaults__
    collector = N(cv2=N(VideoCapture=lambda: None))
    base = N(patch=lambda stack, obj, name, value: None)
    patched(state, collector, base)
    for frame in range(P.UPDATES - 1):
        bridge.completed_frames.append(frame)
        bridge.physical._observed_frame_count += 1
        bridge.consumer(bridge, frame)
    assert not (tmp_path / 'WHOLE_LIMITED_INPUT.json').exists()
    bridge.completed_frames.append(P.UPDATES - 1)
    bridge.physical._observed_frame_count += 1
    with pytest.raises(P.LimitedObserved, match='whole_observed_prefix_completed'):
        bridge.consumer(bridge, P.UPDATES - 1)
    row = json.loads((tmp_path / 'WHOLE_LIMITED_INPUT.json').read_text())
    assert row['physical_count'] == row['producer']['observed_count'] == P.UPDATES
    assert not row['m1_session_created'] and not row['quality_gate_clear']
    assert events == ['guard', 0, 1, 2, 3]


def test_receipt_rejects_nonfinite_before_creation(tmp_path: Path) -> None:
    path = tmp_path / 'invalid.json'
    with pytest.raises(ValueError):
        P.save_receipt(path, dict(value=float('nan')))
    assert not path.exists()


def test_receipt_does_not_overwrite(tmp_path: Path) -> None:
    path = tmp_path / 'receipt.json'
    P.save_receipt(path, dict(value=1))
    with pytest.raises(FileExistsError):
        P.save_receipt(path, dict(value=2))
    assert json.loads(path.read_bytes()) == dict(value=1)


def test_original_consumer_exception_is_not_a_planned_stop(tmp_path: Path) -> None:
    state = {P.A.W.KEY: N(), 'output': tmp_path}
    def fail(*args: Any) -> None:
        raise LookupError('original_consumer')
    state[P.A.W.KEY].consumer = fail
    namespace = dict(constructor_guard=lambda *args: None)
    exec('def original(state, collector, base):\n constructor_guard(None, collector, state, base)\n', namespace)
    P.intercepted(namespace['original'], {})(state, N(cv2=N(VideoCapture=None)), N(patch=lambda *args: None))
    with pytest.raises(LookupError, match='original_consumer'):
        state[P.A.W.KEY].consumer(None, 100)
    assert not (tmp_path / 'WHOLE_LIMITED_INPUT.json').exists()
