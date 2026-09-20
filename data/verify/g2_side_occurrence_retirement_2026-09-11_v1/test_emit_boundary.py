"""原Link.captureのupdate外拒否を再現する。原factory資格の検査は統合側で維持。"""
from __future__ import annotations
import importlib.util
from pathlib import Path
import sys
from types import MethodType, SimpleNamespace as N
from typing import Any
import pytest
import test_retirement as F

ROOT = Path(__file__).resolve().parent.parent / 'g2_directional_history_handoff_2026-09-09_v1'


def load() -> Any:
    previous = sys.modules.get('fixed')
    def module(name: str, path: Path) -> Any:
        spec = importlib.util.spec_from_file_location(name, path)
        value = importlib.util.module_from_spec(spec)
        sys.modules[name] = value
        spec.loader.exec_module(value)
        return value
    try:
        module('fixed', ROOT / 'fixed.py')
        return module('_g2_retirement_original_link', ROOT / 'evidence.py')
    finally:
        if previous is None: sys.modules.pop('fixed', None)
        else: sys.modules['fixed'] = previous


def test_original_link_rejects_reset_outside_update() -> None:
    _, adapter, runtime, _ = F.objects()
    native = F.F.native
    controller = native.NextEnqueueController(type(runtime.pipe), N(emit=lambda _: None), {})
    adapter.controller = controller
    link = object.__new__(load().Link)
    link.adapter, link.error = adapter, None
    with pytest.raises(ValueError, match='実updateと捕捉clock'):
        link.capture(runtime.pipe, '1P', F.F.FRAME, 'side_reset_retired', False)


def test_original_link_accepts_next_invocation_and_rejects_stale() -> None:
    _, adapter, runtime, _ = F.objects()
    controller = F.F.native.NextEnqueueController(type(runtime.pipe), N(), {})
    adapter.controller = controller
    runtime.pipe._pending_tsumo_1p = []
    link = object.__new__(load().Link)
    link.adapter, link.error, link.latest = adapter, None, {}
    frame = F.F.FRAME
    runtime.histories['1P'] = F.F.native.History(1, (4, 5), frame)
    adapter.states[(id(runtime.pipe), '1P')] = F.F.O.empty_state(1, 'new')
    inv = F.F.native.Invocation(runtime, frame, frame / F.F.FPS, slides={})
    controller.active = inv
    link.capture(runtime.pipe, '1P', frame, 'initial_baseline', False)
    assert link.latest['1P']['invocation'] is inv and link.latest['1P']['epoch'] == 1
    assert link.latest['1P']['frame'] == frame
    controller.active = None
    with pytest.raises(ValueError, match='実updateと捕捉clock'):
        link.capture(runtime.pipe, '1P', frame, 'initial_baseline', False)
