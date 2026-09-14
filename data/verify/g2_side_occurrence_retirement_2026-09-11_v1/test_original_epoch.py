"""v37の原prepare拒否を最小CPUで再現。修復はまだ接続しない。"""
from __future__ import annotations
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace as N
from typing import Any
import pytest
from scripts import next_enqueue_live_shadow_v1 as native

SOURCE = Path(__file__).resolve().parent.parent / 'g2_directional_next_enqueue_2026-09-09_v2/occurrence.py'
spec = importlib.util.spec_from_file_location('_g2_side_original_occurrence', SOURCE)
O = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = O
spec.loader.exec_module(O)
FRAME, FPS = 34932, 60


def objects(old_epoch: int, segment: str = 'new-segment') -> tuple[Any, Any, Any]:
    pipe = N()
    runtime = N(pipe=pipe, histories={'1P': native.History(1), '2P': native.History(0)})
    controller = N(enqueue=lambda *_: None)
    adapter = O.Adapter(native, controller, lambda *_: None, 'a' * 64, 'cpu', False)
    adapter.states[(id(pipe), '1P')] = O.empty_state(old_epoch, segment)
    invocation = N(main=N(p1=N(dnext_pair=(3, 3))))
    view = O.MotionObservation(invocation, '1P', 1, 'new-segment', FRAME, FRAME / FPS,
                               (FRAME - O.STRIDE, FRAME), None)
    return adapter, runtime, view


def test_original_prepare_rejects_unretired_old_epoch() -> None:
    adapter, runtime, view = objects(0)
    with pytest.raises(ValueError, match='^segment_changed$'):
        adapter.prepare(runtime, '1P', FRAME, (4, 5), view, True)


def test_original_prepare_accepts_consistent_fresh_epoch() -> None:
    adapter, runtime, view = objects(1)
    pair, state, committed, reason = adapter.prepare(runtime, '1P', FRAME, (4, 5), view, True)
    assert pair == (4, 5) and state['epoch'] == 1 and not committed and reason == 'initial_baseline'
