"""原Pipelineの重複反例と限定条件を検査。mode/Registry輸送は人工、原J検収は別。"""
from __future__ import annotations
from contextlib import ExitStack, contextmanager
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace as N
from typing import Any
import pytest
import baseline_retirement as D
import cascade_inputs as I
import test_cascade_input_path as F

S = N(encode=lambda value: value.encoded)


class Connection:
    def __init__(self, current: Any, factory: Any) -> None:
        self.binding = N(scope=current.scope)
        self.registry = N(factory=factory, current=lambda _binding: current)
        self.recovery = N(factory=factory)


def model(output: Path) -> tuple[Any, Any]:
    current = N(frame=0, scope=('source', 'run', 1, 2, 3, 0, '1P'), encoded={})
    mode = N(connection=Connection(current, object()), basis_cascade_closed=False,
        basis_origin=None, native=N(pending=[], seen_occurrences=set()), applied=[], error=None)
    return dict(output=output, probabilistic_tracking_mode=mode), current


def drive(real: Any, patch: Any, output: Path) -> Any:
    pipe, _, _, image, _ = real
    module = sys.modules[type(pipe).__module__]
    packet = json.loads(I.SOURCE.read_bytes())['state']
    board = module.Board.from_dict({'grid': packet['hidden_worlds'][0]['cells'] + packet['visible']})
    final = pipe._chain_tracker_1p._simulator.simulate(board).final_board
    clock, cap = {'frame': F.FIRST}, F.Capture(image)
    state, current = model(output)
    mode, origins = state['probabilistic_tracking_mode'], set()
    def body(context: Any, result: Any) -> None:
        for offset in range(0, I.POST_COUNT*I.STRIDE, I.STRIDE):
            clock['frame'] = F.FIRST + offset
            raw = final if offset >= I.FIRE_OFFSET + I.WINDOW_FRAMES else board
            patch.setattr(pipe._reader, 'read_both_boards', lambda *_a, **_k: (raw.copy(), module.Board()))
            pipe.update(clock['frame'], clock['frame']/F.FPS, cap.read()[1])
            origin = pipe._active_chain_1p
            if origin is not None:
                origins.add(origin.mechanism)
                if mode.basis_origin is None:
                    mode.basis_origin = dict(first_observed_frame=clock['frame'],
                        grid=tuple(map(tuple, origin.before_board.to_dict()['grid'])))
            if mode.basis_origin and origin is None and pipe._sm_1p.context.state.value == 'stable' and not mode.basis_cascade_closed:
                current.frame, current.encoded = clock['frame'], dict(visible=final.to_dict()['grid'][1:])
                mode.basis_cascade_closed = True
                mode.applied.append(dict(kind='basis_cascade', applied_frame=current.frame,
                                         state=current.encoded, source_call_token='人工輸送'))
    with ExitStack() as stack:
        value = D.install(stack, pipe, state)
        I.extend(body, F.FIRST)(dict(pipe=pipe, cap=cap, clock=clock, stack=stack), {})
    assert origins == {'formula_read'} and pipe._active_chain_1p is None
    assert pipe._sm_1p.context.state.value == 'stable' and value.used
    assert len([row for row in value.rows if row['suppressed']]) == 1
    assert 'update' not in vars(pipe._chain_tracker_1p)
    return value


def test_original_pipeline_duplicate_closed(tmp_path: Path) -> None:
    path = I.ROOT.parent / 'g2_chain_firing_continuous_cpu_2026-09-10_v1/constructor_input.py'
    spec = importlib.util.spec_from_file_location('_g2_baseline_test_ctor', path)
    constructor = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(constructor)
    evidence: dict[str, Any] = {}
    with contextmanager(F.F.F.frozen.__wrapped__)() as frozen:
        with pytest.MonkeyPatch.context() as patch:
            supplied = F.T.transport(constructor.transport(F.F.F.real.__wrapped__, F.FLAGS, evidence), evidence)
            with contextmanager(supplied)(frozen, patch) as real:
                drive(real, patch, tmp_path)
    report = json.loads((tmp_path / 'BASELINE_RETIREMENT.json').read_bytes())
    assert report['used'] and report['tracker_restored']


@pytest.mark.parametrize('fault', ['new_hand', 'late', 'changed_current', 'repeat', 'before'])
def test_normal_boundaries_not_suppressed(fault: str, tmp_path: Path) -> None:
    state, current = model(tmp_path)
    mode = state['probabilistic_tracking_mode']
    grid = tuple((0,) * 6 for _ in range(13))
    current.frame, current.encoded = 100, {'visible': []}
    mode.basis_cascade_closed, mode.basis_origin = True, dict(first_observed_frame=80, grid=grid)
    mode.applied = [dict(kind='basis_cascade', applied_frame=100, state=deepcopy(current.encoded))]
    value = object.__new__(D.Retirement)
    value.mode, value.state, value.connection = mode, state, mode.connection
    value.factory, value.used, value.tracker = mode.connection.recovery.factory, False, object()
    value.pipe = N(_chain_tracker_1p=value.tracker)
    event = N(trigger_sec=100/60, before_board=N(to_dict=lambda: {'grid': grid}))
    if fault == 'new_hand': mode.native.seen_occurrences.add('new')
    if fault == 'late': event.trigger_sec = 101/60
    if fault == 'changed_current': current.encoded = {'visible': [1]}
    if fault == 'repeat': value.used = True
    if fault == 'before': event.before_board = None
    assert value.qualified(event, 102/60) is not None


@pytest.mark.parametrize('body_fault,foreign', [(True, False), (True, True), (False, False)])
def test_cleanup_preserves_original_error(body_fault: bool, foreign: bool, monkeypatch: Any) -> None:
    class Tracker:
        def update(self, clock: float, board: Any) -> Any:
            return None
    tracker, state = Tracker(), {}
    body, saving = ValueError('body_failure'), ValueError('save_failure')
    def close() -> None:
        raise saving
    value = N(tracker=tracker, original=tracker.update, update=lambda *_: None, close=close)
    monkeypatch.setattr(D, 'Retirement', lambda *_: value)
    with pytest.raises(ValueError) as caught:
        with ExitStack() as stack:
            D.install(stack, None, state)
            if foreign: tracker.update = lambda *_: None
            if body_fault: raise body
    assert caught.value is (body if body_fault else saving)
    assert bool('update' in vars(tracker)) is foreign
    assert len(state['baseline_cleanup_errors']) == (2 if foreign else 1)
