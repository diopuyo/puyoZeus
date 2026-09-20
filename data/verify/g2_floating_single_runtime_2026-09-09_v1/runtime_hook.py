"""2Pの限定実stepへ負例退出ガードを接続する。"""
from __future__ import annotations
import functools
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parent
GUARD = ROOT.parent / 'g2_floating_single_exit_guard_2026-09-09_v1/guard.py'
GUARD_SHA = '8837cf74585c154fc03e2baee3d6932993a90000e7170a9f123522d7e1a13752'
FIRST, LAST, STRIDE, FPS, SIDE = 34620, 34702, 2, 60, '2P'
SIDECAR, STATUS, RECEIPT = 'FLOATING_EXIT.jsonl', 'FLOATING_EXIT_STATUS.json', 'FLOATING_EXIT_RECEIPT.json'
REQUIRED = {SIDECAR, STATUS, RECEIPT}
OWN = ('runtime_hook.py', 'live_cli.py', 'test_runtime.py', 'run_cpu.py', 'preflight.py', 'launcher.sh', 'CONTRACT.md')


def sha(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def require(value: bool, reason: str) -> None:
    if not value:
        raise RuntimeError(reason)


def write(path: Path, value: Any) -> None:
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False, indent=2)


def load_guard() -> Any:
    require(sha(GUARD) == GUARD_SHA, 'guard_changed')
    alias = '_floating_exit_fixed_guard'
    require(alias not in sys.modules, 'guard_alias_collision')
    spec = importlib.util.spec_from_file_location(alias, GUARD)
    module = importlib.util.module_from_spec(spec)
    sys.modules[alias] = module
    spec.loader.exec_module(module)
    return module


G = load_guard()


def guards() -> dict[str, str]:
    require(sha(GUARD) == GUARD_SHA, 'guard_source_changed')
    return {str(GUARD): GUARD_SHA} | {str(ROOT / name): sha(ROOT / name) for name in OWN}


def selected(frame: int, side: str) -> bool:
    return type(frame) is int and side == SIDE and FIRST <= frame <= LAST and frame % STRIDE == 0


class Sink:
    def __init__(self, state: Any, history: Any) -> None:
        self.output, self.history = state['output'], history
        self.rows: list[dict[str, Any]] = []
        self.errors: list[str] = []
        self.closed = False
        self.stream = (self.output / SIDECAR).open('x', encoding='utf-8')

    def emit(self, kind: str, frame: int, clock: float, **fields: Any) -> None:
        require(selected(frame, SIDE) and clock == frame / FPS, 'scope_clock')
        require((self.history.frame, self.history.time_sec) == (frame, clock), 'actual_history_clock')
        value = {'kind': kind, 'frame_idx': frame, 'time_sec': clock, 'side': SIDE, **fields}
        encoded = json.dumps(value, ensure_ascii=False, allow_nan=False)
        self.stream.write(encoded + '\n')
        self.rows.append(json.loads(encoded))

    def close(self) -> None:
        try:
            self.stream.close()
            self.closed = True
        except BaseException as exc:
            self.errors.append(repr(exc))
            raise
        finally:
            write(self.output / STATUS, {'closed': self.closed, 'errors': self.errors})


def scoped(original: Any, sink: Sink) -> Any:
    @functools.wraps(original)
    def step(pipe: Any, side: str, frame_idx: int, time_sec: float, *args: Any, **kwargs: Any) -> Any:
        if not selected(frame_idx, side):
            return original(pipe, side, frame_idx, time_sec, *args, **kwargs)
        try:
            sm = pipe._sm_2p
            require(kwargs.get('sm') is sm, 'side_sm_identity')
            detector = sys.modules['src.state_detectors'].TsumoPhaseDetector
            def emit(row: Any) -> None:
                require(row['frame_idx'] == frame_idx and row['time_sec'] == time_sec, 'guard_clock')
                sink.emit('reject', frame_idx, time_sec, guard=row)
            sink.emit('enter', frame_idx, time_sec, state=sm.context.state.value)
            with G.install(type(sm), detector, enabled=True, observer=emit) as control:
                result = original(pipe, side, frame_idx, time_sec, *args, **kwargs)
            require(control.sticky_error is None, 'caught_guard_error')
            sink.emit('return', frame_idx, time_sec, state=sm.context.state.value,
                original_step_calls=1, rejects=len(control.records))
            return result
        except BaseException as exc:
            sink.errors.append(repr(exc))
            raise
    return step


def install(stack: Any, collector: Any, history: Any, state: Any) -> None:
    sink = Sink(state, history)
    state['floating_exit_sink'] = sink
    stack.callback(sink.close)
    cls = collector.RecognitionPipeline
    original = cls._step_side
    stack.callback(setattr, cls, '_step_side', original)
    cls._step_side = scoped(original, sink)


def check_row(row: Any) -> None:
    """保存後にもscope/型/schemaを再検証し、余分な票を無視しない。"""
    common = {'kind', 'frame_idx', 'time_sec', 'side'}
    fields = {'enter': {'state'}, 'return': {'state', 'original_step_calls', 'rejects'},
        'reject': {'guard'}}
    require(type(row) is dict and row.get('kind') in fields, 'saved_kind')
    require(set(row) == common | fields[row['kind']], 'saved_schema')
    frame, clock = row['frame_idx'], row['time_sec']
    require(type(row['side']) is str and selected(frame, row['side']), 'saved_scope')
    require(type(clock) in (int, float) and clock == frame / FPS, 'saved_clock')
    if row['kind'] != 'reject':
        require(type(row['state']) is str and row['state'] in
            {'menu', 'stable', 'tsumo_fall', 'ojama_fall', 'chain', 'gravity_settle'}, 'saved_state')
        if row['kind'] == 'return':
            require(type(row['original_step_calls']) is int and row['original_step_calls'] == 1
                and type(row['rejects']) is int and row['rejects'] >= 0, 'saved_calls')
        return
    guard = row['guard']
    require(type(guard) is dict, 'saved_guard_type')
    cell = guard.get('cell')
    require(type(cell) is list and len(cell) == 2 and all(type(v) is int for v in cell)
        and 1 <= cell[0] < 12 and 0 <= cell[1] < 6, 'saved_guard_cell')
    expected = {'frame_idx': frame, 'time_sec': clock,
        'reason': 'ordinary_tsumo_visible_floating_single', 'cell': cell,
        'within_calls': 1, 'transition_calls': 0, 'row0_read': False,
        'detector_return_modified': False, 'state_after': 'tsumo_fall', 'quality_gate_clear': False}
    require(set(guard) == set(expected) and all(type(guard[k]) is type(v) and guard[k] == v
        for k, v in expected.items()), 'saved_guard_payload')


def check(output: Path) -> dict[str, Any]:
    status = json.loads((output / STATUS).read_text())
    require(type(status) is dict and set(status) == {'closed', 'errors'}
        and status['closed'] is True and type(status['errors']) is list and not status['errors'], 'sink_failed')
    rows = [json.loads(line) for line in (output / SIDECAR).read_text().splitlines()]
    previous_frame = FIRST
    for row in rows:
        check_row(row)
        require(row['frame_idx'] >= previous_frame, 'saved_global_order')
        previous_frame = row['frame_idx']
    expected = list(range(FIRST, LAST + STRIDE, STRIDE))
    enters = [r for r in rows if r['kind'] == 'enter']
    returns = [r for r in rows if r['kind'] == 'return']
    rejects = [r for r in rows if r['kind'] == 'reject']
    require([r['frame_idx'] for r in enters] == expected, 'enter_coverage')
    require([r['frame_idx'] for r in returns] == expected, 'return_coverage')
    for frame in expected:
        scope = [r for r in rows if r['frame_idx'] == frame]
        require(scope[0]['kind'] == 'enter' and scope[-1]['kind'] == 'return', 'order')
        require(all(r['kind'] == 'reject' for r in scope[1:-1]), 'kind')
        require(scope[-1]['original_step_calls'] == 1 and scope[-1]['rejects'] == len(scope) - 2, 'calls')
    require([r['frame_idx'] for r in rejects] == [34682, 34686], 'actual_reject_plan')
    require(all(r['state'] == 'tsumo_fall' for r in returns if 34624 <= r['frame_idx'] < LAST), 'premature_exit')
    require(returns[-1]['state'] == 'stable', 'finite_exit_missing')
    return {'frames': expected, 'reject_frames': [r['frame_idx'] for r in rejects],
        'finite_sm_exit': LAST, 'source_guards': guards(), 'quality_gate_clear': False,
        'full_pipeline_executed': True, 'physical_hand_certified': False, 'production_permission': False}


def finish(state: Any) -> None:
    write(state['output'] / RECEIPT, check(state['output']))


def verify(output: Path) -> None:
    require(json.loads((output / RECEIPT).read_text()) == check(output), 'receipt_mismatch')
