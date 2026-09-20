"""原採録2side→原observe呼出ASTを使う局所CPU。動画/モデル/開始資格は人工・非対象。"""
from __future__ import annotations
import ast
from contextlib import ExitStack
import importlib.util
import json
from pathlib import Path
from types import CodeType, FunctionType, ModuleType, SimpleNamespace as N
from typing import Any
import pytest
import whole_collector_capture as W

REPO = Path(__file__).resolve().parents[3]
SOURCE = REPO / '.runtime_snapshots/event_first30_observed_context_v5_2026-08-30/scripts/collect_boards_lean.py'
BASE = REPO / 'data/verify/g2_producer_time_capture_2026-09-12_v1/capture.py'
spec = importlib.util.spec_from_file_location('_whole_test_original_capture', BASE)
C = importlib.util.module_from_spec(spec)
spec.loader.exec_module(C)


class Tail:
    def __init__(self) -> None:
        self.count, self.rows, self.stream = 0, [], N(count=0)

    def check(self, frame: int) -> None:
        W.require(self.rows == [dict(frame=frame, side=s) for s in ('1P', '2P')], 'test_tail')

    def append(self, frame: int, side: str) -> None:
        self.rows = (self.rows + [dict(frame=frame, side=side)])[-2:]
        self.count += 1
        self.stream.count += 1


def extracted() -> ast.Module:
    original = next(n for n in ast.parse(SOURCE.read_bytes()).body
                    if isinstance(n, ast.FunctionDef) and n.name == 'collect_lean')
    init = next(n for n in original.body if isinstance(n, ast.Assign)
                and ast.unparse(n.targets[0]) == 'accounting_recorder')
    loop = next(n for n in original.body if isinstance(n, ast.For) and ast.unparse(n.target) == 'local_i')
    first = next(i for i, n in enumerate(loop.body) if isinstance(n, ast.Expr)
                 and isinstance(n.value, ast.Call) and ast.unparse(n.value.func) == '_process_side_lean')
    block = loop.body[first:first + 4]
    assert len(block) == 4 and ast.unparse(block[-1].test) == 'physical_recorder is not None'
    calls = [n for n in ast.walk(block[2]) if isinstance(n, ast.Call)]
    assert len(calls) == 1 and len(calls[0].args) == 3
    assert [key.arg for key in calls[0].keywords] == ['game_idx', 'chain_events']
    physical_init = next(n for n in original.body if isinstance(n, ast.Assign)
                         and ast.unparse(n.targets[0]) == 'physical_recorder')
    first_init, last_init = original.body.index(init), original.body.index(physical_init)
    assert first_init < last_init and last_init == first_init + 1
    function = ast.parse('def collect_lean(shared_game, pipeline, ojama_tracker, result, start_frame=100, '
                         'fps=60, effective_interval_frames=2, enable_event_accounting_sidecar=True, '
                         'enable_event_physical_sidecar=True):\n pass').body[0]
    cycle = ast.parse('for fi in (100, 102):\n t_sec = fi / fps').body[0]
    cycle.body.extend(block)
    function.body = original.body[first_init:last_init + 1] + [cycle]
    return ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))


def fixture(monkeypatch: Any, output: Path, *, defect: str = '') -> tuple[Any, ...]:
    tail, events = Tail(), []
    class Recorder:
        def __init__(self) -> None:
            self._observed_frame_count = int(defect == 'late')
        def observe(self, frame: int, clock: float, tracker: Any, **kwargs: Any) -> None:
            events.append(('accounting', frame))
            if defect == 'error':
                raise LookupError('original_observe_failure')
            self._observed_frame_count += int(defect != 'count')
        def sidecar_value(self, first: int, last: int) -> dict:
            return dict(first=first, last=last, artificial=True)
    tree, collector = extracted(), ModuleType('_whole_test_collector')
    collector.__dict__.update({node.id: None for node in ast.walk(tree) if isinstance(node, ast.Name)})
    def side(*args: Any, **kwargs: Any) -> None:
        label, frame = args[2], args[8]
        events.append((label, frame))
        if not (defect == 'metadata' and label == '2P'):
            tail.append(frame, label)
    collector.__dict__.update(_process_side_lean=side, EventAccountingRecorder=Recorder,
                              enable_event_accounting_sidecar=True, _physical_match_evidence=lambda value: True,
                              getattr=getattr)
    class Physical:
        def __init__(self) -> None:
            self._observed_frame_count = 0
        def observe(self, frame: int, *args: Any, **kwargs: Any) -> None:
            events.append(('physical', frame))
            if defect == 'physical_error':
                raise LookupError('physical_observe_failure')
            self._observed_frame_count += int(defect != 'physical_count')
    collector.EventPhysicalRecorder = Physical
    exec(compile(tree, str(SOURCE), 'exec'), vars(collector))
    # 局所抽出codeのみの試験許可。本番Bridgeのsource認証を通ったとは呼ばない。
    monkeypatch.setattr(W, 'authentic_code', lambda value: collector.collect_lean.__code__)
    state = dict(output=output, reset_metadata_tail=tail,
                 private_suffix_factory=N(provider=N(journal=N(source_id='fixture', run_id='fixture'))))
    shared = N(**{key: [] for key in C.LISTS}, game_idx=0)
    value = N(confirmed_board=None, state=None, score=None, next_pair=None, dnext_pair=None, chain_event=None)
    def consume(bridge: Any, frame: int) -> None:
        assert bridge.capture.snapshot()['last_frame'] == frame
        events.append(('consumer', frame))
    return collector, state, Recorder, shared, N(), N(), N(p1=value, p2=value), consume, events


def test_original_source_code_authentication() -> None:
    compiled = compile(SOURCE.read_bytes(), str(SOURCE), 'exec', dont_inherit=True)
    code = next(c for c in compiled.co_consts if isinstance(c, CodeType) and c.co_name == 'collect_lean')
    collector = ModuleType('_whole_original_source')
    collector.__file__ = str(SOURCE)
    collector.collect_lean = FunctionType(code, vars(collector))
    assert W.authentic_code(collector) is code
    collector.collect_lean = lambda: None
    with pytest.raises(ValueError, match='original_whole_code'):
        W.authentic_code(collector)


def test_original_side_accounting_order_and_saved_lifetime(monkeypatch: Any, tmp_path: Path) -> None:
    collector, state, cls, shared, pipe, tracker, result, consumer, events = fixture(monkeypatch, tmp_path)
    with ExitStack() as stack:
        bridge = W.Bridge(collector, state, stack, cls, C, (100, 102), consumer)
        collector.collect_lean(shared, pipe, tracker, result)
        # 原ループ後のreconcile相当。保存済み因果原票は変えない。
        shared.anomalies.append('posthoc')
    assert events == [(kind, f) for f in (100, 102) for kind in ('1P', '2P', 'accounting', 'physical', 'consumer')]
    assert bridge.physical._observed_frame_count == 2 and 'observe' not in vars(bridge.physical)
    assert bridge.closed and bridge.capture.closed and collector.EventAccountingRecorder is cls
    assert 'observe' not in vars(bridge.capture.recorder)
    assert state['joint_producer_capture_snapshot']['boundary_events'] == []
    assert (tmp_path / 'JOINT_PRODUCER_CAPTURE.json').is_file()


@pytest.mark.parametrize('defect,reason', [('late', 'late_recorder'), ('count', 'unprocessed_or_count'),
    ('metadata', 'test_tail'), ('error', 'original_observe_failure'),
    ('physical_count', 'physical_post_count'), ('physical_error', 'physical_observe_failure')])
def test_original_failure_and_references(monkeypatch: Any, tmp_path: Path, defect: str, reason: str) -> None:
    collector, state, cls, shared, pipe, tracker, result, consumer, events = fixture(monkeypatch, tmp_path, defect=defect)
    with pytest.raises((ValueError, LookupError), match=reason):
        with ExitStack() as stack:
            bridge = W.Bridge(collector, state, stack, cls, C, (100, 102), consumer)
            collector.collect_lean(shared, pipe, tracker, result)
    assert collector.EventAccountingRecorder is cls
    assert not (tmp_path / 'JOINT_PRODUCER_CAPTURE.json').exists()
    assert not any(kind == 'consumer' for kind, _ in events)
    if bridge.capture is not None:
        assert bridge.capture.closed and 'observe' not in vars(bridge.capture.recorder)
        failed = json.loads((tmp_path / 'JOINT_PRODUCER_CAPTURE.failure.json').read_text())
        assert reason in failed['error'] and failed['attempted_frame'] == 100
        assert not failed['full_snapshot'] and not failed['quality_gate_clear']
    if bridge.physical is not None:
        assert 'observe' not in vars(bridge.physical)
    assert collector.EventPhysicalRecorder is bridge.original_physical_factory


def test_one_update_transport_is_never_an_execution_path() -> None:
    with pytest.raises(RuntimeError, match='whole_transport_is_not_a_collector'):
        W.forbidden_collect(None, None, 100, 1, 2, 60)


def test_failure_keeps_partial_start_trace(monkeypatch: Any, tmp_path: Path) -> None:
    collector, state, cls, shared, pipe, tracker, result, consumer, events = fixture(
        monkeypatch, tmp_path, defect='physical_error')
    original = C.Capture.completed
    def completed(self: Any, frame: int) -> None:
        original(self, frame)
        self.start_counter_trace = [dict(frame=frame, artificial=True)]
    monkeypatch.setattr(C.Capture, 'completed', completed)
    with pytest.raises(LookupError, match='physical_observe_failure'):
        with ExitStack() as stack:
            bridge = W.Bridge(collector, state, stack, cls, C, (100, 102), consumer)
            collector.collect_lean(shared, pipe, tracker, result)
    saved = json.loads((tmp_path / 'JOINT_PRODUCER_CAPTURE.failure.json').read_bytes())
    assert saved['start_counter_trace'] == [dict(frame=100, artificial=True)]
    assert saved['completed_frames'] == [] and saved['capture_last'] == 100
    assert saved['error_type'] == 'LookupError' and saved['stage'] == 'physical'
    assert not saved['full_snapshot'] and bridge.failure_save_error is None


def test_terminal_consumer_failure_keeps_producer_receipt(monkeypatch: Any, tmp_path: Path) -> None:
    collector, state, cls, shared, pipe, tracker, result, consumer, events = fixture(monkeypatch, tmp_path)
    def failed_consumer(bridge: Any, frame: int) -> None:
        if frame == 102:
            raise RuntimeError('terminal_consumer_failure')
    with pytest.raises(RuntimeError, match='terminal_consumer_failure'):
        with ExitStack() as stack:
            bridge = W.Bridge(collector, state, stack, cls, C, (100, 102), failed_consumer)
            collector.collect_lean(shared, pipe, tracker, result)
    assert bridge.closed and bridge.capture.closed
    assert collector.EventAccountingRecorder is cls
    assert (tmp_path / 'JOINT_PRODUCER_CAPTURE.json').is_file()


@pytest.mark.parametrize('changed', ['code', 'globals'])
def test_caller_identity_rejects_foreign_frame(monkeypatch: Any, tmp_path: Path, changed: str) -> None:
    collector, state, cls, shared, pipe, tracker, result, consumer, events = fixture(monkeypatch, tmp_path)
    with pytest.raises(ValueError, match='caller_' + changed):
        with ExitStack() as stack:
            bridge = W.Bridge(collector, state, stack, cls, C, (100, 102), consumer)
            fake = N(f_code=bridge.code if changed == 'globals' else (lambda: None).__code__,
                     f_globals={} if changed == 'globals' else vars(collector), f_locals={})
            bridge.caller(fake)
    assert collector.EventAccountingRecorder is cls


def test_duplicate_bridge_rejected(monkeypatch: Any, tmp_path: Path) -> None:
    collector, state, cls, shared, pipe, tracker, result, consumer, events = fixture(monkeypatch, tmp_path)
    with pytest.raises(ValueError, match='duplicate_install'):
        with ExitStack() as stack:
            W.Bridge(collector, state, stack, cls, C, (100, 102), consumer)
            W.Bridge(collector, state, stack, cls, C, (100, 102), consumer)
    assert collector.EventAccountingRecorder is cls


def test_physical_disabled_does_not_wait_for_missing_observer(monkeypatch: Any, tmp_path: Path) -> None:
    collector, state, cls, shared, pipe, tracker, result, consumer, events = fixture(monkeypatch, tmp_path)
    with ExitStack() as stack:
        bridge = W.Bridge(collector, state, stack, cls, C, (100, 102), consumer)
        collector.collect_lean(shared, pipe, tracker, result, enable_event_physical_sidecar=False)
    assert bridge.closed and bridge.physical is None
    assert events == [(kind, f) for f in (100, 102) for kind in ('1P', '2P', 'accounting', 'consumer')]
