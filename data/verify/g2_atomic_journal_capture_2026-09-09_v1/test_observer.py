"""人工初期状態の実生成stepと、保存/例外の追加境界だけを検査する。"""
from __future__ import annotations
from collections import Counter, deque
import contextlib
import copy
import dataclasses
from enum import Enum
from dataclasses import dataclass
import importlib.util
import inspect
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any
import pytest

ROOT = Path(__file__).resolve().parent


def load(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


J = sys.modules.get('_atomic_journal') or load('_atomic_journal', ROOT / 'observer.py')
OLD = ROOT.parent / 'g2_t2_fresh_landing_guard_2026-09-09_v1'
sys.path.insert(0, str(OLD))
T = load('_atomic_latest_t2_tests', OLD / 'test_guard.py')
P = T.P
frozen, prepared, guarded_hash_cache = T.frozen, T.prepared, T.guarded_hash_cache
OUT = Path(os.environ.get('MANUFACTURING_OUTPUT', ROOT))


def serialized(value: Any) -> Any:
    if type(value) is bytes:
        return {'bytes_hex': value.hex()}
    if type(value) in (tuple, list):
        return [serialized(v) for v in value]
    if type(value) is dict:
        return {k: serialized(v) for k, v in value.items()}
    if dataclasses.is_dataclass(value):
        return {f.name: serialized(getattr(value, f.name)) for f in dataclasses.fields(value)}
    if isinstance(value, Enum):
        return value.name
    if hasattr(value, '_grid'):
        return value._grid.tolist()
    if hasattr(value, 'tolist'):
        return value.tolist()
    return value


def instrumenter(original: Any, enabled: bool, scopes: list[Any]) -> Any:
    def installed(stack: Any, collector: Any, history: Any, receipt: Any, state: Any) -> None:
        T.A.install(stack, P.S.M, enabled=True)
        original(stack, collector, history, receipt, state)
        P.H.install(stack, collector, history, state)
        P.T.W.install(stack, collector, history, state)
        if enabled:
            J.install(stack, collector, history, state, expected_scopes=scopes,
                      source_id='artificial-cpu-source', run_id='artificial-cpu-run')
    return installed


def actual_call(real: Any, state: Any, raw: Any, frame: int) -> Any:
    import numpy as np
    pipe = real[0]
    with P.S.C.direct_clock(state, pipe, frame):
        return pipe._step_side('2P', frame, frame / 60, True, raw, None,
            score_d_2p_for_ojama=0, sm=pipe._sm_2p, gen=pipe._gen_2p, drift=pipe._drift_2p,
            score_tracker=None, next_pair=(1, 5), slide_motion=False,
            frame_bgr=np.zeros((1080, 1920, 3), dtype=np.uint8), score_d_for_self=1)


def exercise(frozen: Any, prepared: Any, monkeypatch: Any, enabled: bool) -> tuple[Any, Any]:
    scopes, original = [(34702, '2P'), (34704, '2P')], P.S.M.instrument
    output = OUT / ('actual_on' if enabled else 'actual_off')
    with monkeypatch.context() as scoped:
        scoped.setattr(P.S.M, 'instrument', instrumenter(original, enabled, scopes))
        with P.S.runtime(frozen, scoped, output, prepared) as (real, state):
            pipe = real[0]
            raw, votes, seed = P.seed(pipe, 'ordinary')
            controller = vars(sys.modules[type(pipe).__module__])['__next_live']
            controller._runtime(pipe)  # direct step試験のみの人工開始。実updateではbeginが生成する。
            trace = P.WriterTrace(pipe, state)
            with trace.installed():
                first = actual_call(real, state, raw, 34702)
                raw = pipe._sm_2p.context.confirmed_board.copy()
                pipe._sm_2p.context.next_queue.append((2, 4))
                second = actual_call(real, state, raw, 34704)
            value = {'result': [P.T.comparable_result(first), P.T.comparable_result(second)],
                'account': J.account(pipe, '2P'), 'calls': dict(trace.counts), 'classify_calls': len(votes),
                'evidence': copy.deepcopy(state['evidence_sink'].rows),
                'witness': copy.deepcopy(state['hsv_witness'].rows), 'seed': seed}
        if enabled:
            assert state[J.KEY].closed and not state[J.KEY].errors
            J.finish(state)
    return value, state


def test_actual_generated_step_noninterference(frozen: Any, prepared: Any, monkeypatch: Any) -> None:
    before = inspect.getattr_static(frozen.RecognitionPipeline, '_step_side')
    old, _ = exercise(frozen, prepared, monkeypatch, False)
    new, state = exercise(frozen, prepared, monkeypatch, True)
    assert old == new
    assert new['calls'] == {'infer': 1, 'resolve': 1} and new['classify_calls'] == 2
    assert inspect.getattr_static(frozen.RecognitionPipeline, '_step_side') is before
    rows = [json.loads(line) for line in (state['output'] / J.SIDECAR).read_text().splitlines()]
    stages = {v['stage'] for row in rows for v in row['events']}
    assert {'fifo_before', 'fifo_after', 'resolve_before', 'resolve_after', 'origin_before',
            'origin_after', 'next_validation_before', 'next_validation_after'} <= stages
    J.write(OUT / 'ACTUAL_COMPARISON.json', {'off': serialized(old), 'on': serialized(new), 'artificial_initial_state': True,
        'actual_video_certified': False, 'source_code_unchanged': True, 'quality_gate_clear': False})


def test_actual_update_enqueue_occurrences(frozen: Any, prepared: Any, monkeypatch: Any) -> None:
    scopes = [(f, s) for f in (35844, 35846) for s in J.SIDES]
    original, outputs = P.S.M.instrument, []
    for enabled in (False, True):
        output = OUT / ('update_on' if enabled else 'update_off')
        with monkeypatch.context() as scoped:
            scoped.setattr(P.S.M, 'instrument', instrumenter(original, enabled, scopes))
            with P.S.runtime(frozen, scoped, output, prepared) as (real, state):
                controller = vars(sys.modules[type(real[0]).__module__])['__next_live']
                assert id(real[0]) not in controller.instances
                for frame, pair in ((35844, (5, 5)), (35846, (2, 3))):
                    P.S.C.T.OLD.OLD.drive(real, frame, pair)
                outputs.append(P.S.C.T.OLD.OLD.recorded(real))
            if enabled:
                assert not state[J.KEY].errors and state[J.KEY].closed
                rows = [json.loads(line) for line in (output / J.SIDECAR).read_text().splitlines()]
    assert outputs[0] == outputs[1]
    enqueues = [r for r in rows if r['kind'] == 'enqueue']
    assert len(enqueues) == 4 and sum(len(r['added_occurrence_tokens']) for r in enqueues) == 2
    assert len({token for r in enqueues for token in r['added_occurrence_tokens']}) == 2
    J.write(OUT / 'ACTUAL_ENQUEUE.json', {'rows': rows, 'old_result_equal': True,
        'artificial_detector_reader': True, 'real_begin_created_instance': True, 'quality_gate_clear': False})


@dataclass(frozen=True)
class Generation:
    reset: int = 2
    action: int = 7


def minimal(tmp_path: Path) -> tuple[Any, Any]:
    pipe = SimpleNamespace(_pending_tsumo_1p=deque(), _tsumo_count_1p=Counter({4: 2}), _sm_1p=object())
    tracker = SimpleNamespace(_pipeline=pipe, _machines={'1P': pipe._sm_1p}, generation=lambda side: Generation())
    controller = SimpleNamespace(instances={id(pipe): SimpleNamespace(histories={'1P': SimpleNamespace(epoch=2)})})
    rec = J.Recorder(tmp_path, SimpleNamespace(frame=60, time_sec=1.0), tracker, controller,
                     'artificial-source', 'artificial-run', [(60, '1P')])
    return rec, pipe


def test_fifo_same_color_occurrences_and_reset() -> None:
    pipe, fifo = SimpleNamespace(_pending_tsumo_1p=deque()), J.Fifo()
    saved = fifo.sync(pipe, '1P', 0)
    first, second = tuple([4, 4]), tuple([4, 4])
    pipe._pending_tsumo_1p.append(first)
    assert fifo.appended(saved, 'enqueue:0') == ['enqueue:0:slot:0']
    pipe._pending_tsumo_1p.append(second)
    assert fifo.appended(saved, 'enqueue:1') == ['enqueue:1:slot:1']
    pipe._pending_tsumo_1p.popleft()
    assert fifo.consumed(saved) == 'enqueue:0:slot:0'
    changed = fifo.sync(pipe, '1P', 1)
    assert changed['tokens'] == [None] and changed['discarded_tokens'] == ['enqueue:1:slot:1']
    assert changed['initial_reason'] == 'software_reset_invalidated'


def test_fifo_unexplained_prefix_rejected() -> None:
    pipe, fifo = SimpleNamespace(_pending_tsumo_1p=deque([tuple([4, 4])])), J.Fifo()
    fifo.sync(pipe, '1P', 0)
    pipe._pending_tsumo_1p[0] = tuple([4, 4])
    with pytest.raises(ValueError, match='unexplained_fifo'):
        fifo.sync(pipe, '1P', 0)


def test_inactive_clear_does_not_become_consumption() -> None:
    pair = tuple([4, 4])
    pipe = SimpleNamespace(_pending_tsumo_1p=deque(), _tsumo_count_1p=Counter({4: 2}))
    fifo = J.Fifo()
    saved = fifo.sync(pipe, '1P', 0)
    pipe._pending_tsumo_1p.append(pair)
    fifo.appended(saved, 'enqueue:0')
    beginning = {'queue': pipe._pending_tsumo_1p, 'refs': (pair,)}
    pipe._pending_tsumo_1p.clear()
    pipe._tsumo_count_1p.clear()
    fifo.inactive_clear(pipe, '1P', 0, beginning)
    assert fifo.sync(pipe, '1P', 0)['tokens'] == []
    assert saved['discarded_tokens'] == ['enqueue:0:slot:0']
    assert 'writer_not_traced' in saved['initial_reason']


@pytest.mark.parametrize('kind', ['step', 'enqueue'])
def test_original_exception_wins_emit_failure(tmp_path: Path, monkeypatch: Any, kind: str) -> None:
    rec, pipe = minimal(tmp_path)
    error = RuntimeError('original_marker')
    def original(*args: Any) -> Any:
        raise error
    def broken(value: Any) -> None:
        raise OSError('observer_disk_failure')
    monkeypatch.setattr(rec, 'emit', broken)
    rec.codes = {original.__code__}
    before_trace, before_profile = sys.gettrace(), sys.getprofile()
    wrapper = rec.wrap_step(original) if kind == 'step' else rec.wrap_enqueue(original)
    try:
        with pytest.raises(RuntimeError) as caught:
            wrapper(pipe, '1P', 60, 1.0, True, (4, 4))
        assert caught.value is error
        assert sys.gettrace() is before_trace and sys.getprofile() is before_profile
        assert rec.active is None and any('save:' in v for v in rec.errors)
    finally:
        rec.close()


def test_scope_clock_missing_begin_and_counter_copy(tmp_path: Path) -> None:
    rec, pipe = minimal(tmp_path)
    try:
        with pytest.raises(ValueError, match='journal_clock'):
            rec.scope(pipe, '1P', True, 1.0)
        rec.controller.instances.clear()
        with pytest.raises(ValueError, match='next_begin_not_observed'):
            rec.epoch(pipe, '1P')
        value = J.account(pipe, '1P')
        assert value['tsumo_count'] == {'4': 2} and value['pending_tsumo'] == []
        value['tsumo_count']['4'] = 999
        assert pipe._tsumo_count_1p[4] == 2
    finally:
        rec.close()


@pytest.mark.parametrize('change', ['code', 'extra_after', 'empty', 'order', 'guards', 'permission', 'status_bool'])
def test_saved_contract_negatives(change: str, tmp_path: Path) -> None:
    source = OUT / 'actual_on'
    for name in J.REQUIRED:
        (tmp_path / name).write_bytes((source / name).read_bytes())
    receipt = json.loads((tmp_path / J.RECEIPT).read_text())
    rows = [json.loads(line) for line in (tmp_path / J.SIDECAR).read_text().splitlines()]
    if change == 'code':
        rows[0]['code_sha256'] = '0' * 64
    elif change == 'extra_after':
        rows[0]['events'].append(copy.deepcopy(rows[0]['events'][1]))
    elif change == 'empty':
        for row in rows:
            row['events'] = []
    elif change == 'order':
        rows[0]['events'] = list(reversed(rows[0]['events']))
    elif change == 'guards':
        receipt['guards'] = {}
    elif change == 'permission':
        receipt['accounting_permission'] = True
    else:
        (tmp_path / J.STATUS).write_text(J.encoded({'closed': 1, 'errors': [], 'active': False}))
        receipt['sha256'][J.STATUS] = J.sha(tmp_path / J.STATUS)
    (tmp_path / J.SIDECAR).write_text(''.join(J.encoded(r) + '\n' for r in rows))
    receipt['sha256'][J.SIDECAR] = J.sha(tmp_path / J.SIDECAR)
    (tmp_path / J.RECEIPT).write_text(J.encoded(receipt))
    with pytest.raises(ValueError):
        J.verify(tmp_path, expected_scopes=[(34702, '2P'), (34704, '2P')])


def test_completed_step_releases_actual_frame_reference(tmp_path: Path) -> None:
    rec, pipe = minimal(tmp_path)
    item = {'pipe': pipe, 'scope': rec.scope(pipe, '1P', 60, 1.0), 'token': 'step:0', 'events': [],
        'epoch': 2, 'return_line': 1, 'frame': SimpleNamespace(f_code=test_completed_step_releases_actual_frame_reference.__code__)}
    try:
        rec.complete_step(item, None, RuntimeError('artificial_original'), sys.getprofile())
        assert item['frame'] is None
    finally:
        rec.close()
