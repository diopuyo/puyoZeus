"""実Mode/Registry型と人工取得票で契約を検収。原publish/実Jの合格ではない。"""
from dataclasses import dataclass, asdict
from contextlib import ExitStack
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace as N
from typing import Any
import pytest
import first_static_reflection as F

sys.path.insert(0, str(F.PUB))
from test_second_physical import physical
from test_joint_evaluation import states
import registry as R
import serialization as S
import live_session as L

A = N(Observer=N)  # callback検査用の人工Connectionメタデータ。


@dataclass(frozen=True)
class Candidate:
    scope: tuple
    frame: int
    source_call_token: str


@pytest.fixture
def mode_type(physical: Any) -> type:
    path = F.VERIFY / 'g2_basis_cascade_candidate_2026-09-11_v1/mode_v2.py'
    spec = importlib.util.spec_from_file_location('_first_static_actual_mode', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    prior = sys.modules.get('mode')
    sys.modules['mode'] = physical
    try:
        spec.loader.exec_module(module)
    finally:
        if prior is None:
            sys.modules.pop('mode')
        else:
            sys.modules['mode'] = prior
    return module.Mode


@pytest.fixture
def sample(tmp_path: Path, mode_type: type) -> Any:
    value = states()[0][0]
    factory = N()
    registry = R.Registry(factory)
    binding = registry.bind(factory, value, 'step:10')
    setattr(factory, F.KEY, registry)
    journal = N(closed=False, errors=[], active=None)
    recovery = N(factory=factory, journal=journal, error=None, failure=None)
    candidate = Candidate(value.scope, value.frame, binding.initial_call_token)
    observer = N(recovery=recovery, error=None, gate=N(state='ISSUED', candidate=candidate, deadline=12))
    with (tmp_path / 'PROBABILISTIC_BASIS.jsonl').open('x', encoding='utf-8') as stream:
        c = N(registry=registry, binding=binding, observer=observer, recovery=recovery, stream=stream, rows=1, deadline=30)
        state = dict(output=tmp_path, probabilistic_basis_connection=c, **{F.KEY: registry})
        mode = mode_type(c, state, stream)
        state['probabilistic_tracking_mode'] = mode
        mode.native = N(connection=c, last_frame=12, seen_calls={'step:12'}, pending=[], seen_occurrences=[])
        mode.activation = dict(frame=10, source_call_token='step:10', tracking_deadline=30, acquisition_deadline=12)
        row = dict(kind='actual_settled_probabilistic_basis', source_call_token='step:10', state=S.encode(value),
            tracking_deadline=30, basis_deadline=12, initial_candidate=asdict(candidate),
            integer_current_published=False, legacy_collector_append=False, quality_gate_clear=False)
        stream.write(json.dumps(row) + '\n')
        stream.flush()
        calls: list[Any] = []
        old = N(verify=lambda *args: calls.append(args))
        verifier = F.Verifier(state, factory, mode, old, S, type(observer))
        yield N(v=verifier, mode=mode, value=value, c=c, row=row, path=Path(stream.name), calls=calls)


def test_static_actual_mode_and_saved_receipt(sample: Any) -> None:
    s = sample
    s.v.verify(s.mode, s.value, 'step:12', 12)
    assert not s.calls and s.v.report['checks'][0]['basis_frame'] == 10
    assert s.v.report['checks'][0]['evaluation_frame'] == 12


@pytest.mark.parametrize('case', ('pending', 'consumed', 'unseen', 'origin', 'gate', 'registry', 'deadline', 'token', 'distribution',
                                 'candidate', 'basis_deadline', 'expired', 'cascade_closed', 'error', 'closed', 'retired'))
def test_bad_inputs_rejected(sample: Any, case: str) -> None:
    s = sample
    if case == 'pending': s.mode.native.pending.append(object())
    elif case == 'consumed': s.mode.native.seen_occurrences.append('used')
    elif case == 'unseen': s.mode.native.seen_calls.clear()
    elif case == 'origin': s.mode.origins['unsettled'] = {}
    elif case == 'gate': s.c.observer.gate.state = 'BROKEN'
    elif case == 'registry': setattr(s.v.factory, F.KEY, object())
    elif case == 'expired':
        s.mode.native.last_frame = 31
    elif case == 'cascade_closed': s.mode.basis_cascade_closed = True
    elif case == 'error': s.mode.error = 'prior failure'
    elif case == 'closed': s.mode.closed = True
    elif case == 'retired': s.mode.retired_receipt = {}
    else:
        if case == 'deadline': s.row['tracking_deadline'] = 31
        elif case == 'token': s.row['source_call_token'] = 'foreign'
        elif case == 'candidate': s.row['initial_candidate']['frame'] = 11
        elif case == 'basis_deadline': s.row['basis_deadline'] = 13
        else: s.row['state']['frame'] = 11
        s.path.write_text(json.dumps(s.row) + '\n', encoding='utf-8')
    with pytest.raises(ValueError):
        s.v.verify(s.mode, s.value, 'step:12', 31 if case == 'expired' else 12)
    assert s.v.report['error'] and not s.v.report['checks']


def test_other_mode_and_transition_delegate(sample: Any) -> None:
    s = sample
    other = N()
    s.v.verify(other, s.value, 'step:12', 12)
    s.mode.applied.append({'test': True})
    s.v.verify(s.mode, s.value, 'step:12', 12)
    assert [args[0] for args in s.calls] == [other, s.mode]


def capture_session(sample: Any) -> Any:
    class Connection:
        def __init__(self, previous: Any) -> None:
            self.__dict__.update(vars(previous))
    c = Connection(sample.c)
    sample.mode.connection = c
    sample.mode.native.connection = c
    sample.v.state['probabilistic_basis_connection'] = c
    session = object.__new__(L.Session)
    session.state, session.factory = sample.v.state, sample.v.factory
    return session


def test_original_capture_instance_restored(sample: Any) -> None:
    session = capture_session(sample)
    original = session.capture
    with ExitStack() as stack:
        report = F.install(stack, session)
        assert session.capture != original and 'capture' in vars(session)
        with pytest.raises(ValueError, match='capture_already_owned'):
            F.install(stack, session)
    assert session.capture == original and 'capture' not in vars(session) and report['restored']


def test_foreign_capture_not_silently_restored(sample: Any) -> None:
    session = capture_session(sample)
    with pytest.raises(ValueError, match='capture_foreign_hook'):
        with ExitStack() as stack:
            F.install(stack, session)
            session.capture = lambda: None
