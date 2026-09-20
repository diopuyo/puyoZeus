"""固定原codeの終了経路。constructor/archive/Registryの入力は人工対照と明示する。"""
from __future__ import annotations
from contextlib import ExitStack
import importlib.util
from types import CellType, FunctionType, SimpleNamespace as N
from typing import Any
import pytest
import probability_outer_scope as S
import test_probability_owner as O
import test_live_outer_repro as R

COMMON_PATH = S.BASE / 'g2_history_publication_probe_runtime_2026-09-10_v13/common.py'
COMMON_SPEC = importlib.util.spec_from_file_location('_outer_test_actual_common', COMMON_PATH)
K = importlib.util.module_from_spec(COMMON_SPEC)
COMMON_SPEC.loader.exec_module(K)


def chain(calls: list[str]) -> Any:
    functions = []
    namespace = dict(__file__=str(S.BASE / S.PINS[2][0]), K=K,
                     __builtins__=__builtins__)
    raw_code = compile((S.BASE / S.PINS[2][0]).read_bytes(), namespace['__file__'], 'exec', dont_inherit=True)
    status_code = next(c for c in S.codes(raw_code) if c.co_name == 'scope_status')
    namespace['scope_status'] = FunctionType(status_code, namespace)
    for index in (2, 1, 0):
        path = S.BASE / S.PINS[index][0]
        compiled = compile(path.read_bytes(), str(path), 'exec', dont_inherit=True)
        code = next(c for c in S.codes(compiled) if c.co_name == 'verify')
        members = dict(original_verify=functions[-1] if functions else None,
                       checked=functions[-1] if functions else None,
                       connection=N(verify=lambda state: calls.append('capture')))
        cells = tuple(CellType(members[name]) for name in code.co_freevars) or None
        functions.append(FunctionType(code, namespace, 'verify', None, cells))
    return N(verify=functions[-1], scope_status=namespace['scope_status'])


def state(calls: list[str], probability: bool = True) -> dict[str, Any]:
    value = R.state(not probability)
    value['repeat_scope_guard'].frame = K.FRAMES[-1]
    if probability:
        sample = O.sample()
        guard = sample.lease.guard
        guard.reset_lease, guard.factory = sample.lease, sample.factory
        guard.error, guard.record, guard.frame = None, None, K.FRAMES[-1]
        value.update(sample.runtime)
        sample.mode.state = value
        value['repeat_scope_guard'] = guard
    value.update(combined_restored=lambda: calls.append('combined'), conditional_full_installs=1,
                 combined_install=dict(configure_calls=1), conditional_revision_connection=dict(installs=1))
    return value


@pytest.mark.parametrize('probability', [False, True])
def test_full_chain_restore_raw_truth(probability: bool) -> None:
    calls: list[str] = []
    repeated = chain(calls)
    original, raw = repeated.verify, repeated.scope_status
    main = FunctionType((lambda: None).__code__, dict(REPEAT=repeated))
    value = state(calls, probability)
    before = dict(value)
    with ExitStack() as stack:
        S.install(stack, main)
        assert repeated.verify.__code__ is original.__code__
        result = repeated.verify(value)
        assert result['same_scope_guard']['same_binding'] is (not probability)
        assert result['original_capture_observer_closed'] and result['conditional_shared_installed']
        assert bool(result.get('probability_owner_verified')) is probability
        assert calls == ['combined', 'capture']
        assert repeated.scope_status is raw
    assert repeated.verify is original and repeated.scope_status is raw
    assert value.keys() == before.keys() and all(value[k] is v for k, v in before.items())


@pytest.mark.parametrize('defect', ['constructor', 'frame', 'error', 'record', 'combined', 'count',
                                  'guard', 'state', 'registry', 'integer', 'receipt'])
def test_rejection_preserves_checks(defect: str) -> None:
    calls: list[str] = []
    repeated, value = chain(calls), state(calls)
    guard = value['repeat_scope_guard']
    if defect == 'constructor': value['repeated_firing_constructor']['closed'] = False
    elif defect == 'frame': guard.frame -= 2
    elif defect == 'error': guard.error = RuntimeError('original')
    elif defect == 'record': guard.record = {}
    elif defect == 'combined': value['combined_restored'] = lambda: (_ for _ in ()).throw(AssertionError('combined'))
    elif defect == 'count': value['combined_install']['configure_calls'] = 2
    elif defect == 'guard': guard.reset_lease.guard = N(binding=None)
    elif defect == 'state': value['probabilistic_tracking_mode'].state = dict(value)
    elif defect == 'registry': value[O.P.REGISTRY_KEY] = object()
    elif defect == 'integer': guard.factory.controller.history['1P'] = object()
    elif defect == 'receipt': guard.reset_lease.empty_evidence.receipt_sha = 'bad'
    expected = ValueError if defect in ('constructor', 'frame', 'error', 'record') else AssertionError
    with pytest.raises(expected):
        S.clone(repeated.verify)(value)
    assert 'capture' not in calls


def test_installed_but_inactive_mode_uses_original_integer_path() -> None:
    calls: list[str] = []
    repeated, value = chain(calls), state(calls, False)
    # 原Mode.__init__はinstall時に存在し、activate前は三者ともNone。
    value['probabilistic_tracking_mode'] = N(native=None, activation=None, connection=N(binding=None), error=None)
    result = S.clone(repeated.verify)(value)
    assert result['same_scope_guard']['same_binding'] is True
    assert 'probability_owner_verified' not in result
    assert calls == ['combined', 'capture']


@pytest.mark.parametrize('field', ['native', 'activation', 'binding', 'error'])
def test_partial_mode_never_falls_back(field: str) -> None:
    mode = N(native=None, activation=None, connection=N(binding=None), error=None)
    setattr(mode.connection if field == 'binding' else mode, field, object())
    assert O.P.tracking_selected({'probabilistic_tracking_mode': mode})


def test_cpu_prefix_end_cannot_qualify_actual_live_end() -> None:
    calls: list[str] = []
    repeated, value = chain(calls), state(calls)
    assert K.FRAMES[-1] == 36298 and R.END == 34980
    value['repeat_scope_guard'].frame = R.END
    with pytest.raises(ValueError, match='^same_scope_not_closed$'):
        S.clone(repeated.verify)(value)
    assert calls == []


def test_actual_mode_initializer_is_unselected() -> None:
    path = S.BASE / 'g2_reset_settled_basis_gate_2026-09-11_v1/tracking_mode.py'
    compiled = compile(path.read_bytes(), str(path), 'exec', dont_inherit=True)
    code = next(c for c in S.codes(compiled) if c.co_name == '__init__')
    initialize = FunctionType(code, {})
    mode, value = N(), {}
    initialize(mode, N(binding=None), value, None)
    value['probabilistic_tracking_mode'] = mode
    assert mode.state is value and not O.P.tracking_selected(value)


def test_actual_mode_type_selection(monkeypatch: Any) -> None:
    for directory in ('g2_probabilistic_scope_candidate_2026-09-11_v1',
                      'g2_reset_settled_basis_gate_2026-09-11_v1'):
        monkeypatch.syspath_prepend(str(S.BASE / directory))
    module = importlib.import_module('tracking_mode')
    assert module.__file__ == str(S.BASE / 'g2_reset_settled_basis_gate_2026-09-11_v1/tracking_mode.py')
    value: dict[str, Any] = {}
    connection = N(binding=None)
    value['probabilistic_tracking_mode'] = module.Mode(connection, value, None)
    assert not O.P.tracking_selected(value)
    connection.binding = object()
    assert O.P.tracking_selected(value)


def test_binding_without_activation_named_rejection() -> None:
    calls: list[str] = []
    repeated, value = chain(calls), state(calls)
    value['probabilistic_tracking_mode'].native = None
    value['probabilistic_tracking_mode'].activation = None
    with pytest.raises(AssertionError, match='^probability_binding_without_activation$'):
        S.clone(repeated.verify)(value)
    assert calls == []
