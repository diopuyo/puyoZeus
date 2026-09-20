"""原producer/consumer本体の契約検査。人工carrierであり実constructor合格ではない。"""
from __future__ import annotations
import ast
import json
from pathlib import Path
from types import SimpleNamespace as N
from typing import Any
import pytest

VERIFY = Path(__file__).resolve().parent.parent
LIVE = VERIFY / 'g2_empty_tail_reset_live_adapter_2026-09-11_v1'
PRIVATE = VERIFY / 'g2_private_suffix_live_adapter_2026-09-10_v1'


def functions(path: Path, names: tuple[str, ...]) -> dict[str, Any]:
    tree = ast.parse(path.read_bytes())
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in nodes} == set(names)
    namespace = dict(Any=Any, json=json)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace


@pytest.mark.parametrize('baseline,pending,expected', [(0, None, False), (1, None, True), (1, {}, False)])
def test_live_close_actual_field(tmp_path: Path, baseline: int, pending: Any, expected: bool) -> None:
    api = functions(LIVE / 'proof_status.py', ('recovery_status', 'close'))
    recovery = N(error=None, failure=None, reset_count=1, baseline_count=baseline,
                 wait_count=7, pending=pending, rows=[])
    carrier = N(state=dict(output=tmp_path), rows=[], recovery=recovery, error=None)
    api['close'](carrier)
    value = json.loads((tmp_path / 'LIVE_EMPTY_RESET.json').read_bytes())
    assert value['baseline_recovered'] is expected
    assert value['recovery_status']['baseline_count'] == baseline
    assert value['quality_gate_clear'] is False


def test_cpu_close_does_not_produce_live_field(tmp_path: Path) -> None:
    tree = ast.parse((LIVE / 'proof.py').read_bytes())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Context')
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'close')
    namespace = dict(json=json)
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(LIVE / 'proof.py'), 'exec'), namespace)
    namespace['close'](N(state=dict(output=tmp_path), rows=[], error=None, recovery=None))
    value = json.loads((tmp_path / 'LIVE_EMPTY_RESET.json').read_bytes())
    assert 'baseline_recovered' not in value
    with pytest.raises(KeyError, match='baseline_recovered'):
        _ = value['baseline_recovered']


def carrier() -> tuple[Any, dict[str, Any], list[str]]:
    calls: list[str] = []
    factory = N(provider=object())
    factory.controller = N(provider=factory.provider, calls=[], tickets=[])
    state = dict(repeated_firing_constructor=dict(installed=True, closed=True, references_restored=True),
                 conditional_runtime_factory=factory, private_suffix_factory=factory,
                 combined_restored=lambda: calls.append('combined'), conditional_full_installs=1,
                 conditional_revision_connection=dict(installs=1), combined_install=dict(configure_calls=1))
    def verify(value: Any) -> None:
        assert value is state
        calls.append('private')
    state['private_suffix_modules'] = N(verify=verify)
    return factory, state, calls


def test_original_context_normal_control() -> None:
    factory, state, calls = carrier()
    actual = functions(PRIVATE / 'live_finalizer.py', ('context',))['context']
    assert actual(state) is factory
    assert calls == ['combined', 'private']


@pytest.mark.parametrize('field', ['installed', 'closed', 'references_restored'])
def test_original_context_rejects_unqualified_constructor(field: str) -> None:
    _, state, calls = carrier()
    state['repeated_firing_constructor'][field] = False
    actual = functions(PRIVATE / 'live_finalizer.py', ('context',))['context']
    with pytest.raises(AssertionError):
        actual(state)
    assert calls == []


def test_original_context_missing_constructor_fails_before_saved_status() -> None:
    _, state, calls = carrier()
    del state['repeated_firing_constructor']
    actual = functions(PRIVATE / 'live_finalizer.py', ('context',))['context']
    with pytest.raises(KeyError, match='repeated_firing_constructor'):
        actual(state)
    assert calls == []


@pytest.mark.parametrize('defect', ['factory', 'provider', 'calls', 'tickets', 'installs', 'configure'])
def test_original_context_keeps_remaining_rejections(defect: str) -> None:
    factory, state, _ = carrier()
    if defect == 'factory':
        state['private_suffix_factory'] = object()
    elif defect == 'provider':
        factory.controller.provider = object()
    elif defect in ('calls', 'tickets'):
        setattr(factory.controller, defect, [object()])
    elif defect == 'installs':
        state['conditional_full_installs'] = 2
    else:
        state['combined_install']['configure_calls'] = 2
    actual = functions(PRIVATE / 'live_finalizer.py', ('context',))['context']
    with pytest.raises(AssertionError):
        actual(state)
