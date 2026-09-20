"""元live_identityの整数正常/確率退役拒否を確認。実factoryの代替ではない。"""
from __future__ import annotations
import ast
from dataclasses import asdict, dataclass
from pathlib import Path
from types import CodeType, FunctionType, SimpleNamespace as N
from typing import Any
import pytest

PATH = Path(__file__).resolve().parent.parent / 'g2_conditional_finalizer_compatibility_2026-09-10_v1/compat.py'


@dataclass
class State:
    frame: int = 34930


def actual() -> Any:
    compiled = compile(PATH.read_bytes(), str(PATH), 'exec', dont_inherit=True)
    code = next(c for c in compiled.co_consts if isinstance(c, CodeType) and c.co_name == 'live_identity')
    def require(ok: Any, reason: str) -> None:
        if not ok:
            raise AssertionError(reason)
    ns = dict(__file__=str(PATH), __builtins__=__builtins__, F=N(require=require), asdict=asdict, Any=Any)
    ns['live_identity'] = FunctionType(code, ns)
    return ns['live_identity']


def sample() -> tuple[Any, Any, Any, Any, Any]:
    binding = N(owner=N(state=State()))
    provider = N(journal=N(controller=object()))
    control = N(history={'1P': binding}, provider=provider)
    factory = N(controller=control, provider=provider)
    rows = [dict(scope=dict(side='1P', frame_idx=34930), decision=dict(history_state=asdict(State())))]
    def same(left: Any, right: Any, reason: str) -> None:
        if left != right:
            raise AssertionError(reason)
    return control, factory, rows, N(E=N(same=same)), binding


def test_original_normal_and_retired_failure() -> None:
    control, factory, rows, module, _ = sample()
    original = actual()
    assert original(control, factory, rows, module) is True
    del control.history['1P']
    with pytest.raises(KeyError, match='1P'):
        original(control, factory, rows, module)


@pytest.mark.parametrize('defect,reason', [('controller', 'live_controller_identity'),
    ('provider', 'live_controller_identity'), ('state', 'live_final_state'), ('journal', 'live_J_missing')])
def test_original_guard_rejections(defect: str, reason: str) -> None:
    control, factory, rows, module, _ = sample()
    if defect == 'controller':
        factory.controller = object()
    elif defect == 'provider':
        factory.provider = object()
    elif defect == 'state':
        rows[-1]['decision']['history_state']['frame'] += 2
    else:
        factory.provider.journal.controller = None
    with pytest.raises(AssertionError, match='^' + reason + '$'):
        actual()(control, factory, rows, module)


def test_original_missing_parts_contract() -> None:
    control, factory, rows, module, _ = sample()
    assert actual()(None, None, rows, module) is False
    with pytest.raises(AssertionError, match='^live_parts_missing$'):
        actual()(control, None, rows, module)
