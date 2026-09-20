"""原transaction.inferの一段輸送を含む原着地ブロックで旧→新を比較。"""
from __future__ import annotations
import ast
from contextlib import ExitStack
from pathlib import Path
import sys
from types import MethodType, SimpleNamespace as N
from typing import Any
import pytest
import test_quarantine as F
import quarantine_v3 as OLD
import quarantine_v4 as NEW


def fixture(transport: bool = True) -> tuple[Any, Any, Any, Any]:
    ns, r, raw = F.setup()
    path = F.ROOT.parent / 'g2_normal_completion_transaction_2026-09-09_v1/transaction.py'
    method = next(n for n in ast.walk(ast.parse(path.read_bytes())) if isinstance(n, ast.FunctionDef) and n.name == 'infer')
    values = dict(sys=sys, Any=Any, require=lambda ok, reason: None)
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), 'exec'), values)
    r.control = N(tickets={})
    r.control.infer = MethodType(values['infer'], r.control)
    tree = ast.parse(F.SOURCE.read_bytes())
    nodes = [n for n in ast.walk(tree) if isinstance(n, ast.If) and n.lineno == 7426]
    if transport:
        call = next(n for n in ast.walk(nodes[0]) if isinstance(n, ast.Call) and ast.unparse(n.func) == 'infer_placement')
        call.args.insert(0, ast.Name(id='infer_placement', ctx=ast.Load()))
        call.func = ast.Attribute(value=ast.Name(id='CONTROL', ctx=ast.Load()), attr='infer', ctx=ast.Load())
    tree = ast.Module(body=ast.parse('REGISTER(sys._getframe())').body + nodes, type_ignores=[])
    code = compile(ast.fix_missing_locations(tree), str(F.SOURCE) + '#transport-test', 'exec')
    r.journal.codes.add(code)
    scope = dict(frame_idx=35172, time_sec=35172 / 60, side='1P')
    r.journal.scope = lambda *_: scope
    r.journal.epoch = lambda *_: r.pending['epoch']
    ns.update(CONTROL=r.control, REGISTER=lambda frame: setattr(r.journal, 'active',
              dict(frame=frame, pipe=r.pipe, token='fixture-J', scope=scope, epoch=r.pending['epoch'])))
    return ns, r, raw, code


@pytest.mark.parametrize('transport', (False, True))
def test_original_body_transport_keeps_raw(transport: bool) -> None:
    ns, r, raw, code = fixture(transport)
    previous = ns['infer_placement']
    counter = dict(r.pipe._tsumo_count_1p)
    with ExitStack() as stack:
        guard = NEW.install(stack, r, ns)
        exec(code, ns)
        assert len(guard.rows) == 1 and guard.rows[0]['passed_pair'] is None
        assert ns['inferred_landing'] is None and ns['ctx'].confirmed_board.to_dict()['grid'] == raw
    assert ns['infer_placement'] is previous and dict(r.pipe._tsumo_count_1p) == counter


def test_old_guard_silently_skipped_transaction_transport() -> None:
    ns, r, raw, code = fixture()
    with ExitStack() as stack:
        guard = OLD.install(stack, r, ns)
        # 旧版は推論を実行し、切り出したfixtureにない後段文脈へ到達する。
        # 後段全体の成功は主張せず、原infer返却値までを反例として照合する。
        with pytest.raises(NameError, match='_effect_gate_window_active'):
            exec(code, ns)
        assert not guard.rows and ns['inferred_landing'] is not None
        assert ns['inferred_landing'].to_dict()['grid'] != raw


def test_foreign_intermediate_is_not_unwrapped() -> None:
    ns, r, _, code = fixture()
    original = r.control.infer
    def bridge(*args: Any, **kwargs: Any) -> Any:
        return original(*args, **kwargs)
    ns['CONTROL'] = N(infer=bridge)
    with ExitStack() as stack:
        NEW.install(stack, r, ns)
        with pytest.raises(RuntimeError, match='transaction_original_J'):
            exec(code, ns)
