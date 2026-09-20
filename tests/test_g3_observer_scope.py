"""元窓検査器と実metadataで、一体束縛・欠側拒否・二重復元を確認する。"""
from __future__ import annotations

import ast
from contextlib import ExitStack
from pathlib import Path
import sys
from types import SimpleNamespace as N
from typing import Any

import pytest
from scripts import g3_observer_scope as S
from tests.test_g3_metadata_scope import B, M, calls, collector

sys.path.insert(0, str(S.WINDOW_SOURCE.parent))
import observer_window_candidate as W


def owned_replace() -> Any:
    """元replace_ownedの関数ASTをそのまま実行し、単なるsetattrで代用しない。"""
    path = S.G.ROOT / 'data/verify/g2_owned_loader_connection_2026-09-12_v1/owned_adapter.py'
    tree = ast.parse(path.read_text())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                    and node.name == 'replace_owned')
    namespace = {'Any': Any}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace['replace_owned']


@pytest.mark.parametrize('failure', [False, True])
def test_real_replace_nested_restore(failure: bool) -> None:
    replace, owner = owned_replace(), N(value=object())
    initial, middle, final = owner.value, object(), object()
    try:
        with ExitStack() as stack:
            replace(stack, owner, 'value', middle)
            replace(stack, owner, 'value', final)
            assert owner.value is final
            if failure:
                raise LookupError('original')
    except LookupError as error:
        assert failure and str(error) == 'original'
    assert owner.value is initial


@pytest.mark.parametrize('bad', [None, 'pb', 'context', 'metadata'])
def test_original_window_guard_and_actual_metadata(collector: Any, tmp_path: Path, bad: str | None) -> None:
    replace, state = owned_replace(), dict(output=tmp_path)
    scope = N(WINDOWS=((S.OLD_FIRST, S.OLD_LAST),))
    context = N(OUTER_FIRST=S.OLD_FIRST, OUTER_LAST=S.OLD_LAST, STRIDE=S.G.STRIDE)
    with ExitStack() as outer:
        replace(outer, B.O, 'LAST', S.OLD_LAST)
        with ExitStack() as inner:
            B.install(inner, collector, None, state, enabled=True)
            receipt = S.rebind(outer, W, dict(scope=scope, context=context, metadata=B.O), replace)
            pairs = [(frame, side) for frame in W.FRAMES for side in W.SIDES]
            observer = N(expected=list(W.FRAMES), pb_expected=list(pairs))
            state.update(hidden_probability_observer=N(expected=list(pairs)),
                         current_scope_sink=N(expected=list(pairs)), provisional_context_observer=observer,
                         private_publication_consumer=N(recorder=observer))
            if bad == 'pb':
                state['hidden_probability_observer'].expected.pop()
            if bad == 'context':
                observer.expected.pop()
            if bad == 'metadata':
                replace(outer, B.O, 'FIRST', S.OLD_FIRST)
            if bad is None:
                assert W.verify_state(state)['updates'] == len(S.FRAMES)
                tail = M.install(inner, state)
                calls(collector, (0, 2))
                tail.check(2)
            else:
                with pytest.raises(ValueError, match='observer_window_'):
                    W.verify_state(state)
        assert state[B.KEY].closed
    assert receipt['inner_binding_restored']
    assert W.FIRST == S.OLD_FIRST and B.O.FIRST == S.OLD_FIRST and B.O.LAST == 36298


def test_bad_clock_does_not_partially_patch() -> None:
    scope = N(WINDOWS=((S.OLD_FIRST, S.OLD_LAST),))
    context = N(OUTER_FIRST=S.OLD_FIRST, OUTER_LAST=S.OLD_LAST, STRIDE=2)
    metadata = N(FIRST=S.OLD_FIRST, LAST=S.OLD_LAST, STRIDE=2, FPS=30)
    with ExitStack() as stack, pytest.raises(ValueError, match='observer_source_clock'):
        S.rebind(stack, W, dict(scope=scope, context=context, metadata=metadata), owned_replace())
    assert scope.WINDOWS == ((S.OLD_FIRST, S.OLD_LAST),) and W.FIRST == S.OLD_FIRST
