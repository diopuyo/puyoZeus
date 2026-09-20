"""既存SM保留選択への次手資格追加と、小さい原ソース境界検査。"""
from __future__ import annotations
import ast
from pathlib import Path
from types import SimpleNamespace as N
import pytest
from test_conditional import current
import hidden_current_connection as H


def test_conditional_next_hand_enters_nonstable() -> None:
    _,state,binding = current()
    binding.owner,binding.next_token = N(state=state),'real-token-fixture'
    sm,signals = N(context=N(state=N(value='stable'))),N(is_match_active=True,chain_event=None)
    assert H.next_fall(lambda *args:False,sm,signals,binding)
    binding.next_token = None
    assert not H.next_fall(lambda *args:False,sm,signals,binding)
    assert H.next_fall(lambda *args:True,sm,signals,binding)


@pytest.mark.parametrize('case',('inactive','chain','already_falling','bad_correspondence'))
def test_new_next_gate_rejects_unqualified(case: str) -> None:
    _,state,binding = current()
    binding.owner,binding.next_token = N(state=state),'token'
    sm,signals = N(context=N(state=N(value='stable'))),N(is_match_active=True,chain_event=None)
    if case=='inactive': signals.is_match_active=False
    if case=='chain': signals.chain_event=object()
    if case=='already_falling': sm.context.state.value='tsumo_fall'
    if case=='bad_correspondence': binding.current=()
    assert not H.next_fall(lambda *args:False,sm,signals,binding)


def test_functions_bounded_and_annotated() -> None:
    for path in Path(__file__).resolve().parent.glob('*.py'):
        for node in ast.walk(ast.parse(path.read_bytes())):
            if isinstance(node,ast.FunctionDef):
                assert node.returns is not None,(path.name,node.name)
                assert node.end_lineno-node.lineno+1<=50,(path.name,node.name,node.end_lineno-node.lineno+1)
