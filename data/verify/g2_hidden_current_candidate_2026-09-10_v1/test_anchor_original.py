"""原Sの旧整数slot保持と別型対応を検査。人工policyで実J権限を認証しない。"""
from __future__ import annotations
from dataclasses import replace
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace as N
from typing import Any
from test_conditional import current
import conditional_current as C

PATH = Path(__file__).resolve().parent.parent/'g2_current_accounting_split_2026-09-08_v1/split_contract.py'
SPEC = importlib.util.spec_from_file_location('_hidden_anchor_original_split',PATH)
S = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = S
SPEC.loader.exec_module(S)


def owner() -> Any:
    scope = S.Scope('a'*64,'artificial_anchor_fixture','1P','game',2)
    value = S.SplitOwner(scope,N(authorize=lambda *args:None))
    clock = S.Clock(34796,34796/60,34796*4)
    value.establish_baseline(S.BaselineEvidence(scope,clock,(0,0,0,0,0),'fixture',clock),clock)
    grid = tuple((0,)*6 for _ in range(13))
    later = S.Clock(34798,34798/60,34798*4+3)
    value.recover_current(S.CurrentEvidence(scope,0,later,later,'STABLE',grid,'old-slot'),later)
    value.advance_action(1,S.Clock(34800,34800/60,34800*4+2))
    return value


def bind(state: Any) -> tuple[Any,Any]:
    value,_,binding = current()
    proof = __import__('json').loads(value.evidence_json)
    proof.update(integer_anchor=C.anchor(state),action=state.action)
    updated = replace(value,integer_anchor=C.anchor(state),action=state.action,evidence_json=C.encoded(proof))
    binding.hidden_current = updated
    return updated,binding


def test_original_action_keeps_old_integer_anchor_content() -> None:
    original = owner()
    value,binding = bind(original.state)
    assert original.state.current.available is False and C.compatible(original.state,binding)
    old = C.anchor(original.state)
    later = original.advance_action(2,S.Clock(34860,34860/60,34860*4+2))
    assert C.anchor(later)==old and C.compatible(later,binding)
    assert later.current.grid!=value.sm_grid and later.current.evidence_id=='old-slot'


def test_new_integer_current_invalidates_old_correspondence() -> None:
    original = owner()
    _,binding = bind(original.state)
    now = S.Clock(34860,34860/60,34860*4+3)
    value = original.recover_current(S.CurrentEvidence(original.state.scope,1,now,now,
        'STABLE',original.state.current.grid,'different-slot'),now)
    assert not C.compatible(value,binding)
