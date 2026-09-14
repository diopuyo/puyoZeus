"""原factoryの確率型を共有し、公開接続を既存の専用aliasローダーで読む。"""
from __future__ import annotations

from pathlib import Path
import sys
from typing import Any,Callable

ROOT=Path(__file__).resolve().parent
MATH_ROOT=ROOT.parent/'g2_probabilistic_scope_candidate_2026-09-11_v1'
ORDER=('journal_context','journal_witness','live_binding','reflection','live_binding_v2',
       'second_observation','second_basis','second_retirement','second_tracking','second_physical',
       'reflection_v3','live_binding_v3','sidecar','trained_connection','trained_sidecar','live_session')


def modules(state: Any,load: Callable[...,Any]) -> dict[str,Any]:
    first=state['probabilistic_basis_connection']
    base=type(first).__init__.__globals__
    native=sys.modules['_g2_tracking_native']
    assert native.B is base['B'],'publication_native_belief_identity'
    values=dict(belief=base['B'],conditioning=base['C'],serialization=base['S'],
                native_consumption=native)
    values['joint_evaluation']=load('_g2_pub_runtime_joint',MATH_ROOT/'joint_evaluation.py',values)
    for name in ORDER:
        values[name]=load('_g2_pub_runtime_'+name,ROOT/(name+'.py'),values)
    assert values['live_binding'].B is base['B'],'publication_registry_type_identity'
    assert values['second_basis'].B is base['B'],'publication_second_type_identity'
    assert values['reflection_v3'].P is values['second_physical'],'publication_mode_type_identity'
    return values
