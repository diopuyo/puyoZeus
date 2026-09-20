"""原aliasローダーと別Belief型で、専用公開接続の型共有を検査する。"""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace as N
from typing import Any
import numpy as np
import check_saved_inputs as INPUT
import runtime_loader as R


def test_private_belief_identity_through_loader(monkeypatch: Any) -> None:
    root=Path(__file__).resolve().parent.parent
    spec=importlib.util.spec_from_file_location('_publication_test_original_loader',
        root/'g2_reset_inflight_quarantine_2026-09-11_v1/inflight_loader.py')
    loader=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=loader
    spec.loader.exec_module(loader)
    math=root/'g2_probabilistic_scope_candidate_2026-09-11_v1'
    b=loader.load('_publication_test_private_belief',math/'belief.py',
                  {'frozen_physics':sys.modules['frozen_physics']})
    values={'belief':b}
    for name in ('conditioning','serialization','registry','native_consumption'):
        values[name]=loader.load('_publication_test_private_'+name,math/(name+'.py'),values)
    base=loader.load('_publication_test_private_binding',
        root/'g2_reset_settled_basis_gate_2026-09-11_v1/basis_registry_connection.py',
        values|{'actual_connection_v2':N()})
    # constructor未実行の原Connection型。実factory全体の合格にはしない。
    first=base.Connection.__new__(base.Connection)
    monkeypatch.setitem(sys.modules,'_g2_tracking_native',values['native_consumption'])
    modules=R.modules({'probabilistic_basis_connection':first},loader.load)
    assert modules['live_binding'].B is b and b.Belief is not INPUT.S.B.Belief
    board=b.Board()
    states=tuple(b.establish(('source','run',0,1,i,0,side),0,10,board,b.ProbabilisticBoard.from_board(board))
                 for i,side in ((2,'1P'),(3,'2P')))
    result=modules['joint_evaluation'].evaluate(states,2,('STABLE','STABLE'),(board,board),
        lambda batch:np.full(len(batch),0.5),sample_count=4,seed=1)
    assert result.probability_p1==0.5 and not result.quality_gate_clear
    for name in ('second_basis','second_retirement','reflection_v3'):
        assert modules[name].S is values['serialization']
