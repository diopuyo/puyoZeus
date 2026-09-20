"""元J→左右観測→正常Session基準→失敗保存/解放。M1採録前の限定CPU。"""
from contextlib import ExitStack
import json
from pathlib import Path
from types import SimpleNamespace as N
from typing import Any
import pytest
import test_normal_graph as G
import session_fixture as F
parts, graph = G.parts, G.graph


def test_actual_journal_session_basis_and_error_close(parts: Any, graph: Any, tmp_path: Path) -> None:
    f = F.create(parts, graph, tmp_path)
    original = f.journal.complete_step
    with pytest.raises(LookupError, match='planned_before_M1'):
        with ExitStack() as stack:
            session = graph['normal_session'].Session(stack,
                dict(state=f.state, factory=f.factory, pipe=f.pipe), parts.policy,
                N(Mode=parts.original.mode.BASE.Mode), None, None, (33726, 33766))
            F.update(f, session, 33722)
            assert session.current_modes == [None, None] and not session.owner.registry._bindings
            F.update(f, session, 33724)
            assert [m.side for m in session.current_modes] == ['1P', '2P']
            assert all(m.initial_receipt['state']['frame'] == 33724 for m in session.current_modes)
            assert f.journal.count == 4 and session.witness.pair(33724)[0]['token'] == 'step:2'
            assert not session.saved
            raise LookupError('planned_before_M1')
    assert f.journal.complete_step == original and session.owner.closed
    assert all(e.closed and e.error is None for e in session.evidence)
    assert all(m.closed and m.error is None for m in session.modes)
    saved = json.loads((tmp_path / 'BELIEF_M1_SESSION.json').read_bytes())
    assert len(saved['modes']) == 2 and 'planned_before_M1' in saved['error']
    assert saved['saved'] == [] and saved['quality_gate_clear'] is False
