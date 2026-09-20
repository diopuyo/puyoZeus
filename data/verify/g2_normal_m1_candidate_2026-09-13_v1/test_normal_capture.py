"""実J/Registry/反射/原RowInputs→保存入口まで。モデル実行前で意図して停止する。"""
from contextlib import ExitStack
import json
from pathlib import Path
from types import SimpleNamespace as N
from typing import Any
import pytest
import test_normal_graph as G
import session_fixture as F
parts, graph = G.parts, G.graph
ROOT = Path(__file__).resolve().parent


def test_session_original_capture_before_model(parts: Any, graph: Any, tmp_path: Path,
                                              monkeypatch: Any) -> None:
    f = F.create(parts, graph, tmp_path)
    contract = parts.loader('_normal_capture_contract', ROOT.parent /
                            'g2_model_process_bridge_2026-09-11_v1/pure_contract.py')
    captured = []
    def before_model(capture: Any, members: Any, path: Path, *, seed: int) -> dict:
        bound = capture()
        assert bound.frame == seed == 33726 and bound.tokens == ('step:4', 'step:5')
        assert type(bound.inputs) is contract.RowInputs
        assert [value.scope[-1] for value in bound.values] == ['1P', '2P']
        captured.append(bound)
        raise LookupError('planned_before_actual_model')
    monkeypatch.setattr(graph['normal_session'].SAVE, 'run', before_model)
    with pytest.raises(LookupError, match='planned_before_actual_model'):
        with ExitStack() as stack:
            session = graph['normal_session'].Session(stack, dict(state=f.state, factory=f.factory, pipe=f.pipe),
                parts.policy, N(Mode=parts.original.mode.BASE.Mode), contract, None, (33726, 33766))
            for frame in (33722, 33724, 33726):
                F.update(f, session, frame)
    assert len(captured) == 1 and session.owner.closed and not session.saved
    assert not (tmp_path / 'BELIEF_M1_33726.json').exists()
    rows = [json.loads(line) for line in (tmp_path / 'M1_CAPTURE_SCHEDULE.jsonl').read_text().splitlines()]
    assert rows[-1]['action'] == 'REQUEST' and rows[-1]['saved'] is False
    assert 'planned_before_actual_model' in rows[-1]['error']
