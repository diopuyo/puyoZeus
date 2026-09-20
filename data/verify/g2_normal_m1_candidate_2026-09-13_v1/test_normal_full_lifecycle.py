"""正常driver→元J→capture→人工保存二件→元終了を全prefixで検査。実モデル/動画ではない。"""
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
FRAMES = tuple(range(29052, 34292, 2))


def run(parts: Any, graph: Any, output: Path, monkeypatch: Any, *, missing: bool = False) -> Any:
    f = F.create(parts, graph, output)
    contract = parts.loader('_normal_full_contract', ROOT.parent / 'g2_model_process_bridge_2026-09-11_v1/pure_contract.py')
    original = parts.loader('_normal_full_original_driver', ROOT.parent / 'g2_live_probability_context_2026-09-12_v1/whole_session_driver.py')
    normal = parts.loader('_normal_full_driver', ROOT / 'normal_driver.py')
    def save(capture: Any, members: Any, path: Path, *, seed: int) -> dict:
        bound = capture()
        assert bound.frame == seed
        value = dict(frame=seed, artificial_model=True, quality_gate_clear=False)
        path.write_text(json.dumps(value))
        return value
    monkeypatch.setattr(graph['normal_session'].SAVE, 'run', save)
    def create(stack: Any, context: Any) -> Any:
        return graph['normal_session'].Session(stack, context, parts.policy,
            N(Mode=parts.original.mode.BASE.Mode), contract, None, (33726, 33766))
    driver = normal.driver_class(original)(f.state, create, (33726, 33766))
    bridge = N(state=f.state, capture=N(last=FRAMES[0]), completed_frames=[FRAMES[0]], lifetime=ExitStack(),
               initial=dict(pipeline=f.pipe), consumer=driver, frames=FRAMES)
    try:
        f.state['provisional_context_observer'].rows.append(F.context_row(f, FRAMES[0]))
        driver(bridge, FRAMES[0])
        for frame in FRAMES[1:]:
            if missing and frame == FRAMES[2]:
                continue
            for side in ('1P', '2P'):
                F.step(f, side, frame)
            f.state['provisional_context_observer'].rows.append(F.context_row(f, frame))
            bridge.capture.last = frame
            bridge.completed_frames.append(frame)
            driver(bridge, frame)
    except BaseException as error:
        bridge.lifetime.__exit__(type(error), error, error.__traceback__)
        raise
    bridge.lifetime.close()
    assert driver.closed and driver.stopped and driver.session.owner.closed
    assert bridge.consumer is original.forbidden_collect
    return driver.session


def test_full_prefix_normal_session_closes(parts: Any, graph: Any, tmp_path: Path, monkeypatch: Any) -> None:
    session = run(parts, graph, tmp_path, monkeypatch)
    assert tuple(value['frame'] for value in session.saved) == (33726, 33766)
    assert session.schedule.last == 34290 and session.schedule_rows == len(FRAMES) - 1
    saved = json.loads((tmp_path / 'BELIEF_M1_SESSION.json').read_bytes())
    assert saved['error'] is None and saved['session_error'] is None and saved['restored']
    assert not (tmp_path / 'M1_SESSION_CLOSE_FAILURE.json').exists()


def test_missing_prefix_is_correctly_rejected(parts: Any, graph: Any, tmp_path: Path, monkeypatch: Any) -> None:
    with pytest.raises(ValueError, match='continuous_clock'):
        run(parts, graph, tmp_path, monkeypatch, missing=True)
    saved = json.loads((tmp_path / 'BELIEF_M1_SESSION.json').read_bytes())
    assert saved['saved'] == [] and 'continuous_clock' in saved['error']
