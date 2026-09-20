"""resetのstate key無しで正常module graphを実loadし、型/窓/復元を検査する。"""
from contextlib import ExitStack
from pathlib import Path
from typing import Any
import pytest
import test_normal_basis as F
parts = F.parts
ROOT = Path(__file__).resolve().parent


@pytest.fixture(scope='module')
def graph(parts: Any) -> Any:
    settings = parts.loader('_normal_graph_settings', ROOT / 'normal_settings.py')
    loader = parts.loader('_normal_graph_loader', ROOT / 'normal_loader.py',
                          dict(normal_dependencies=parts.original, normal_settings=settings))
    with ExitStack() as stack:
        values = loader.modules(dict(joint_capture_stack=stack), parts.loader)
        yield values
    assert values['capture_schedule'].END == 36298
    assert values['capture_schedule'].EARLIEST == (35370, 35410)


def test_normal_graph_without_reset_keys(parts: Any, graph: Any) -> None:
    assert graph['normal_owner'].B is parts.original.binding.B
    assert graph['normal_owner'].R is parts.original.binding.OLD.R
    assert graph['reflection_v3'].P is graph['second_physical']
    assert graph['normal_session'].P is graph['second_physical']
    assert graph['normal_eligibility'].P is graph['second_physical']
    assert graph['normal_session'].PLAN is graph['capture_schedule']
    assert graph['capture_schedule'].END == 34290 and graph['capture_schedule'].EARLIEST == (33726, 33766)
    assert graph['live_session'].Session is graph['normal_session'].Session
    assert graph['second_tracking'].S is graph['second_basis']


@pytest.mark.parametrize('side,index', [('1P', 0), ('2P', 1)])
def test_graph_qualifier_keeps_side(parts: Any, graph: Any, side: str, index: int) -> None:
    f = F.fixture(parts)
    row, grid = graph['second_basis'].qualify(f.evidence[index], f.witness, f.pipe, side=side)
    assert row['scope'][-1] == side
