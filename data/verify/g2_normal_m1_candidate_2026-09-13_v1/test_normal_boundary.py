"""1P/2Pのaction境界を区別し、元資格の他の失敗はHOLDへ隠さない。"""
from pathlib import Path
from typing import Any
import pytest
import test_normal_basis as F
parts = F.parts
ROOT = Path(__file__).resolve().parent


@pytest.mark.parametrize('index,side', [(0, '1P'), (1, '2P')])
def test_explicit_side_action_boundary(parts: Any, index: int, side: str) -> None:
    boundary = parts.loader('_normal_boundary_test', ROOT / 'second_basis_boundary.py')
    f = F.fixture(parts)
    step = f.steps[index]
    step['generation']['action_revision'] = 0
    step['generation_after']['action_revision'] = 1
    selected = boundary.wrapped(parts.basis.qualify, parts.basis)
    with pytest.raises(parts.basis.BasisHold, match='second_action_boundary_wait'):
        selected(f.evidence[index], f.witness, f.pipe, side=side)
    step['generation_after']['action_revision'] = 0
    row, grid = selected(f.evidence[index], f.witness, f.pipe, side=side)
    assert row['scope'][-1] == side
    step['generation_after']['reset_epoch'] = 1
    with pytest.raises(ValueError, match='second_basis_J'):
        selected(f.evidence[index], f.witness, f.pipe, side=side)
