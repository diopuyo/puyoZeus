"""既存実型fixtureで2P専用反映gateの制限を再現。実v66 capture再現ではない。"""
from pathlib import Path
import sys
from typing import Any
import pytest

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / 'g2_belief_live_publication_2026-09-11_v1'))
from test_initial_reflection import unchanged, saved, live, context, policy, observed, physical
import reflection_v3 as R


def test_physical_mode_type_cannot_use_second_side_initial_branch(unchanged: Any) -> None:
    second, value = unchanged
    first_type = second.physical
    native = first_type.native
    assert first_type.connection.registry.current(first_type.connection.binding) is value
    assert not first_type.applied and not native.pending
    assert native.last_frame == 35372 and 'step:601' in native.seen_calls
    assert type(first_type) is not R.P.Mode
    with pytest.raises(ValueError, match='initial_reflection_mode'):
        R.verify(first_type, value, 'step:601', 35372)
