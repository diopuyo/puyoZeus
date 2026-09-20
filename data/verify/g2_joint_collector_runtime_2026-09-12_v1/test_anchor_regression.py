"""旧原collect fixtureの開始/遅延/未開始を新捕捉器でも維持する。"""
from __future__ import annotations
from pathlib import Path
import sys
from types import FunctionType
from typing import Any
import pytest
import collector_connector as C

sys.path.insert(0, str(C.ROOT.parent / 'g2_observed_start_anchor_2026-09-12_v1'))
import test_anchor as T
real = T.real


@pytest.mark.parametrize('mode,qualified', [('good', 90), ('delayed', 94), ('no_boundary', None)])
def test_original_positive_and_delayed_start(real: Any, tmp_path: Path, mode: str, qualified: int | None) -> None:
    run = FunctionType(T.run.__code__, dict(vars(T), A=C.ANCHOR), argdefs=T.run.__defaults__)
    value = run(real, tmp_path / mode, mode)
    assert value['unavailable_boards'] == []
    assert value['start_observation_eligible'] is (qualified is not None)
    if qualified is not None:
        assert value['start_anchor']['qualification_frame'] == qualified
    assert all(r['checks']['board_present'] for r in value['stable_snapshots'])
