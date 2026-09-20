"""元NEXT変換fixture/元bound_updateで新出口のclosure到達性を検査する。"""
from __future__ import annotations

from contextlib import ExitStack
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace as N
from typing import Any

import pytest
from scripts import g3_current_exit as C
from tests.test_g3_observer_scope import owned_replace


@pytest.mark.parametrize('changed', [False, True])
def test_actual_next_binding(tmp_path: Path, changed: bool) -> None:
    """実変換への到達は保ち、偽の同名同path変換は元guardが拒否する。"""
    root = Path(__file__).resolve().parents[1]
    path = root / 'data/verify/g2_firing_transform_independent_2026-09-10_v1/test_transform.py'
    paths = list(sys.path)
    try:
        spec = importlib.util.spec_from_file_location('_g3_existing_transform_fixture', path)
        fixture = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(fixture)
        pipe, controller, actual = fixture.fixture()
        assert fixture.Q.bound_update(pipe) is actual
        if changed:
            type(pipe).update = fixture.fake_code(actual, actual.__wrapped__)
        before = type(pipe).update
        observer = N(output=tmp_path, frames=0, active=None)
        state = {C.A.KEY: observer}
        with ExitStack() as stack:
            output = C.install(stack, N(RecognitionPipeline=type(pipe)), state, owned_replace())
            if changed:
                with pytest.raises(AssertionError, match='firing_final_NEXT_transform_changed'):
                    fixture.Q.bound_update(pipe)
            else:
                assert fixture.Q.bound_update(pipe) is actual
                assert actual.__globals__['__next_live'] is controller
        assert type(pipe).update is before and output.stream.closed
    finally:
        sys.path[:] = paths
