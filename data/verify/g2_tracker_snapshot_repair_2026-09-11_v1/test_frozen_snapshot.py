"""原凍結constructorとtrackerで検査する。人工画像、原J完走とは別。"""
from __future__ import annotations
from contextlib import ExitStack, contextmanager
from dataclasses import replace
from pathlib import Path
import sys
from types import SimpleNamespace as N
import pytest
import group_snapshot as G
import reproduce as R

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / 'g2_cascade_connection_plan_2026-09-11_v1'))
import test_tracker_input as F


def test_frozen_tracker_same_depth_and_changes() -> None:
    evidence = {}
    with contextmanager(F.F.frozen.__wrapped__)() as frozen:
        with pytest.MonkeyPatch.context() as patch:
            with contextmanager(F.T.transport(F.F.real.__wrapped__, evidence))(frozen, patch) as real:
                pipe, controller, _, _, _ = real
                join = R.load_join()
                grid = [[0]*6 for _ in range(13)]
                grid[12][:4] = [1]*4
                board = frozen.Board.from_dict({'grid': grid})
                tracker = pipe._chain_tracker_2p
                tracker._simulator.simulate(board)
                with pytest.raises(join.SideJoinUninspectable, match='max_depth:PuyoGroup'):
                    join._vrepr(tracker)
                with ExitStack() as stack:
                    receipt = {}
                    G.install(stack, join, G.group_type(pipe), receipt)
                    before = join._vrepr(tracker)
                    generations = N(generation=lambda side: N(reset_epoch=0, action_revision=0))
                    runtime = controller._runtime(pipe)
                    whole = join.snapshot(pipe, generations, runtime, '1P', frozenset())
                    result = next(iter(tracker._simulator._cache.values()))
                    group = result.steps[0].erased_groups[0]
                    assert join._vrepr(group, 7) == join._vrepr(group, 2)
                    result.steps[0].erased_groups[0] = replace(group, color=2)
                    assert join._vrepr(tracker) != before
                    changed = join.snapshot(pipe, generations, runtime, '1P', frozenset())
                    assert 'side_attrs:2P' in join._violations(whole, changed, '1P')
                assert receipt['restored']
    assert evidence['tracker_constructor_restored']
