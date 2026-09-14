"""元geometryとNEXT決定関数で、基準onlyに二手motionを混ぜた矛盾を再現する。"""
from __future__ import annotations
from dataclasses import replace
from pathlib import Path
import sys
from types import SimpleNamespace as N
from typing import Any
import pytest

ROOT = Path(__file__).resolve().parent.parent
for name in ('g2_directional_next_enqueue_2026-09-09_v2', 'g2_reset_recovery_candidate_2026-09-10_v1'):
    sys.path.insert(0, str(ROOT / name))
import occurrence as O
import reset_inputs as I
from scripts import next_enqueue_live_shadow_v1 as LIVE


def view(frame: int) -> Any:
    invocation = N(frame=frame, time_sec=frame / 60, runtime=N(histories={'1P': N(epoch=1)}))
    return I.geometry(O, invocation, '1P')


def state() -> dict[str, Any]:
    value = O.empty_state(1, 'artificial-reset-segment:1')
    _, commit, reason = O.decide(value, view(I.QUIET[1]), True, (4, 5), (3, 3), None, LIVE)
    assert not commit and reason == 'initial_baseline'
    return value


def test_old_basis_only_motion_competes_with_uncommitted_candidate() -> None:
    value = state()
    _, commit, reason = O.decide(value, view(I.FIRST), True, (4, 5), (3, 3), (4, 5), LIVE)
    assert not commit and reason == 'await_dnext_successor' and value['pending'] is not None
    with pytest.raises(ValueError, match='ambiguous_two_episodes'):
        O.decide(value, view(I.SECOND), True, (4, 5), (3, 3), (4, 5), LIVE)


def test_no_hand_scenario_without_motion_has_no_commit() -> None:
    value = state()
    for frame in (I.FIRST, I.SECOND):
        observation = replace(view(frame), candidate=None)
        _, commit, reason = O.decide(value, observation, True, (4, 5), (3, 3), (4, 5), LIVE)
        assert not commit and reason == 'await_motion' and value['pending'] is None


def test_matching_next_successor_commits_original_motion() -> None:
    value = state()
    _, commit, reason = O.decide(value, view(I.FIRST), True, (3, 3), (2, 2), (4, 5), LIVE)
    assert commit and reason == 'occurrence_committed' and value['pending'] is None
