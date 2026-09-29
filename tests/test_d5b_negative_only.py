"""D5b: 生存枝・相殺可能の証明だけが単発死亡候補を取り消す。"""
from __future__ import annotations

from types import SimpleNamespace as NS

import numpy as np
import pytest

from src.exchange_event_overlay import ExchangeEventOverlay
from tests.test_d5_single_death_safety import metrics, scenario


def prepared(negative_only: bool, monkeypatch: pytest.MonkeyPatch, proof: dict | None,
             blocker: str | None = None) -> tuple:
    """静穏済みの単発死亡候補と、固定した証明結果/blockerを用意する。"""
    projection, overlay, latest, boards = scenario(True, monkeypatch)
    projection.single_death_proof_negative_only = negative_only
    projection.safety.guards[0].observe(boards[0], tuple(np.asarray((1, 2, 3, 4))), 0., 9.)
    monkeypatch.setattr(projection.safety, 'blocker', lambda *a: blocker)
    monkeypatch.setattr('src.exchange_single_death_proof.cached_proof', lambda *a: proof)
    return projection, overlay, latest, boards


@pytest.mark.parametrize('reason', ['surviving_response', 'optimistic_cancel'])
def test_survival_branch_cancels(reason: str, monkeypatch: pytest.MonkeyPatch) -> None:
    value = metrics(*prepared(True, monkeypatch, dict(dead=False, reason=reason)))
    assert value['dead_sides'] == [] and value['proofs'][0]['reason'] == reason


@pytest.mark.parametrize('blocker', ['board_unsettled', 'color_score_mismatch'])
def test_blocker_keeps_legacy(blocker: str, monkeypatch: pytest.MonkeyPatch) -> None:
    value = metrics(*prepared(True, monkeypatch, None, blocker))
    assert value['dead_sides'] == ['1P']
    assert value['proofs'][0]['undecided_reason'] == blocker


@pytest.mark.parametrize('reason', ['node_limit', 'landing_limit', 'unknown_board', 'invalid_start'])
def test_cutoff_keeps_legacy(reason: str, monkeypatch: pytest.MonkeyPatch) -> None:
    value = metrics(*prepared(True, monkeypatch, dict(dead=False, reason=reason)))
    assert value['dead_sides'] == ['1P']
    assert value['proofs'][0]['reason'] == 'legacy_kept'
    assert value['proofs'][0]['undecided_reason'] == reason


def test_all_dead_proof_still_confirms(monkeypatch: pytest.MonkeyPatch) -> None:
    value = metrics(*prepared(True, monkeypatch, dict(dead=True, reason='all_responses_dead')))
    assert value['dead_sides'] == ['1P']


@pytest.mark.parametrize('proof,blocker', [(dict(dead=False, reason='node_limit'), None),
                                            (None, 'board_unsettled')])
def test_flag_off_matches_d5(proof: dict | None, blocker: str | None,
                             monkeypatch: pytest.MonkeyPatch) -> None:
    value = metrics(*prepared(False, monkeypatch, proof, blocker))
    assert value['dead_sides'] == []
    assert value['proofs'][0]['reason'] == (blocker or 'node_limit')


def test_flag_default_off_needs_d5_and_is_forwarded() -> None:
    build = lambda **k: ExchangeEventOverlay(NS(), lambda *a: None, lambda *a: None, **k)._landing_projection
    assert not build().single_death_proof_negative_only
    assert not build(single_death_proof_negative_only=True).single_death_proof_negative_only
    assert build(single_death_proof_guard=True,
                 single_death_proof_negative_only=True).single_death_proof_negative_only
