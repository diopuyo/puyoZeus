"""単発着弾の死断定も整合済み入力を必要とし、OFF互換を保つ。"""
from __future__ import annotations

from types import SimpleNamespace as NS

import numpy as np
import pytest

from src.board import Board
from src.exchange_event_landing import ExchangeLandingProjection
from src.exchange_event_overlay import ExchangeEventOverlay

QUEUE = np.asarray((1, 2, 3, 4))
INCOMING = 106
STAMP = 10.0


def scenario(enabled: bool, monkeypatch: pytest.MonkeyPatch) -> tuple:
    """応手不足で旧単発経路だけが死亡する盤面を用意する。"""
    projection = ExchangeLandingProjection(single_death_proof_guard=enabled)
    board = Board()
    board._grid[-8:, :] = 9
    boards = (board, Board())
    latest = tuple(NS(board=b, queue=QUEUE) for b in boards)
    tracker = NS(_score_elapsed=0., latest_chain=lambda side: None)
    monkeypatch.setattr(projection, '_verified_attack', lambda *a: True)
    monkeypatch.setattr('src.exchange_event_landing.future_send', lambda *a: 0.)
    return projection, NS(tracker=tracker), latest, boards


def metrics(projection: ExchangeLandingProjection, overlay: NS,
            latest: tuple, boards: tuple) -> dict:
    """確率合成と無関係の死亡経路を直接検査する。"""
    return projection._single_death_metrics(overlay, latest, [INCOMING, 0], (4, 1),
        STAMP, boards, boards, [True, True], [0, 0], None)


def test_off_preserves_legacy_certainty(monkeypatch: pytest.MonkeyPatch) -> None:
    assert metrics(*scenario(False, monkeypatch))['dead_sides'] == ['1P']


def test_missing_stable_board_cannot_certify(monkeypatch: pytest.MonkeyPatch) -> None:
    assert metrics(*scenario(True, monkeypatch))['dead_sides'] == []


def test_newly_changed_board_waits_for_quiet(monkeypatch: pytest.MonkeyPatch) -> None:
    projection, overlay, latest, boards = scenario(True, monkeypatch)
    projection.safety.guards[0].observe(boards[0], tuple(QUEUE), 0., STAMP)
    assert metrics(projection, overlay, latest, boards)['dead_sides'] == []
    projection.safety.guards[0].since = STAMP-1.
    monkeypatch.setattr('src.exchange_single_death_proof.cached_proof',
                        lambda *a: dict(dead=True, reason='all_responses_dead'))
    assert metrics(projection, overlay, latest, boards)['dead_sides'] == ['1P']


@pytest.mark.parametrize('reason', ['surviving_response', 'node_limit', 'optimistic_cancel'])
def test_firepower_shortage_alone_is_not_a_proof(reason: str, monkeypatch: pytest.MonkeyPatch) -> None:
    projection, overlay, latest, boards = scenario(True, monkeypatch)
    projection.safety.guards[0].observe(boards[0], tuple(QUEUE), 0., STAMP-1.)
    monkeypatch.setattr('src.exchange_single_death_proof.cached_proof',
                        lambda *a: dict(dead=False, reason=reason))
    value = metrics(projection, overlay, latest, boards)
    assert value['required_cancel'][0] > value['near_future_send'][0]
    assert value['dead_sides'] == [] and value['proofs'][0]['reason'] == reason


def test_score_mismatch_invalidates_held_death(monkeypatch: pytest.MonkeyPatch) -> None:
    projection, overlay, latest, boards = scenario(True, monkeypatch)
    before = boards[0].copy()
    before._grid[-1, 0] = 1
    guard = projection.safety.guards[0]
    guard.observe(before, tuple(QUEUE), 0., STAMP-2.)
    guard.observe(boards[0], tuple(QUEUE), 0., STAMP-1.)
    projection.death = dict(dead_sides=['1P'])
    assert guard.reason == 'color_score_mismatch'
    assert metrics(projection, overlay, latest, boards)['dead_sides'] == []


def test_flag_is_default_off_and_forwarded() -> None:
    default = ExchangeEventOverlay(NS(), lambda *a: None, lambda *a: None)
    enabled = ExchangeEventOverlay(NS(), lambda *a: None, lambda *a: None,
                                  single_death_proof_guard=True)
    assert not default._landing_projection.single_death_proof_guard
    assert enabled._landing_projection.single_death_proof_guard
    assert enabled._landing_projection.safety.guard_enabled
    assert not enabled._landing_projection.safety.ledger_enabled
