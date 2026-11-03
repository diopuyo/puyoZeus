"""交換持越し・二重控除・改訂・試合境界を検証する。"""
from __future__ import annotations

import pytest
from src.exchange_pending_ledger import PendingLedger


def test_carry_across_exchange_and_counter() -> None:
    ledger = PendingLedger()
    ledger.observe(5, {(0, 2): (435, True)}, (21, 0))
    assert ledger.pending == [0, 435]
    ledger.observe(5, {(0, 2): (435, True), (1, 3): (597, True)}, (21, 0))
    assert ledger.pending == [162, 0]
    ledger.observe(5, {(0, 2): (435, True), (1, 3): (597, True), (0, 4): (236, True)}, (21, 0))
    assert ledger.pending == [0, 74]


@pytest.mark.parametrize('repeats', [1, 2, 10])
def test_repeated_totals_and_drop_not_deducted_twice(repeats: int) -> None:
    ledger = PendingLedger()
    totals = {(0, 1): (90, True)}
    ledger.observe(1, totals, (0, 0))
    for _ in range(repeats):
        ledger.observe(1, totals, (0, 30))
    assert ledger.pending == [0, 60]


def test_zero_pending_drop_does_not_consume_future_attack() -> None:
    ledger = PendingLedger()
    ledger.observe(1, {}, (0, 0))
    ledger.observe(1, {}, (0, 30))
    ledger.observe(1, {(0, 1): (100, True)}, (0, 30))
    assert ledger.pending == [0, 100]


def test_game_boundary_clears_old_pending_and_counters() -> None:
    ledger = PendingLedger()
    ledger.observe(1, {(0, 1): (600, True)}, (0, 0))
    ledger.observe(2, {}, (0, 0))
    assert ledger.pending == [0, 0] and ledger.events == []
    ledger.observe(2, {(1, 1): (10, True)}, (0, 0))
    assert ledger.pending == [10, 0]


@pytest.mark.parametrize('revised,expected', [(100, [100, 0]), (150, [50, 0]), (300, [0, 100])])
def test_score_revision_rebuilds_prior_cancellation(revised: int, expected: list[int]) -> None:
    ledger = PendingLedger()
    ledger.observe(1, {(0, 1): (300, False)}, (0, 0))
    ledger.observe(1, {(0, 1): (300, False), (1, 2): (200, True)}, (0, 0))
    ledger.observe(1, {(0, 1): (revised, True), (1, 2): (200, True)}, (0, 0))
    assert ledger.pending == expected


def test_revision_does_not_restore_already_landed_amount() -> None:
    ledger = PendingLedger()
    ledger.observe(1, {(0, 1): (300, False)}, (0, 0))
    ledger.observe(1, {(0, 1): (300, False)}, (0, 30))
    ledger.observe(1, {(0, 1): (100, True)}, (0, 30))
    assert ledger.pending == [0, 70]


def test_verified_tracks_pending_origins_only() -> None:
    ledger = PendingLedger()
    ledger.observe(1, {(0, 1): (30, False)}, (0, 0))
    assert not ledger.verified(1)
    ledger.observe(1, {(0, 1): (30, False)}, (0, 30))
    ledger.observe(1, {(0, 1): (30, False), (0, 2): (30, True)}, (0, 30))
    assert ledger.verified(1)


def test_increment_after_partial_drop_credits_only_delta() -> None:
    ledger = PendingLedger()
    ledger.observe(1, {(0, 1): (60, True)}, (0, 0))
    ledger.observe(1, {(0, 1): (60, True)}, (0, 30))
    ledger.observe(1, {(0, 1): (90, True)}, (0, 30))
    assert ledger.pending == [0, 60]
