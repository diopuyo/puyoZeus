"""独立台帳の差替えは死亡入力だけに限定する。"""
from __future__ import annotations

from types import SimpleNamespace as NS
from copy import deepcopy
import pytest
from src.board import Board
from src.exchange_pending_ledger import PendingLedger
from src.exchange_death_inputs import projected_pending, hidden_proof, death_inputs
from src.exchange_event_landing import ExchangeLandingProjection


def ledger_setup() -> tuple:
    """先行攻撃と打ち返しが相殺された履歴を用意する。"""
    ledger = PendingLedger()
    ledger.observe(1, {(0, 1): (100, True)}, (0, 0))
    ledger.observe(1, {(0, 1): (100, True), (1, 2): (20, True)}, (0, 0))
    safety = NS(ledger=ledger, elapsed={(0, 1): 0., (1, 2): 0.})
    return safety, NS(chain_id=2)


def test_projected_total_does_not_mutate_ledger() -> None:
    safety, chain = ledger_setup()
    saved = deepcopy(vars(safety.ledger))
    assert projected_pending(safety, chain, 3500., 1) == 50
    assert vars(safety.ledger) == saved


def test_projected_total_rebuilds_lower_revision() -> None:
    safety, chain = ledger_setup()
    assert projected_pending(safety, chain, 700., 1) == 90


def test_projected_total_keeps_consumed_landings() -> None:
    safety, chain = ledger_setup()
    safety.ledger.observe(1, safety.ledger.totals, (0, 30))
    assert projected_pending(safety, chain, 3500., 1) == 20


def test_new_flag_preserves_probability_incoming() -> None:
    projection = ExchangeLandingProjection(death_pending_ledger=True)
    projection.safety.ledger.observe(1, {(0, 1): (100, True)}, (0, 0))
    tracker = NS(current=NS(chains=[NS(side='1P', provisional_score=700.,)]), _score_elapsed=0.)
    assert projection._incoming(tracker, (0, 0)) == [0, 10]
    assert projection.safety.ledger.pending == [0, 100]
    assert not projection.safety.ledger_enabled


def test_death_context_uses_ledger_but_keeps_original_input(monkeypatch: pytest.MonkeyPatch) -> None:
    projection = ExchangeLandingProjection(death_pending_ledger=True)
    projection.safety.ledger.observe(1, {(0, 1): (100, True)}, (0, 0))
    boards = (Board(), Board())
    monkeypatch.setattr(projection, '_receivers', lambda *a: (boards, [0, 0]))
    monkeypatch.setattr(projection, '_death_boards', lambda *a: (boards, boards, [True, True]))
    overlay = NS(tracker=NS(latest_chain=lambda side: None))
    incoming = [0, 700]
    context = death_inputs(projection, overlay, tuple(NS(board=b) for b in boards), incoming)
    assert context['incoming'] == [0, 100] and incoming == [0, 700]


@pytest.mark.parametrize('outcomes,dead,checked', [([True, True], True, 2), ([True, False], False, 2),
                                                  ([False, True], False, 1)])
def test_tied_boards_require_all_death(monkeypatch: pytest.MonkeyPatch,
                                     outcomes: list, dead: bool, checked: int) -> None:
    from src import exchange_event_multilanding as module
    iterator = iter(outcomes)
    monkeypatch.setattr(module, 'cached_proof', lambda *a: dict(dead=next(iterator)))
    calls = []
    projection = NS(hidden_death=NS(mark_used=lambda *a: calls.append(a)))
    bound = dict(score=100., options=[dict(board=Board()._grid.tolist())]*2)
    value = hidden_proof(projection, bound, (), 100, 1, 0., 2.)
    assert value['dead'] is dead and value['checked'] == checked and len(calls) == 1
