"""基準自身の1連鎖→次手、原NEXT未消費とjoint保持の人工対照。"""
from __future__ import annotations
import io
from pathlib import Path
import sys
from typing import Any
import pytest

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / 'g2_settled_basis_connection_review_2026-09-11_v1'))
import test_physical_tracking as F
import mode as C

B = F.B


def setup() -> tuple[Any, Any, Any, Any]:
    previous, current, origin, final = F.setup()
    connection = previous.connection
    pb = B.ProbabilisticBoard.from_board(origin)
    pb.set_distribution(0, 5, {4: 0.3, 5: 0.7})
    value = B.establish(F.F.SCOPE, 10, 30, origin, pb)
    connection.registry = F.R.Registry(connection.recovery.factory)
    connection.binding = connection.registry.bind(connection.recovery.factory, value, 'basis')
    mode = C.Mode(connection, {}, io.StringIO())
    mode.native = F.P.M.N.Recorder(connection)
    return mode, current, origin, final


def test_basis_chain_then_next_hand_keeps_distribution() -> None:
    mode, current, origin, final = setup()
    first = F.step(mode, current, 12, 'step1', final, origin=origin)
    assert first['reason'] == 'basis_cascade_applied' and not mode.native.seen_occurrences
    assert first['transition']['distribution_report']['chain_probabilities'] == ((1, 1.0),)
    assert not first['transition']['next_consumed'] and mode.basis_cascade_closed
    next_board = final.copy()
    next_board.set(10, 0, 2)
    next_board.set(11, 0, 3)
    second = F.step(mode, current, 14, 'step2', next_board, number=1, pair=(2, 3))
    assert second['physical_transition_applied'] and len(mode.native.seen_occurrences) == 1
    value = mode.connection.registry.current(mode.connection.binding)
    assert len(value.tokens) == 2 and value.tokens[0].startswith('basis-cascade:')
    assert B.marginals(value).cell(0, 5).probs == pytest.approx({4: 0.3, 5: 0.7})
    assert not value.quality_gate_clear


@pytest.mark.parametrize('case', ('nonstable', 'zero_support', 'active_origin'))
def test_basis_chain_holds_without_next_consumption(case: str) -> None:
    mode, current, origin, final = setup()
    before = mode.connection.registry.current(mode.connection.binding)
    board = final.copy()
    if case == 'zero_support': board.set(12, 1, 5)
    if case == 'active_origin': mode.connection.recovery.provider.no_origin = lambda *_: False
    row = F.step(mode, current, 12, 'step1', board, origin=origin,
                 state='chain' if case == 'nonstable' else 'stable')
    assert not mode.basis_cascade_closed and not mode.native.pending
    assert mode.connection.registry.current(mode.connection.binding) is before
    mode.connection.recovery.provider.no_origin = lambda *_: True
    row = F.step(mode, current, 14, 'step2', final)
    assert row['reason'] == 'basis_cascade_applied' and len(mode.applied) == 1


def test_no_arbitrary_origin_adoption() -> None:
    mode, current, origin, final = setup()
    origin.set(12, 1, 5)
    with pytest.raises(ValueError, match='basis_origin_visible_mismatch'):
        F.step(mode, current, 12, 'step1', final, origin=origin)
    assert not mode.applied and mode.basis_origin is None


def test_new_hand_during_unresolved_basis_chain_is_rejected() -> None:
    mode, current, origin, final = setup()
    F.step(mode, current, 12, 'step1', final, origin=origin, state='chain')
    with pytest.raises(ValueError, match='native_hand_before_basis_cascade_closed'):
        F.step(mode, current, 14, 'step2', final, number=1, pair=(2, 3))
    assert not mode.applied and len(mode.native.pending) == 1


def test_normal_existing_hand_route(monkeypatch: Any) -> None:
    monkeypatch.setattr(F.P, 'Mode', C.Mode)
    F.test_native_origin_chain_and_next_hand_update_once()
