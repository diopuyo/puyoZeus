"""Fableの既知色消失/entry支持消失を原候補経路で再現する。"""
from __future__ import annotations
from dataclasses import replace
from typing import Any
import pytest
import test_hidden_basis as F
import hidden_basis_gate_v1 as OLD


def damaged(case: str) -> tuple[Any, list[Any]]:
    clock, rows = F.items()
    row = rows[-1]
    col, color = (4, 3) if case == 'known_color' else (0, 0)
    sm = [list(values) for values in row.sm_confirmed_grid]
    sm[0][col] = color
    hidden = list(row.hidden_probability)
    hidden[col] = ((color, 1.0),)
    rows[-1] = replace(row, sm_confirmed_grid=tuple(map(tuple, sm)),
                      returned_grid=tuple(map(tuple, sm)), hidden_probability=tuple(hidden))
    return clock, rows


@pytest.mark.parametrize('case', ['known_color', 'entry_loss'])
def test_old_broad_initialization_reproduced(case: str) -> None:
    clock, rows = damaged(case)
    gate = F.feed(rows, clock, OLD.SettledBasisGate)
    assert gate.candidate is not None
    assert (4 if case == 'known_color' else 0) in gate.candidate.newly_unobserved_columns


@pytest.mark.parametrize('case', ['known_color', 'entry_loss'])
def test_repaired_rejection(case: str) -> None:
    clock, rows = damaged(case)
    gate = F.feed(rows, clock)
    assert gate.candidate is None
    assert gate.rows[-1]['reason'] == ('known_hidden_color_not_reset' if case == 'known_color'
                                      else 'hidden_distribution_lost')
