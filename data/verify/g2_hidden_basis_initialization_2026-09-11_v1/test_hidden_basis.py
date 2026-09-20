"""保存観測＋人工可視修復で隠し契約だけを分離。原動画/実factory合格ではない。"""
from __future__ import annotations
from dataclasses import replace
import io
import json
from pathlib import Path
import sys
from typing import Any
import pytest

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / 'g2_settled_basis_connection_review_2026-09-11_v1'))
import test_basis_registry as F
sys.path.insert(0, str(ROOT))
import hidden_basis_gate as H
import basis_registry_v2 as C


def items() -> tuple[Any, list[Any]]:
    clock, rows = F.OLD.observations()
    index = next(i for i, row in enumerate(rows) if row.frame == 35172)
    row = F.OLD.artificial_normal_stable(rows[index])
    grid = [list(values) for values in row.sm_confirmed_grid]
    grid[0][4] = 0
    grid = tuple(map(tuple, grid))
    hidden = list(row.hidden_probability)
    hidden[4] = ((0, 1.0),)
    rows[index] = replace(row, sm_confirmed_grid=grid, returned_grid=grid,
                          hidden_probability=tuple(hidden))
    return clock, rows[:index + 1]


def feed(rows: list[Any], clock: Any, gate_type: Any = H.SettledBasisGate) -> Any:
    gate = gate_type(rows[0].scope, clock['reset_frame'], clock['deadline'])
    for row in rows:
        gate.observe(row)
    return gate


def test_old_rejection_and_new_initialization() -> None:
    clock, rows = items()
    original = rows[-1]
    old = feed(rows, clock, H.OLD.SettledBasisGate)
    assert old.candidate is None and old.rows[-1]['reason'] == 'raw_SM_mismatch'
    value = feed(rows, clock).candidate
    assert value is not None and value.newly_unobserved_columns == (4,)
    assert value.actual_hidden_probability[4] == ((0, 1.0),)
    assert value.hidden_probability[4] == H.UNOBSERVED_PRIOR
    for col in (0, 1, 5):
        assert value.hidden_probability[col] == original.hidden_probability[col]
    assert rows[-1] is original and original.sm_confirmed_grid[0][4] == 0
    assert not value.hidden_prior_calibrated and not value.quality_gate_clear


@pytest.mark.parametrize('fault', ['visible', 'known_hidden', 'native_probability', 'effect', 'origin', 'epoch'])
def test_no_other_waiver(fault: str) -> None:
    clock, rows = items()
    row = rows[-1]
    grid = [list(values) for values in row.sm_confirmed_grid]
    if fault == 'visible': grid[1][4] = 0
    if fault == 'known_hidden': grid[0][3] = 5
    if fault in ('visible', 'known_hidden'):
        grid = tuple(map(tuple, grid))
        row = replace(row, sm_confirmed_grid=grid, returned_grid=grid)
    if fault == 'native_probability':
        hidden = list(row.hidden_probability)
        hidden[4] = ((1, 1.0),)
        row = replace(row, hidden_probability=tuple(hidden))
    if fault == 'effect': row = replace(row, effect_gate_window_active=True)
    if fault == 'origin': row = replace(row, origin_present=True)
    if fault == 'epoch': row = replace(row, epoch=row.epoch + 1)
    rows[-1] = row
    assert feed(rows, clock).candidate is None


def test_saved_prior_provenance_and_joint_roundtrip(tmp_path: Any) -> None:
    recovery, observer, item = F.fixture(tmp_path)
    clock, rows = items()
    observer.gate = feed(rows, clock)
    stream = io.StringIO()
    connection = C.Connection(recovery, observer, 36298, stream)
    connection.publish(item)
    connection.publish(item)
    packet = json.loads(stream.getvalue())
    value = connection.registry.current(connection.binding)
    assert connection.rows == 1 and len(value.worlds) == 2401
    assert C.S.decode(packet['state']) == value
    assert packet['fresh_prior_not_temporal_posterior']
    assert packet['initial_candidate']['actual_hidden_probability'][4] == [[0, 1.0]]
    assert packet['initial_candidate']['newly_unobserved_columns'] == [4]
    assert not packet['initial_candidate']['hidden_prior_calibrated']
    assert not packet['integer_current_published'] and not packet['quality_gate_clear']
