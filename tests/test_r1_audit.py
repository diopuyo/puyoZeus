"""R1の後続観測監査と固定母数採点の故障・欠測を検証する。"""
from __future__ import annotations
from collections import deque
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from scripts.r1_correction_audit import CorrectionAudit
from scripts.measure_r1_recognition import score_event
from scripts.r1_measure_helpers import contact, first_reflection, lines
from scripts.collect_r1 import R1Observations
from scripts.e34c_observations import Observations
from tests.test_placement_signal_reconcile import observation


def correction(frame: int = 10) -> dict:
    return dict(side=0, frame=frame, t_sec=frame/60, signal='next',
                corrections=[dict(row=12, col=0, before=0, after=1)])


def test_new_fire_keeps_its_causal_snapshot(monkeypatch: pytest.MonkeyPatch) -> None:
    reader = R1Observations.__new__(R1Observations)
    required: set[tuple] = set()
    reader.required = required
    result = SimpleNamespace(p1=SimpleNamespace(chain_event=SimpleNamespace(trigger_sec=2.)),
                             p2=SimpleNamespace(chain_event=None))
    def observed(self: Observations, value: object, stamp: float, game: int) -> object:
        assert (game, 0, 2.) in self.required
        return value
    monkeypatch.setattr(Observations, 'observed', observed)
    assert reader.observed(result, 3., 1) is result
    assert required == {(1, 0, 2.)}


@pytest.mark.parametrize('value,status', [(1, 'verified'), (2, 'wrong')])
def test_followup_uses_later_agreement(tmp_path: Path, value: int, status: str) -> None:
    audit = CorrectionAudit(tmp_path/'audit.jsonl')
    audit.signal(correction())
    obs = observation(10)
    runtime = SimpleNamespace(history=[deque([obs]), deque([obs])])
    audit.observe(runtime, 10)
    assert len(audit.pending) == 1
    obs = observation(11)
    obs.cnn[12, 0] = obs.hsv[12, 0] = value
    runtime.history = [deque([obs]), deque([obs])]
    audit.observe(runtime, 11)
    audit.observe(runtime, 11)
    audit.close()
    assert [r['status'] for r in lines(audit.path)] == [status]
    assert len(list(lines(audit.path.with_suffix('.observations.jsonl.gz')))) == 2


@pytest.mark.parametrize('quality,erasing', [('smoke', False), ('', True)])
def test_bad_followup_is_not_counted_correct(tmp_path: Path, quality: str, erasing: bool) -> None:
    audit = CorrectionAudit(tmp_path/'audit.jsonl')
    audit.signal(correction())
    obs = replace(observation(11), quality=quality, erasing=erasing)
    audit.observe(SimpleNamespace(history=[deque([obs]), deque([obs])]), 11)
    audit.signal(dict(correction(12), corrections=[]))
    audit.close()
    assert [r['status'] for r in lines(audit.path)] == ['unverified']


def test_record_end_keeps_unverified_denominator(tmp_path: Path) -> None:
    audit = CorrectionAudit(tmp_path/'audit.jsonl')
    audit.signal(correction())
    audit.signal(dict(frame=11, corrections=[], reason='reader_error'))
    audit.close()
    assert list(lines(audit.path))[0]['end_reason'] == 'record_end'


def test_reset_does_not_score_against_next_game(tmp_path: Path) -> None:
    audit = CorrectionAudit(tmp_path/'audit.jsonl')
    audit.signal(correction())
    audit.end_pending('recognition_reset')
    obs = observation(11)
    obs.cnn[12, 0] = obs.hsv[12, 0] = 2
    audit.observe(SimpleNamespace(history=[deque([obs]), deque([obs])]), 11)
    audit.close()
    assert list(lines(audit.path))[0]['status'] == 'unverified'


def test_erasure_after_formula_cannot_be_false_correction(tmp_path: Path) -> None:
    audit = CorrectionAudit(tmp_path/'audit.jsonl')
    audit.signal(dict(correction(), signal='formula'))
    erased = replace(observation(10), erasing=True)
    audit.observe(SimpleNamespace(history=[deque([erased]), deque([erased])]), 10)
    empty = observation(11)
    empty.cnn[:] = empty.hsv[:] = 0
    audit.observe(SimpleNamespace(history=[deque([empty]), deque([empty])]), 11)
    audit.close()
    assert list(lines(audit.path))[0]['end_reason'] == 'erasure'


def row(grid: list | None, stamp: float = 1.) -> dict:
    return dict(t=stamp, game=1, sides=[dict(confirmed_board=grid, state=dict(state='STABLE'))])


def test_fixed_error_cells_and_missing_board_are_not_dropped() -> None:
    old = np.zeros((13, 6), dtype=int)
    reference = old.copy()
    reference[0] = -1
    reference[12, 0] = 1
    event = dict(source='test', t=1., side=0, i=0, early=True,
                 cells=[dict(row=12, col=0, reference=1)])
    measured = score_event(event, [row(old.tolist())], [row(None)], reference)
    assert measured['old_residual'] == measured['new_residual'] == 1
    assert measured['proxy_cells'] == 72 and measured['new_correct'] == 0
    assert measured['missing_board']


def test_contact_and_reflection_keep_same_placement() -> None:
    board = np.zeros((13, 6), dtype=int)
    board[12, 0] = 1
    event = dict(fps=60, side=0, cycle_start=.8, write_t=1.,
                 write_board=board.tolist(), write_additions=[[12, 0]], lower=0, upper=1, game=1)
    assert contact(event, {(59, 0): board}) == 59/60
    assert contact(event, {}) is None
    records = [row(np.zeros_like(board).tolist(), .9), row(board.tolist())]
    assert first_reflection(event, records) == 1.
    records[1]['game'] = 2
    assert first_reflection(event, records) is None
