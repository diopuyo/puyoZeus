"""元complete_step/Witnessを使い現clock原票の取得・保存・解除を検査する。"""
from contextlib import ExitStack
from dataclasses import asdict
import json
from pathlib import Path
import sys
from types import SimpleNamespace as N
from typing import Any, Callable
import pytest
from test_journal_witness import recorder, Generation
import journal_witness as W
from journal_pair_reader import read_pair

FRAME = 100


def setup() -> tuple:
    journal = recorder()
    pipe = N(_sm_1p=object(), _sm_2p=object())
    values = {side: Generation(side) for side in ('1P', '2P')}
    journal.tracker = N(_pipeline=pipe, _machines={'1P': pipe._sm_1p, '2P': pipe._sm_2p},
                        generation=lambda side: values[side])
    journal.pipe, journal.closed, journal.active, journal.steps = pipe, False, None, 0
    journal.source_id, journal.run_id = 'artificial-source', 'artificial-run'
    journal.expected = [(FRAME, '1P'), (FRAME, '2P')]
    journal.selected = set(journal.expected)
    journal.history = N(frame=FRAME, time_sec=FRAME / 60)
    journal.controller = N(instances={id(pipe): N(histories={side: N(epoch=0) for side in values})})
    return journal, pipe, values


def complete(journal: Any, pipe: Any, side: str, ordinal: int, *,
             error: Exception | None = None, change: Callable[[], None] | None = None,
             events: list | None = None, frame: int = FRAME) -> None:
    scope = journal.scope(pipe, side, frame, frame / 60)
    item = dict(scope=scope, token=f'step:{ordinal}', events=events or [], epoch=journal.epoch(pipe, side),
                return_line=None, frame=None)
    if change is not None:
        change()
    journal.steps += 1
    journal.complete_step(item, None, error, sys.getprofile())


def test_pair_save_restore_and_nonstable_not_required(tmp_path: Path) -> None:
    journal, pipe, _ = setup()
    original, profile = journal.emit, sys.getprofile()
    with ExitStack() as stack:
        witness = W.install(stack, journal)
        complete(journal, pipe, '1P', 0)
        assert read_pair(witness, journal, pipe, FRAME).holds == ('both_side_J_not_available',)
        complete(journal, pipe, '2P', 1)
        before = journal.stream.getvalue()
        result = read_pair(witness, journal, pipe, FRAME)
        assert not result.holds and not result.accounting_permission
        path = tmp_path / 'pair.json'
        path.write_text(json.dumps(asdict(result)))
        assert json.loads(path.read_text())['rows_json'] == result.rows_json
        assert journal.stream.getvalue() == before and sys.getprofile() is profile
    assert journal.emit == original and witness.closed
    with pytest.raises(ValueError, match='projected_J_lifetime'):
        read_pair(witness, journal, pipe, FRAME)


@pytest.mark.parametrize('fault', ['clock', 'pipe', 'generation', 'ordinal', 'exception'])
def test_wrong_current_input_is_rejected(fault: str) -> None:
    journal, pipe, values = setup()
    with ExitStack() as stack:
        witness = W.install(stack, journal)
        complete(journal, pipe, '1P', 6 if fault == 'ordinal' else 0,
                 error=ValueError('original_failure') if fault == 'exception' else None)
        complete(journal, pipe, '2P', 1)
        if fault == 'clock': journal.history.frame += 2
        elif fault == 'pipe': journal.tracker._pipeline = object()
        elif fault == 'generation': values['1P'].action_revision += 1
        before = journal.stream.getvalue()
        with pytest.raises(ValueError):
            read_pair(witness, journal, pipe, FRAME)
        assert journal.stream.getvalue() == before


def test_real_step_generation_change_is_retained_as_hold() -> None:
    journal, pipe, values = setup()
    with ExitStack() as stack:
        witness = W.install(stack, journal)
        complete(journal, pipe, '1P', 0, change=lambda: setattr(values['1P'], 'action_revision', 1))
        complete(journal, pipe, '2P', 1)
        result = read_pair(witness, journal, pipe, FRAME)
        assert result.holds == ('1P:generation_changed_within_step',)
        rows = json.loads(result.rows_json)
        assert rows[0]['generation']['action_revision'] == 0
        assert rows[0]['generation_after']['action_revision'] == 1


def test_partial_next_pair_holds_then_complete_pair_becomes_readable() -> None:
    journal, pipe, _ = setup()
    with ExitStack() as stack:
        witness = W.install(stack, journal)
        complete(journal, pipe, '1P', 0)
        complete(journal, pipe, '2P', 1)
        journal.history.frame, journal.history.time_sec = 102, 102 / 60
        journal.expected.extend([(102, '1P'), (102, '2P')])
        journal.selected = set(journal.expected)
        complete(journal, pipe, '1P', 2, frame=102)
        assert read_pair(witness, journal, pipe, 102).holds == ('both_side_J_not_current',)
        complete(journal, pipe, '2P', 3, frame=102)
        assert not read_pair(witness, journal, pipe, 102).holds
