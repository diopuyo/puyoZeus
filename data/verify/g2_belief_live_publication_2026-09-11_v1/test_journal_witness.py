"""原complete_step/emit本体を人工最小状態で駆動し、偽callerと書込失敗を拒否する。"""
from __future__ import annotations

from contextlib import ExitStack
from dataclasses import dataclass
import importlib.util
import io
import json
import sys
from types import SimpleNamespace as N
from typing import Any
import pytest
import journal_witness as W


@dataclass
class Generation:
    side: str
    reset_epoch: int = 0
    action_revision: int = 0
    identity_scope: str = 'software_observation_only_not_physical_identity'


def recorder() -> Any:
    spec = importlib.util.spec_from_file_location('_original_J_witness_test', W.SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    value = module.Recorder.__new__(module.Recorder)
    value.stream, value.count, value.errors = io.StringIO(), 0, []
    value.tracker = N(generation=lambda side: Generation(side))
    return value


def complete(journal: Any, side: str, ordinal: int, error: Any = None) -> None:
    scope = dict(frame_idx=100, time_sec=100/60, side=side)
    item = dict(scope=scope, token='step:'+str(ordinal), events=[], epoch=0, return_line=None, frame=None)
    journal.complete_step(item, None, error, sys.getprofile())
    assert item['frame'] is None


def test_original_emission_and_restore() -> None:
    journal = recorder()
    original = journal.emit
    with ExitStack() as stack:
        witness = W.install(stack, journal)
        complete(journal, '1P', 0)
        complete(journal, '2P', 1)
        actual = [json.loads(line) for line in journal.stream.getvalue().splitlines()]
        captured = witness.pair(100)
        assert all(all(actual[i][k] == v for k, v in row.items()) for i, row in enumerate(captured))
        captured[0]['exception'] = 'tampered_copy'
        assert witness.pair(100)[0]['exception'] is None
    assert witness.closed and journal.emit == original and 'emit' not in vars(journal)


def test_direct_fake_step_rejected() -> None:
    journal = recorder()
    with ExitStack() as stack:
        witness = W.install(stack, journal)
        with pytest.raises(ValueError, match='J_original_caller'):
            journal.emit(dict(kind='step', side='1P', token='step:0'))
        assert not witness.rows and journal.count == 0


def test_exception_is_not_erased() -> None:
    journal = recorder()
    with ExitStack() as stack:
        witness = W.install(stack, journal)
        complete(journal, '1P', 0, ValueError('original_failure'))
        complete(journal, '2P', 1)
        assert witness.pair(100)[0]['exception'] == "ValueError('original_failure')"
        with pytest.raises(ValueError, match='J_witness_frame'):
            witness.pair(102)


def test_writer_error_identity_preserved() -> None:
    journal = recorder()
    error = OSError('disk_failure')
    def write(value: str) -> None:
        raise error
    journal.stream = N(write=write)
    with ExitStack() as stack:
        witness = W.install(stack, journal)
        with pytest.raises(OSError) as caught:
            complete(journal, '1P', 0)
        assert caught.value is error and witness.error is error and not witness.rows


def test_original_writer_continues_after_witness_fault() -> None:
    journal = recorder()
    with ExitStack() as stack:
        witness = W.install(stack, journal)
        with pytest.raises(ValueError):
            journal.emit(dict(kind='step', side='1P', token='step:0'))
        error = witness.error
        complete(journal, '1P', 0)
        assert journal.count == 1 and witness.error is error and not witness.rows
        with pytest.raises(ValueError):
            witness.pair(100)


def test_preinstalled_foreign_emit_rejected() -> None:
    journal = recorder()
    journal.emit = lambda value: None
    with ExitStack() as stack, pytest.raises(ValueError, match='J_original_emit'):
        W.install(stack, journal)
