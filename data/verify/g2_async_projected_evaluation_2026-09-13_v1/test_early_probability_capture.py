"""原O/原Jを人工frameで駆動し、対象外と内部欠落・復元・例外優先を検証する。"""
from contextlib import ExitStack
import json
from typing import Any
import pytest
import second_observation as O
import early_probability_capture as E
from test_second_observation import context, generated, live, saved


def replace(stack: Any, owner: Any, name: str, value: Any) -> None:
    previous = getattr(owner, name)
    stack.callback(setattr, owner, name, previous)
    setattr(owner, name, value)


def prepare(value: Any) -> None:
    observer = value.state['hidden_probability_observer']
    observer.installed, observer.closed = True, False
    observer.expected = [(value.row['frame_idx'], '2P')]
    receiver = value.state['postcommit_current_receiver']
    receiver.journal, receiver.closed, receiver.errors = value.journal, False, []
    value.state['provisional_context_observer'] = receiver.rec


@pytest.mark.parametrize('outside', [False, True])
def test_original_capture_selector_snapshot_and_restoration(context: Any, outside: bool) -> None:
    value = context
    prepare(value)
    original, original_capture = value.journal.complete_step, O.capture
    if outside:
        value.state['hidden_probability_observer'].expected = []
        value.state['hidden_probability_observer'].rows = []
    with ExitStack() as stack:
        capture = E.Capture(stack, value.journal, value.state, O, replace)
        generated(value.journal, value.pipe, value.result, value.row)
        step = json.loads(value.journal.stream.getvalue().splitlines()[-1])
        result = capture.snapshot(step)
        assert E.candidate(result) is not outside
        assert not result['basis_registered'] and not result['quality_gate_clear']
        if outside: assert result['hold_reason'] == 'outside_original_PB_window'
        else:
            assert result['probability']['cells'] == value.state['hidden_probability_observer'].rows[-1]['probability']['cells']
            step['code_sha256'] = 'changed'
            with pytest.raises(ValueError, match='clock_or_code'): capture.snapshot(step)
        with pytest.raises(ValueError, match='capture_owner'):
            E.Capture(stack, value.journal, value.state, O, replace)
    assert value.journal.complete_step == original and O.capture is original_capture and capture.closed
    assert capture.journal is None and capture.evidence is None
    with ExitStack() as stack:
        late = O.install(stack, value.journal, value.state)
    assert late.closed and value.journal.complete_step == original


@pytest.mark.parametrize('fault', ['stale', 'empty', 'side', 'expected'])
def test_inside_missing_or_changed_contract_stays_strict(context: Any, fault: str) -> None:
    value = context
    prepare(value)
    observer = value.state['hidden_probability_observer']
    with ExitStack() as stack:
        capture = E.Capture(stack, value.journal, value.state, O, replace)
        if fault == 'stale': observer.rows[-1]['frame_idx'] -= 2
        elif fault == 'empty': observer.rows.clear()
        elif fault == 'side': observer.rows[-1]['side'] = '1P'
        else: observer.expected.append((value.row['frame_idx'] + 2, '2P'))
        with pytest.raises(ValueError): generated(value.journal, value.pipe, value.result, value.row)
        assert value.journal.count == 1 and capture.evidence.error is not None


def test_original_failure_overrides_capture_failure_after_cleanup(context: Any, monkeypatch: Any) -> None:
    value = context
    prepare(value)
    primary = RuntimeError('original_complete_failure')
    original = value.journal.complete_step
    def failed(*args: Any) -> None:
        original(*args)
        raise primary
    monkeypatch.setattr(value.journal, 'complete_step', failed)
    value.state['hidden_probability_observer'].rows.clear()
    with ExitStack() as stack:
        capture = E.Capture(stack, value.journal, value.state, O, replace)
        with pytest.raises(RuntimeError) as caught:
            generated(value.journal, value.pipe, value.result, value.row)
        assert caught.value is primary and capture.evidence.error is primary and value.journal.count == 1
