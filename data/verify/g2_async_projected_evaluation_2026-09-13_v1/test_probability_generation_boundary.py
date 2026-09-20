"""capture後に原Jが別resetを記録する人工境界。原O/Jを実呼出する。"""
from contextlib import ExitStack
from dataclasses import replace as changed
import json
from typing import Any
import pytest
import second_observation as O
import early_probability_capture as E
from test_early_probability_capture import prepare, replace
from test_second_observation import context, generated, live, saved


def test_before_scope_does_not_become_after_generation(context: Any) -> None:
    value = context
    prepare(value)
    original = value.journal.tracker.generation('2P')
    value.journal.tracker.generation = lambda side: changed(original, reset_epoch=original.reset_epoch + 1)
    with ExitStack() as stack:
        capture = E.Capture(stack, value.journal, value.state, O, replace)
        generated(value.journal, value.pipe, value.result, value.row)
        step = json.loads(value.journal.stream.getvalue().splitlines()[-1])
        assert step['generation']['reset_epoch'] + 1 == step['generation_after']['reset_epoch']
        assert capture.evidence.latest['scope'][5] == step['generation']['reset_epoch']
        assert capture.evidence.error is None and value.journal.count == 1
        packet = capture.snapshot(step)
        assert packet['in_step_generation_changed'] and not E.candidate(packet)
        assert packet['generation'] == step['generation']
        assert packet['generation_after'] == step['generation_after']
