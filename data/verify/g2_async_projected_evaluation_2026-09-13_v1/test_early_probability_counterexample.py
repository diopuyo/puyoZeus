"""元captureは採録対象外でも空PBを拒否する。selector導入前の限定CPU反例。"""
from contextlib import ExitStack
from typing import Any
import pytest
import second_observation as O
from test_second_observation import context, generated, live, saved


def test_original_capture_without_selector_rejects_outside_expected(context: Any) -> None:
    value = context
    observer = value.state['hidden_probability_observer']
    observer.expected = [(value.row['frame_idx'] + 2, '2P')]
    observer.rows = []
    assert (value.row['frame_idx'], '2P') not in observer.expected
    with ExitStack() as stack:
        evidence = O.install(stack, value.journal, value.state)
        with pytest.raises(ValueError, match='second_PB_lifetime'):
            generated(value.journal, value.pipe, value.result, value.row)
        assert value.journal.count == 1 and evidence.error is not None
    assert evidence.closed
