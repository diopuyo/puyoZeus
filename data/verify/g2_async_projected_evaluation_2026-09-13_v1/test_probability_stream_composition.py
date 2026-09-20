"""原Oと原Streamの同時装着・解放を原J人工frameで検証する。実動画ではない。"""
from contextlib import ExitStack
import json
from typing import Any
import stream_witness as W
import writer_contract_v2 as K
import reproduce_v2 as TWO
import second_observation as O
import early_probability_capture as E
from test_early_probability_capture import prepare, replace
from test_second_observation import context, generated, live, saved


def test_original_stream_and_observation_close_then_reinstall(context: Any, monkeypatch: Any) -> None:
    value = context
    prepare(value)
    monkeypatch.setattr(W, 'K', K)
    monkeypatch.setattr(TWO.R.T, 'recorder', lambda: value.journal)
    attached, provider, controller, adoption = TWO.attached()
    assert attached is value.journal
    finished = []
    controller.finish = lambda frame, error: finished.append((frame is not None, error))
    original_emit, original_stream = value.journal.emit, value.journal.stream
    original_complete = value.journal.complete_step
    with ExitStack() as stack:
        provider.attach(stack, controller)
        wrapped_emit = value.journal.emit
        wrapped_complete = value.journal.complete_step
        with ExitStack() as early:
            witness = W.install(early, value.journal)
            capture = E.Capture(early, value.journal, value.state, O, replace)
            generated(value.journal, value.pipe, value.result, value.row)
            step = json.loads(original_stream.getvalue().splitlines()[-1])
            proof = capture.snapshot(step)
            assert E.candidate(proof) and witness.error is None and witness.last_index == 0
            assert set(witness.rows) == {'2P'} and finished == [(True, None)]
            assert value.journal.emit is wrapped_emit
        assert capture.closed and witness.closed and value.journal.complete_step is wrapped_complete
        assert value.journal.stream is original_stream and value.journal.emit is wrapped_emit
        with ExitStack() as late:
            later = W.install(late, value.journal)
            observation = O.install(late, value.journal, value.state)
        assert later.closed and observation.closed
    assert adoption.closed and value.journal.emit == original_emit
    assert value.journal.complete_step == original_complete
