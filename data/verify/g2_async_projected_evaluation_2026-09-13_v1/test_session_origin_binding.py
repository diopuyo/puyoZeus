"""元update hookと元J/Witnessを通し、人工Sessionの接続/障害/解除を検査する。"""
from contextlib import ExitStack
import json
from types import SimpleNamespace as N
from typing import Any
import pytest
import check_saved_inputs
import live_session as LIVE
import journal_witness as W
from src import chain_prediction_ledger_v1 as LEDGER
from src.board import Board
from src.chain import ChainSimulator
from test_journal_pair_reader import setup
from test_journal_origin_capture import emit_pair, raw_event
from journal_origin_capture import OriginCapture
from session_origin_binding import Binding, install_class, STREAM_NAME, CLOSE_NAME


def scenario(path: Any, fault: str | None) -> tuple:
    journal, old_pipe, _ = setup()
    journal.expected = []
    error, calls = RuntimeError('original_identity'), []
    raw = raw_event(journal)
    if fault == 'capture': raw['before_board']['sha256'] = 'bad'
    class Pipe:
        def update(self, frame: int, clock: float) -> Any:
            calls.append('update')
            if fault == 'original': raise error
            if self is pipe: emit_pair(journal, self, frame, raw)
            return self
    pipe = Pipe()
    pipe.__dict__.update(vars(old_pipe))
    journal.pipe = journal.tracker._pipeline = pipe
    journal.controller.instances = {id(pipe): journal.controller.instances[id(old_pipe)]}
    class Session:
        def __init__(self, stack: Any) -> None:
            self.state, self.pipe, self.journal = dict(output=path), pipe, journal
            self.error, self.restored = None, False
            stack.callback(lambda: calls.append('base_closed'))
            self.witness = W.install(stack, journal)
        def completed(self, frame: int) -> str:
            calls.append('parent')
            if fault == 'parent': raise error
            return 'parent_result'
    def factory(session: Any, stream: Any) -> OriginCapture:
        return OriginCapture(session.witness, journal, pipe, LEDGER, ChainSimulator(),
                             Board.from_dict, lambda raw: False, stream)
    return Session, pipe, factory, calls, error


@pytest.mark.parametrize('fault', [None, 'original', 'parent', 'capture'])
def test_original_hook_J_consumer_parent_and_cleanup(tmp_path: Any, fault: str | None) -> None:
    cls, pipe, factory, calls, error = scenario(tmp_path, fault)
    original_init, original_completed, original_update = cls.__init__, cls.completed, type(pipe).update
    caught = None
    try:
        with ExitStack() as stack:
            install_class(stack, cls, factory)
            with pytest.raises(ValueError, match='duplicate'): install_class(stack, cls, factory)
            value = cls(stack)
            LIVE.attach(stack, value)
            assert type(value) is cls
            assert pipe.update(100, 100 / 60) is pipe
    except BaseException as failure:
        caught = failure
    assert cls.__init__ is original_init and cls.completed is original_completed
    assert type(pipe).update is original_update and value.restored and value.witness.closed
    receipt = json.loads((tmp_path / CLOSE_NAME).read_text())
    assert receipt['capture_closed'] and receipt['stream_closed']
    assert not receipt['quality_gate_clear'] and calls[-1] == 'base_closed'
    packets = [json.loads(line) for line in (tmp_path / STREAM_NAME).read_text().splitlines()]
    if fault is None: assert caught is None and calls == ['update', 'parent', 'base_closed']
    elif fault == 'capture': assert caught is value.error and 'source_board_hash' in str(caught)
    else: assert caught is error
    assert len(packets) == (0 if fault == 'original' else 1)
    assert ('parent' in calls) == (fault in (None, 'parent'))
    assert value.projected_origin_binding.capture is None


@pytest.mark.parametrize('body_present', [False, True])
def test_cleanup_failure_preserves_body_or_fails_normal_exit(tmp_path: Any, body_present: bool) -> None:
    original, cleanup = RuntimeError('body'), OSError('close')
    class BrokenCapture:
        last_frame, closed = -1, False
        def close(self) -> None:
            raise cleanup
    session = N(state=dict(output=tmp_path))
    caught = None
    try:
        with ExitStack() as stack:
            binding = Binding(stack, session, lambda owner, stream: BrokenCapture())
            if body_present: raise original
    except BaseException as error:
        caught = error
    assert caught is (original if body_present else cleanup)
    receipt = json.loads((tmp_path / CLOSE_NAME).read_text())
    assert receipt['stream_closed'] and not receipt['capture_closed']
    assert receipt['cleanup_errors'] == [repr(cleanup)]
    assert binding.capture is None and binding.stream is None


def test_partial_constructor_failure_closes_opened_stream(tmp_path: Any) -> None:
    original = RuntimeError('factory_failed')
    def factory(session: Any, stream: Any) -> None:
        raise original
    with pytest.raises(RuntimeError) as caught:
        with ExitStack() as stack:
            Binding(stack, N(state=dict(output=tmp_path)), factory)
    assert caught.value is original
    assert json.loads((tmp_path / CLOSE_NAME).read_text())['stream_closed']
