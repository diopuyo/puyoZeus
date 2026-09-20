"""元J/台帳/同stream/Binding終了を再用するcallback。数値予測陽性は別検収。"""
from contextlib import ExitStack
import gc
import json
from typing import Any
import weakref
import pytest
from test_journal_origin_capture import setup, emit_pair, raw_event
from test_session_origin_binding import scenario, install_class, LIVE, STREAM_NAME, CLOSE_NAME


@pytest.mark.parametrize('fault', [False, True])
def test_original_update_callback_save_and_binding_exit(tmp_path: Any, fault: bool) -> None:
    cls, pipe, original_factory, calls, error = scenario(tmp_path, None)
    captured = []
    def factory(session: Any, stream: Any) -> Any:
        capture = original_factory(session, stream)
        def projected(frame: int) -> dict:
            calls.append('projected')
            assert capture.handles['2P'].instance_id > 0  # 原consume後だけに呼ぶ。
            if fault: raise error
            return dict(status='HOLD', reason='artificial_callback', cutoff_frame=frame,
                        accounting_permission=False, quality_gate_clear=False)
        capture.projected_input = projected
        captured.append(capture)
        return capture
    caught = None
    try:
        with ExitStack() as stack:
            install_class(stack, cls, factory)
            value = cls(stack)
            LIVE.attach(stack, value)
            pipe.update(100, 100 / 60)
    except BaseException as failure:
        caught = failure
    capture = captured[0]
    rows = [json.loads(line) for line in (tmp_path / STREAM_NAME).read_text().splitlines()]
    receipt = json.loads((tmp_path / CLOSE_NAME).read_text())
    assert receipt['capture_closed'] and receipt['stream_closed'] and capture.projected_input is None
    assert calls.count('projected') == 1 and len(rows) == 1
    if fault:
        assert caught is error and receipt['original_error'] == repr(error)
        assert rows[0]['kind'] == 'origin_capture_failure' and rows[0]['source_rows_json']
        assert rows[0]['partial_instances']['2P'] > 0 and 'parent' not in calls
    else:
        assert caught is None and rows[0]['projected_input']['status'] == 'HOLD'
        assert calls.index('projected') < calls.index('parent')
    with pytest.raises(ValueError, match='lifetime'):
        capture.completed(100)


def test_callback_reference_is_released_on_close(tmp_path: Any) -> None:
    cls, pipe, factory, _, _ = scenario(tmp_path, None)
    class Reader:
        def read(self, frame: int) -> dict:
            return {'status': 'HOLD', 'cutoff_frame': frame}
    with ExitStack() as stack:
        value = cls(stack)
        stream = stack.enter_context((tmp_path / 'reference.jsonl').open('x'))
        capture = factory(value, stream)
        reader = Reader()
        reference = weakref.ref(reader)
        capture.projected_input = reader.read
        del reader
        assert reference() is not None
        capture.close()
        gc.collect()
        assert reference() is None
