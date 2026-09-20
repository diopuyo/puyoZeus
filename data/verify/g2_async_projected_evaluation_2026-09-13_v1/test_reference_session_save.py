"""元update hook→consumer→参考保存→親completedと、途中失敗時の非到達/保存/解放。"""
from contextlib import ExitStack
import json
from typing import Any
import pytest
from test_session_origin_binding import scenario
import live_session as LIVE
import fixed_origin_reference as F
from session_origin_binding import install_class, STREAM_NAME, CLOSE_NAME
from test_fixed_origin_reference import source


@pytest.mark.parametrize('fault', [None, 'reader', 'writer'])
def test_reference_saved_before_parent_and_failure_blocks_parent(tmp_path: Any, monkeypatch: Any, fault: str | None) -> None:
    cls, pipe, factory, calls, _ = scenario(tmp_path, None)
    original_init, original_completed = cls.__init__, cls.completed
    error, caught = RuntimeError('reference_failure_identity'), None
    if fault == 'reader':
        def failed(*args: Any) -> None: raise error
        monkeypatch.setattr(F, 'read', failed)
    class Writer:
        def __init__(self, stream: Any) -> None: self.stream = stream
        def write(self, text: str) -> Any:
            if 'fixed_origin_reference_pair/v1' in text: raise error
            return self.stream.write(text)
        def flush(self) -> None: self.stream.flush()
    try:
        with ExitStack() as stack:
            install_class(stack, cls, factory, after_completed=F.save)
            value = cls(stack)
            capture = value.projected_origin_binding.capture
            if fault == 'writer': capture.stream = Writer(capture.stream)
            LIVE.attach(stack, value)
            pipe.update(100, 100 / 60)
    except BaseException as failure:
        caught = failure
    assert cls.__init__ is original_init and cls.completed is original_completed
    assert value.restored and capture.closed and value.projected_origin_binding.capture is None
    assert json.loads((tmp_path / CLOSE_NAME).read_text())['stream_closed']
    packets = [json.loads(line) for line in (tmp_path / STREAM_NAME).read_text().splitlines()]
    assert len(packets) == 2
    if fault is None:
        assert caught is None and 'parent' in calls
        assert packets[1]['kind'] == 'fixed_origin_reference_pair/v1'
        assert all(row['status'] == 'HOLD' for row in packets[1]['sides'])
    else:
        assert caught is error and value.error is error and 'parent' not in calls
        assert packets[1]['kind'] == 'origin_capture_failure'
        assert packets[1]['error'] == repr(error) and len(json.loads(packets[1]['source_rows_json'])) == 2


def test_reference_positive_pair_save_uses_original_stream_consumer(source: Any) -> None:
    capture, journal, _, _, _ = source
    before = capture.witness.original.getvalue()
    result = F.save(capture, 100)
    assert [row['status'] for row in result['sides']] == ['HOLD', 'REFERENCE_ONLY']
    assert capture.witness.original.getvalue() == before
    packet = json.loads(capture.stream.getvalue().splitlines()[-1])
    assert packet == json.loads(json.dumps(result)) and not packet['quality_gate_clear']
