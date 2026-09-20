"""新入力の保存・失敗原票・元例外・重複禁止を原captureで検証する。"""
import io
import json
import sys
from typing import Any
import pytest
import session_runtime_binding as B
import fixed_origin_reference as F
import fixed_root_probability as P
from test_fixed_origin_reference import source
from test_fixed_root_probability import history_for


def hook(capture: Any, history: Any, monkeypatch: Any) -> tuple:
    monkeypatch.setitem(sys.modules, '_async_live_fixed_origin_reference', F)
    def load(alias: str, path: Any, injection: dict) -> Any:
        assert alias == '_async_live_fixed_root_probability' and injection['fixed_origin_reference'] is F
        return P
    return B.root_hook(load, F.R, lambda: history, True, F.save)


def test_ready_saved_after_reference_once(source: Any, monkeypatch: Any) -> None:
    capture, _, _, _, _ = source
    history, _ = history_for(capture)
    before = capture.ledger.snapshot(capture.handles['2P'])
    callback, counts = hook(capture, history, monkeypatch)
    callback(capture, 100)
    saved = [json.loads(line) for line in capture.stream.getvalue().splitlines()]
    assert saved[-2]['kind'] == 'fixed_origin_reference_pair/v1'
    assert saved[-1]['kind'] == 'conditional_fixed_root_input/v1' and saved[-1]['status'] == 'READY'
    assert counts == dict(calls=1, READY=1, HOLD=0, last_frame=100)
    assert capture.ledger.snapshot(capture.handles['2P']) == before
    with pytest.raises(ValueError, match='duplicate_save'): callback(capture, 100)


@pytest.mark.parametrize('fault', ['history', 'writer'])
def test_failed_input_keeps_original_exception_and_J_rows(source: Any, monkeypatch: Any, fault: str) -> None:
    capture, _, _, _, _ = source
    history, _ = history_for(capture)
    primary = RuntimeError('root_writer_failure')
    class Sink(io.StringIO):
        calls = 0
        def write(self, value: str) -> int:
            self.calls += 1
            if fault == 'writer' and self.calls == 2: raise primary
            return super().write(value)
    capture.stream = Sink()
    if fault == 'history': history.closed = True
    callback, counts = hook(capture, history, monkeypatch)
    with pytest.raises((RuntimeError, ValueError)) as caught: callback(capture, 100)
    assert capture.error is caught.value and counts['calls'] == 0
    if fault == 'writer': assert caught.value is primary
    saved = [json.loads(line) for line in capture.stream.getvalue().splitlines()]
    assert saved[-1]['kind'] == 'origin_capture_failure'
    assert len(json.loads(saved[-1]['source_rows_json'])) == 2
    assert capture.save_error is None


def test_missing_dependency_is_strict_and_disabled_preserves_previous() -> None:
    previous = lambda capture, frame: None
    result, counts = B.root_hook(None, None, None, False, previous)
    assert result is previous and counts['calls'] == 0
    with pytest.raises(ValueError, match='dependencies'): B.root_hook(None, None, None, True, previous)
