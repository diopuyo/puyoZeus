"""atomic landing shadowの単一軸・左右対称・復元をCPUで検査する。"""

from __future__ import annotations

import ast
import contextlib
import io
import inspect
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from scripts import diagnose_video38_atomic_landing_shadow_v1 as shadow
from src.recognition_pipeline import _is_game_event_chain_exit


def _event(mechanism: str = "landing", trigger: float = 12.5) -> SimpleNamespace:
    return SimpleNamespace(mechanism=mechanism, trigger_sec=trigger)


def _pipe(side: str, enabled: bool = True) -> SimpleNamespace:
    pipe = SimpleNamespace(
        _enable_game_event_chain_exit=enabled, _chain_max_hold_sec=9.0,
        _active_chain_1p="keep-a", _active_chain_2p="keep-b",
        _chain_until_1p=21.0, _chain_until_2p=22.0,
        _chain_entry_t_1p=1.0, _chain_entry_t_2p=2.0,
        _chain_start_next_1p=(1, 2), _chain_start_next_2p=(2, 3),
        _last_seen_next_1p=(4, 4), _last_seen_next_2p=(5, 5),
        _chain_event_max_until_1p=31.0, _chain_event_max_until_2p=32.0,
        _last_chain_event_for_settle_1p="settle-a",
        _last_chain_event_for_settle_2p="settle-b")
    pipe.side = side
    return pipe


def _install(pipe: Any, rec: shadow.AtomicRecorder) -> tuple[Any, contextlib.ExitStack]:
    class Pipeline:
        def _start_chain_estimate(self, side: str, event: Any,
                                  precomputed_result: Any = None) -> str:
            return f"{side}:{event.mechanism}:{precomputed_result}"
    wrapped = Pipeline()
    wrapped.__dict__.update(vars(pipe))
    stack = contextlib.ExitStack()
    shadow.instrument_atomic_metadata(stack, Pipeline, rec)
    return wrapped, stack


def _recorder() -> shadow.AtomicRecorder:
    rec = shadow.AtomicRecorder(io.StringIO(), {})
    rec.begin_frame(shadow.base.TARGET_FRAME, 578.366667)
    return rec


@pytest.mark.parametrize(("side", "suffix"), [("1P", "1p"), ("2P", "2p")])
def test_landing_initializes_metadata_without_touching_active_or_until(
    side: str, suffix: str,
) -> None:
    pipe, rec = _pipe(side), _recorder()
    active = getattr(pipe, f"_active_chain_{suffix}")
    until = getattr(pipe, f"_chain_until_{suffix}")
    settle = getattr(pipe, f"_last_chain_event_for_settle_{suffix}")
    pipe, stack = _install(pipe, rec)
    with stack:
        assert pipe._start_chain_estimate(side, _event(), "verified").endswith("verified")
    assert getattr(pipe, f"_chain_entry_t_{suffix}") == 12.5
    assert getattr(pipe, f"_chain_start_next_{suffix}") == getattr(
        pipe, f"_last_seen_next_{suffix}")
    assert getattr(pipe, f"_chain_event_max_until_{suffix}") == 21.5
    assert getattr(pipe, f"_active_chain_{suffix}") == active
    assert getattr(pipe, f"_chain_until_{suffix}") == until
    assert getattr(pipe, f"_last_chain_event_for_settle_{suffix}") == settle
    assert rec.atomic_metadata_rows[0]["status"] == "applied"


def test_fresh_same_next_prevents_stale_next_exit() -> None:
    pipe, rec = _pipe("2P"), _recorder()
    pipe._last_seen_next_2p = (5, 5)
    pipe._chain_start_next_2p = (2, 2)
    pipe, stack = _install(pipe, rec)
    with stack:
        pipe._start_chain_estimate("2P", _event())
    assert _is_game_event_chain_exit((5, 5), pipe._chain_start_next_2p) is False


def test_non_landing_and_no_call_leave_metadata_unchanged() -> None:
    pipe, rec = _pipe("1P"), _recorder()
    before = shadow._metadata(pipe, "1P")
    pipe, stack = _install(pipe, rec)
    with stack:
        assert shadow._metadata(pipe, "1P") == before
        pipe._start_chain_estimate("1P", _event("formula"))
    assert shadow._metadata(pipe, "1P") == before
    assert rec.atomic_metadata_rows == []


def test_disabled_game_event_keeps_unused_max_until() -> None:
    pipe, rec = _pipe("2P", enabled=False), _recorder()
    pipe, stack = _install(pipe, rec)
    with stack:
        pipe._start_chain_estimate("2P", _event())
    assert pipe._chain_event_max_until_2p == 32.0
    assert pipe._chain_entry_t_2p == 12.5
    assert pipe._chain_start_next_2p == (5, 5)


def test_original_exception_does_not_initialize_and_wrapper_restores() -> None:
    class Pipeline:
        def _start_chain_estimate(self, side: str, event: Any,
                                  precomputed_result: Any = None) -> None:
            raise LookupError("fixture")
    original = Pipeline._start_chain_estimate
    fixture, rec = _pipe("1P"), _recorder()
    pipe = Pipeline()
    pipe.__dict__.update(vars(fixture))
    stack = contextlib.ExitStack()
    shadow.instrument_atomic_metadata(stack, Pipeline, rec)
    with pytest.raises(LookupError), stack:
        pipe._start_chain_estimate("1P", _event())
    assert pipe._chain_entry_t_1p == 1.0
    assert Pipeline._start_chain_estimate is original
    assert rec.atomic_metadata_rows[0]["status"] == "original_exception"


def test_wrapper_preserves_signature() -> None:
    class Pipeline:
        def _start_chain_estimate(self, side: str, event: Any,
                                  precomputed_result: Any = None) -> None:
            return None
    before = inspect.signature(Pipeline._start_chain_estimate)
    with contextlib.ExitStack() as stack:
        shadow.instrument_atomic_metadata(stack, Pipeline, _recorder())
        assert inspect.signature(Pipeline._start_chain_estimate) == before


def test_prepare_guards_new_files_and_marks_unadopted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        shadow, "ORIGINAL_PREPARE",
        lambda _args: ({"input_and_code_sha256": {}}, {}),
    )
    receipt, _ = shadow.atomic_prepare(
        SimpleNamespace(start_sec=560.0, end_sec=583.0,
                        output_root=Path("unused"), allow_native_runtime_mismatch=True))
    for path in (Path(shadow.__file__), shadow.LAUNCHER, shadow.TEST):
        assert receipt["input_and_code_sha256"][str(path)] == shadow.base.sha256(path)
    policy = receipt["atomic_landing_policy"]
    assert policy["slide_guard_enabled"] is False
    assert policy["rollback_problem_fixed"] is False
    assert policy["adoption_status"] == "diagnostic_only_not_adopted"


def test_new_functions_obey_fifty_line_limit() -> None:
    for path in (Path(shadow.__file__), Path(__file__)):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                assert node.end_lineno - node.lineno + 1 <= 50, (path.name, node.name)
