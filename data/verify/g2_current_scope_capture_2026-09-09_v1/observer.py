"""既存SM観測をPBと同じ両側scopeへ追加し、旧sinkは変更しない。"""
from __future__ import annotations
import functools
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[2]
HOOK = ROOT.parent / "g2_current_entry_observer_2026-09-08_v1/observer.py"
HOOK_SHA = "ea55257a4d178a9f2976da3c3a05ad09080f6f9a5d263033b23858e2f45432e7"
HOOK_ALIAS = "_video38_current_entry_observer"
BOUNDARY_ALIAS = "_video38_start_gate_end_observer"
SIDECAR = "current_scope.jsonl"
STATUS = "CURRENT_SCOPE_STATUS.json"
RECEIPT = "CURRENT_SCOPE_RECEIPT.json"
REQUIRED = frozenset((SIDECAR, STATUS, RECEIPT))
SUFFIXES = ("step_enter", "detector_return", "sm_return", "step_return")


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False, indent=2)


class Sink:
    """実呼出のみを排他保存。SM未呼出は欠測として保持する。"""
    def __init__(self, state: dict[str, Any], history: Any) -> None:
        pb = state["hidden_probability_observer"]
        self.output, self.history = state["output"], history
        self.source_id, self.run_id = pb.source_id, pb.run_id
        self.expected = tuple(pb.expected)
        self.selected = set(self.expected)
        self.epoch = state["sink"].epoch
        self.rows: list[dict[str, Any]] = []
        self.failures: list[str] = []
        self.closed = False
        self.stream = (self.output / SIDECAR).open("x", encoding="utf-8")

    def emit(self, row: dict[str, Any]) -> None:
        try:
            scope = row["frame_idx"], row["side"]
            if type(row["frame_idx"]) is not int or type(row["side"]) is not str:
                raise ValueError("scope_clock_or_side_type")
            if scope not in self.selected or (row["frame_idx"], row["time_sec"]) != (self.history.frame, self.history.time_sec):
                raise ValueError("scope_or_active_clock")
            if row["kind"] not in {"current_entry_" + name for name in SUFFIXES}:
                raise ValueError("unknown_current_scope_kind")
            value = row | {"source_id": self.source_id, "run_id": self.run_id,
                "software_epoch": self.epoch(row), "physical_hand_certified": False}
            encoded = json.dumps(value, ensure_ascii=False, allow_nan=False)
            self.stream.write(encoded + "\n")
            self.rows.append(json.loads(encoded))
        except BaseException as error:
            self.failures.append(repr(error))
            raise

    def close(self) -> None:
        try:
            self.stream.close()
            self.closed = True
        except BaseException as error:
            self.failures.append(repr(error))
            raise


def coverage(rows: list[dict[str, Any]], expected: list[Any]) -> dict[str, Any]:
    entered, returned, counts = [], [], {}
    for row in rows:
        key = row["frame_idx"], row["side"]
        name = row["kind"].removeprefix("current_entry_")
        if name not in SUFFIXES:
            raise ValueError("unknown_current_scope_kind")
        current = counts.setdefault(key, dict.fromkeys(SUFFIXES, 0))
        if name == "step_enter":
            entered.append(key)
        elif not current["step_enter"] or current["step_return"]:
            raise ValueError("current_scope_order")
        current[name] += 1
        if name == "step_return":
            returned.append(key)
        if any(current[field] > 1 for field in ("step_enter", "sm_return", "step_return")):
            raise ValueError("duplicate_current_scope")
    if not entered or entered != [tuple(value) for value in expected] or returned != entered:
        raise ValueError("current_scope_coverage")
    return {"scope_count": len(entered), "sm_count": sum(c["sm_return"] for c in counts.values()),
        "sm_not_called": [list(key) for key, value in counts.items() if not value["sm_return"]]}


def sticky(original: Any, sink: Sink) -> Any:
    @functools.wraps(original)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        try:
            return original(*args, **kwargs)
        except BaseException as error:
            sink.failures.append(repr(error))
            raise
    return wrapped


def install(stack: Any, collector: Any, history: Any, state: dict[str, Any]) -> Sink:
    hook, boundary = sys.modules.get(HOOK_ALIAS), sys.modules.get(BOUNDARY_ALIAS)
    if hook is None or Path(hook.__file__).resolve() != HOOK or sha(HOOK) != HOOK_SHA:
        raise RuntimeError("fixed_installed_current_hook_required")
    if boundary is None or "current_scope_sink" in state:
        raise RuntimeError("boundary_missing_or_reentry")
    sink = Sink(state, history)
    stack.callback(sink.close)
    sm_class = sys.modules["src.board_state_machine"].BoardStateMachine
    observer = hook.install(stack, collector, history, boundary, sm_class, sink.emit)
    frames = {frame for frame, _ in sink.expected}
    observer.enabled = lambda: history.frame in frames
    cls = collector.RecognitionPipeline
    boundary.base.patch(stack, cls, "_step_side", sticky(cls._step_side, sink))
    connection = state["provisional_current_connection"]
    original_state = connection.state
    connection.state = dict(original_state, sink=sink)
    stack.callback(setattr, connection, "state", original_state)
    state["current_scope_sink"] = sink
    return sink


def finish(state: dict[str, Any]) -> None:
    sink = state["current_scope_sink"]
    if not sink.closed or sink.failures:
        raise RuntimeError("current_scope_capture_failed")
    result = coverage(sink.rows, sink.expected)
    write(sink.output / STATUS, {"closed": sink.closed, "failures": sink.failures})
    write(sink.output / RECEIPT, result | {"expected_scopes": sink.expected,
        "source_id": sink.source_id, "run_id": sink.run_id,
        "sha256": {name: sha(sink.output / name) for name in (SIDECAR, STATUS)},
        "quality_gate_clear": False, "accounting_permission": False})


def verify(output: Path) -> None:
    receipt = json.loads((output / RECEIPT).read_text())
    status = json.loads((output / STATUS).read_text())
    if status != {"closed": True, "failures": []} or set(receipt["sha256"]) != {SIDECAR, STATUS}:
        raise ValueError("saved_current_scope_status_or_index")
    if any(sha(output / name) != digest for name, digest in receipt["sha256"].items()):
        raise ValueError("saved_current_scope_changed")
    rows = [json.loads(line) for line in (output / SIDECAR).read_text().splitlines()]
    for row in rows:
        if any(row[key] != receipt[key] for key in ("source_id", "run_id")):
            raise ValueError("current_scope_source_or_run")
    result = coverage(rows, receipt["expected_scopes"])
    if any(receipt.get(key) != value for key, value in result.items()):
        raise ValueError("saved_current_scope_summary")
