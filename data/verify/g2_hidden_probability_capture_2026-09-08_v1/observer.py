"""実隠し段producerとstep返却PBの同object対応を非変更で保存する。"""
from __future__ import annotations

import contextlib
from enum import Enum
import functools
import hashlib
import inspect
import json
import math
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[2]
SNAPSHOT = PROJECT / ".runtime_snapshots/event_first30_observed_context_v5_2026-08-30"
WITNESS = ROOT.parent / "g2_hsv_correction_witness_2026-09-08_v1/observer.py"
SIDECAR, STATUS, RECEIPT = "hidden_probability.jsonl", "HIDDEN_PROBABILITY_STATUS.json", "HIDDEN_PROBABILITY_RECEIPT.json"
REQUIRED = frozenset((SIDECAR, STATUS, RECEIPT))
ROWS, COLS, STRIDE = 13, 6, 2
SIDES, COLORS = ("1P", "2P"), frozenset((0, 1, 2, 3, 4, 5, 9))
WINDOWS = ((32640, 32800), (34700, 36000))
SUM_TOLERANCE = 1e-12
FIXED = {
    str(WITNESS): "7b3407ccbc46bd5ad48fdcc1fc15810ab3dce2c39866ab42b0d5da3d1084642d",
    str(SNAPSHOT / "src/probabilistic_board.py"): "705a0e0dd1da525c201db11ba135c2abbfa0c846fc23ede01cd8d80d2cae0ab9",
    str(SNAPSHOT / "src/hidden_row_inferrer.py"): "92d4ac4e0fdfb006ba144e66973a2624eeaf06a063aaf78cc5575f1de05a1b6b",
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False, indent=2)


def guards() -> dict[str, str]:
    if any(sha(Path(path)) != digest for path, digest in FIXED.items()):
        raise RuntimeError("hidden_probability_fixed_source_changed")
    return {**FIXED, **{str(ROOT / name): sha(ROOT / name) for name in
        ("observer.py", "test_observer.py", "run_cpu.py", "ASSET_PREFLIGHT.md")}}


def expected() -> list[tuple[int, str]]:
    return [(frame, side) for first, last in WINDOWS for frame in range(first, last + 1, STRIDE) for side in SIDES]


def scalar(value: Any) -> Any:
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float:
        return value if math.isfinite(value) else {"nonfinite": str(value), "type": "builtins.float"}
    if isinstance(value, Enum):
        return scalar(value.value)
    if type(value) in (tuple, list):
        return [scalar(item) for item in value]
    return {"unsupported_type": type(value).__module__ + "." + type(value).__qualname__}


def capture_scope(row: dict[str, Any]) -> dict[str, Any]:
    # 生成時刻ではなく、この呼出を実捕捉した時計を保存する。
    return {key: row[key] for key in ("source_id", "run_id", "frame_idx", "time_sec", "side")}


def state_name(value: Any) -> Any:
    # enum.valueは実原本で小文字。判定欄はname、元値は別欄に残す。
    return value.name if isinstance(value, Enum) else scalar(value)


def probability_value(value: Any, pb_type: Any) -> dict[str, Any]:
    result = {"present": value is not None, "type_valid": type(value) is pb_type,
              "cells": None, "errors": [], "sha256": None}
    if value is None or type(value) is not pb_type:
        result["errors"].append("missing_probability" if value is None else "probability_type")
        return result
    cells: list[Any] = [[None for _ in range(COLS)] for _ in range(ROWS)]
    try:
        for row, col, cell in value.iter_cells():
            if type(row) is not int or type(col) is not int or not 0 <= row < ROWS or not 0 <= col < COLS:
                raise ValueError("probability_cell_coordinates")
            if cells[row][col] is not None or type(cell.probs) is not dict:
                raise ValueError("probability_cell_schema")
            raw = list(cell.probs.items())
            cells[row][col] = [[scalar(color), scalar(p)] for color, p in raw]
            result["errors"].extend(f"{row},{col}:{error}" for error in distribution_errors(raw))
    except Exception as exc:
        result["errors"].append("probability_copy:" + type(exc).__name__)
    if any(cell is None for row in cells for cell in row):
        result["errors"].append("probability_grid_missing_cells")
    result["cells"] = cells
    result["sha256"] = hashlib.sha256(json.dumps(cells, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode()).hexdigest()
    return result


def distribution_errors(pairs: list[tuple[Any, Any]]) -> list[str]:
    errors = []
    if not pairs:
        return ["empty_distribution"]
    if any(type(color) is not int or color not in COLORS for color, _ in pairs):
        errors.append("invalid_probability_color")
    if any(type(p) not in (int, float) or not math.isfinite(p) or not 0 <= p <= 1 for _, p in pairs):
        errors.append("invalid_probability_value")
    elif not math.isclose(math.fsum(p for _, p in pairs), 1.0, rel_tol=0, abs_tol=SUM_TOLERANCE):
        errors.append("probability_sum_not_one")
    return errors


def holds(row: dict[str, Any]) -> list[str]:
    reasons = []
    if row["state"] != "STABLE":
        reasons.append("non_stable")
    if row["is_match_active"] is not True:
        reasons.append("match_active_not_true")
    probability, confirmed = row["probability"], row["confirmed"]
    if probability["errors"]:
        reasons.append("invalid_or_missing_probability")
    if confirmed is None:
        return reasons + ["missing_confirmed"]
    if not probability["errors"]:
        for r in range(1, ROWS):
            for c in range(COLS):
                positive = [pair for pair in probability["cells"][r][c] if pair[1] > 0]
                if positive != [[confirmed["grid"][r][c], 1.0]] or confirmed["grid"][r][c] not in COLORS:
                    return reasons + ["visible_probability_confirmed_mismatch"]
    return reasons


def runtime_modules(state: dict[str, Any]) -> tuple[Any, Any, Any]:
    witness = state["hsv_witness"]
    w = sys.modules.get(type(witness).__module__)
    if w is None or Path(w.__file__).resolve() != WITNESS or type(witness) is not w.Witness or not witness.installed:
        raise RuntimeError("actual_witness_install_required")
    modules = [sys.modules.get("src." + name) for name in ("probabilistic_board", "hidden_row_inferrer")]
    for name, module in zip(("probabilistic_board", "hidden_row_inferrer"), modules, strict=True):
        if module is None or Path(module.__file__).resolve() != SNAPSHOT / "src" / (name + ".py"):
            raise RuntimeError("nonfrozen_hidden_runtime_module")
    pb, hidden = modules
    if hidden.ProbabilisticBoard is not pb.ProbabilisticBoard:
        raise RuntimeError("hidden_probability_type_identity")
    return w, pb, hidden


class Observer:
    """独自scopeで寿命を保持。CPU helper fixtureは実pipeline検収と分離する。"""
    def __init__(self, output: Path, source_id: str, run_id: str, pb_type: Any,
                 board_value: Any, expected: list[tuple[int, str]]) -> None:
        if any(type(text) is not str or not text.strip() for text in (source_id, run_id)):
            raise ValueError("source_and_run_required")
        self.output, self.source_id, self.run_id = output, source_id, run_id
        self.pb_type, self.board_value, self.expected = pb_type, board_value, expected
        self.active, self.rows, self.failures, self.bindings = None, [], [], []
        self.closed, self.installed = False, False
        self.stream = (output / SIDECAR).open("x", encoding="utf-8")

    def fail(self, reason: str, exc: BaseException) -> None:
        self.failures.append({"reason": reason, "exception": type(exc).__name__})
        if self.active is not None:
            self.active["row"]["instrumentation_errors"].append(reason)

    def begin(self, pipe: Any, side: str, frame: int, clock: float, bound: Any) -> None:
        if self.active is not None:
            raise RuntimeError("nested_probability_scope")
        if side not in SIDES or type(frame) is not int or frame < 0 or type(clock) not in (int, float) or not math.isfinite(clock) or clock < 0:
            raise ValueError("invalid_probability_scope_clock")
        sm = bound.arguments.get("sm")
        row = {"source_id": self.source_id, "run_id": self.run_id, "frame_idx": frame, "time_sec": clock,
            "side": side, "capture_stage": "step_return", "is_match_active": scalar(bound.arguments.get("is_active")),
            "end_lock": None, "end_lock_observed": False, "state": None, "state_value": None,
            "raw": self.board_value(bound.arguments.get("cnn_board")),
            "entry_state": state_name(getattr(getattr(sm, "context", None), "state", None)),
            "entry_state_value": scalar(getattr(getattr(sm, "context", None), "state", None)),
            "entry_confirmed": self.board_value(getattr(getattr(sm, "context", None), "confirmed_board", None)),
            "confirmed": None, "probability": probability_value(None, self.pb_type),
            "next_pair": None, "dnext_pair": None,
            "origin": "missing", "matched_infer_call": None, "matched_distribution_unchanged": None,
            "infer_calls": [], "hold_reasons": [], "instrumentation_errors": [], "step_exception": None,
            "current_publication_allowed": False, "accounting_commit_allowed": False,
            "quality_gate_clear": False, "physical_identity_certified": False}
        row["probability"]["capture_scope"] = capture_scope(row)
        self.active = {"pipe": pipe, "row": row, "objects": []}

    def infer(self, original: Any) -> Any:
        signature = inspect.signature(original)
        @functools.wraps(original)
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            if self.active is None:
                return original(*args, **kwargs)
            scope, item = self.active, None
            try:
                bound = signature.bind(*args, **kwargs)
                item = self.infer_start(bound)
            except Exception as exc:
                self.fail("infer_input_capture", exc)
            try:
                result = original(*args, **kwargs)
            except BaseException as exc:
                if item is not None:
                    item.update(status="exception", exception=type(exc).__name__)
                raise
            try:
                if item is not None and self.active is scope:
                    self.infer_return(item, result)
            except Exception as exc:
                self.fail("infer_return_capture", exc)
            return result
        return wrapped

    def infer_start(self, bound: Any) -> dict[str, Any]:
        data = bound.arguments
        item = {"call_index": len(self.active["row"]["infer_calls"]) + 1, "status": "called",
            "capture_scope": capture_scope(self.active["row"]),
            "inputs": {"previous": self.board_value(data.get("prev_board")),
                       "current": self.board_value(data.get("cur_board")),
                       "previous_next_pair": scalar(data.get("prev_next_pair"))},
            "apply_calibration_argument": scalar(data.get("apply_calibration", False)),
            "probability": None, "result": None, "exception": None, "returned_pb_is_final": False}
        self.active["row"]["infer_calls"].append(item)
        return item

    def infer_return(self, item: dict[str, Any], result: Any) -> None:
        item["status"] = "returned"
        if type(result) is not tuple or len(result) != 2:
            raise ValueError("infer_return_tuple_schema")
        pb, detail = result
        item["probability"] = probability_value(pb, self.pb_type)
        item["probability"]["capture_scope"] = dict(item["capture_scope"])
        item["result"] = {name: scalar(getattr(detail, name, None)) for name in
            ("n_new_cells", "cells_added_to_hidden", "cells_with_distribution", "skipped_reason")}
        self.active["objects"].append((item, pb))

    def end(self, result: Any, error: BaseException | None) -> None:
        if self.active is None:
            return
        scope, self.active = self.active, None
        row = scope["row"]
        row["step_exception"] = None if error is None else type(error).__name__
        try:
            if result is not None:
                if result.side != row["side"]:
                    raise ValueError("side_result_scope_mismatch")
                row["state"], row["state_value"] = state_name(result.state), scalar(result.state)
                row["confirmed"] = self.board_value(result.confirmed_board)
                row["next_pair"] = scalar(getattr(result, "next_pair", None))
                row["dnext_pair"] = scalar(getattr(result, "dnext_pair", None))
                final = result.prob_board
                row["probability"] = probability_value(final, self.pb_type)
                row["probability"]["capture_scope"] = capture_scope(row)
                matches = [item for item, pb in scope["objects"] if pb is not None and pb is final]
                for item in matches:
                    item["returned_pb_is_final"] = True
                row["origin"] = "missing" if final is None else "returned_unclassified"
                if len(matches) == 1:
                    row.update(origin="infer_hidden_row", matched_infer_call=matches[0]["call_index"],
                        matched_distribution_unchanged=matches[0]["probability"]["cells"] == row["probability"]["cells"])
            row["hold_reasons"] = holds(row)
            if error is not None:
                row["hold_reasons"].append("step_exception")
            if any(item["status"] == "exception" for item in row["infer_calls"]):
                row["hold_reasons"].append("infer_exception")
        except Exception as exc:
            self.fail("step_return_capture", exc)
            row["instrumentation_errors"].append("step_return_capture")
        self.rows.append(row)
        self.stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        self.stream.flush()

    def step(self, original: Any, selected: Any) -> Any:
        signature = inspect.signature(original)
        @functools.wraps(original)
        def wrapped(pipe: Any, side: str, frame_idx: int, time_sec: float, *args: Any, **kwargs: Any) -> Any:
            if not selected(frame_idx, side):
                return original(pipe, side, frame_idx, time_sec, *args, **kwargs)
            previous, self.active = self.active, None
            if previous is not None:
                self.fail("nested_probability_scope", RuntimeError())
            try:
                self.begin(pipe, side, frame_idx, time_sec,
                    signature.bind(pipe, side, frame_idx, time_sec, *args, **kwargs))
            except Exception as exc:
                self.fail("step_entry_capture", exc)
            result, error = None, None
            try:
                result = original(pipe, side, frame_idx, time_sec, *args, **kwargs)
                return result
            except BaseException as exc:
                error = exc
                raise
            finally:
                try:
                    self.end(result, error)
                except Exception as exc:
                    self.fail("sidecar_write", exc)
                self.active = previous
        return wrapped

    def close(self) -> None:
        self.stream.close()
        self.closed = True
        write(self.output / STATUS, {"closed": True, "installed": self.installed, "failures": self.failures})


def bind_namespaces(stack: Any, observer: Observer, functions: dict[Any, Any], hidden: Any, w: Any) -> None:
    seen: dict[int, int] = {}
    original = hidden.infer_hidden_row
    if not inspect.isfunction(original) or original.__globals__ is not vars(hidden) or Path(original.__code__.co_filename).resolve() != SNAPSHOT / "src/hidden_row_inferrer.py":
        raise RuntimeError("nonfrozen_hidden_export")
    for code, function in functions.items():
        namespace = function.__globals__
        existed = id(namespace) in seen
        index = seen.setdefault(id(namespace), len(seen))
        observer.bindings.append({"namespace_index": index, "code_first_line": code.co_firstlineno,
            "code_sha256": hashlib.sha256(code.co_code).hexdigest()})
        if not existed:
            if namespace.get("infer_hidden_row") is not original:
                raise RuntimeError("unknown_hidden_wrapper")
            w.P._patch_mapping(stack, namespace, "infer_hidden_row", observer.infer(original))


def install(stack: Any, collector: Any, history: Any, state: dict[str, Any], *, source_id: str, run_id: str) -> None:
    guards()
    if "hidden_probability_observer" in state:
        raise RuntimeError("hidden_probability_install_reentry")
    w, pb, hidden = runtime_modules(state)
    scopes = [(frame, side) for frame, _ in w.expected() for side in SIDES]
    if scopes != expected():
        raise RuntimeError("witness_probability_scope_policy_changed")
    functions = w.step_functions(collector.RecognitionPipeline._step_side, {history.step_code, state["pending_code"]})
    if functions[state["pending_code"]].__globals__ is not state["pending_globals"]:
        raise RuntimeError("pending_globals_identity")
    if any(Path(code.co_filename).resolve() != SNAPSHOT / "src/recognition_pipeline.py" for code in functions):
        raise RuntimeError("nonfrozen_actual_step_code")
    observer = Observer(state["output"], source_id, run_id, pb.ProbabilisticBoard, w.board_value, scopes)
    state["hidden_probability_observer"] = observer
    with contextlib.ExitStack() as pending:
        pending.callback(observer.close)
        bind_namespaces(pending, observer, functions, hidden, w)
        original = collector.RecognitionPipeline._step_side
        wrapped = observer.step(original, lambda frame, side: side in SIDES and w.selected(frame, "2P"))
        w.P._patch(pending, collector.RecognitionPipeline, "_step_side", wrapped)
        observer.installed = True
        stack.enter_context(pending.pop_all())


def summary(rows: list[dict[str, Any]], expected: list[tuple[int, str]]) -> dict[str, Any]:
    if [(row["frame_idx"], row["side"]) for row in rows] != expected or not rows:
        raise RuntimeError("hidden_probability_scope_coverage")
    scope = {(row["source_id"], row["run_id"]) for row in rows}
    if len(scope) != 1 or any(row["instrumentation_errors"] for row in rows):
        raise RuntimeError("hidden_probability_scope_or_instrumentation_failed")
    for row in rows:
        if type(row["frame_idx"]) is not int or type(row["side"]) is not str or row["side"] not in SIDES:
            raise RuntimeError("hidden_probability_clock_schema")
        clock = row["time_sec"]
        if type(clock) not in (int, float) or not math.isfinite(clock) or clock < 0:
            raise RuntimeError("hidden_probability_clock_schema")
        if any(type(row[key]) is not str or not row[key].strip() for key in ("source_id", "run_id")):
            raise RuntimeError("hidden_probability_source_identity")
        calls = row["infer_calls"]
        scope_json = json.dumps(capture_scope(row), sort_keys=True, allow_nan=False)
        saved_scopes = [row["probability"].get("capture_scope")] + [call.get("capture_scope") for call in calls]
        if any(json.dumps(value, sort_keys=True, allow_nan=False) != scope_json for value in saved_scopes):
            raise RuntimeError("hidden_probability_capture_scope_mismatch")
        if any(type(call["call_index"]) is not int or call["call_index"] != index + 1 or
               call["status"] not in ("returned", "exception") for index, call in enumerate(calls)):
            raise RuntimeError("hidden_probability_infer_boundary_incomplete")
    return {"scope_count": len(rows), "infer_calls": sum(len(row["infer_calls"]) for row in rows),
        "origins": {origin: sum(row["origin"] == origin for row in rows) for origin in
            ("infer_hidden_row", "returned_unclassified", "missing")},
        "held_scopes": sum(bool(row["hold_reasons"]) for row in rows),
        "current_publication_allowed": False, "accounting_commit_allowed": False, "quality_gate_clear": False}


def finish(state: dict[str, Any]) -> dict[str, Any]:
    observer = state["hidden_probability_observer"]
    if not observer.closed or not observer.installed or observer.failures or observer.active is not None:
        raise RuntimeError("hidden_probability_lifetime_or_sticky_failure")
    result = summary(observer.rows, observer.expected)
    result.update(expected_scopes=observer.expected, bindings=observer.bindings,
        sha256={name: sha(observer.output / name) for name in (SIDECAR, STATUS)})
    write(observer.output / RECEIPT, result)
    return result


def verify(output: Path, *, expected_scopes: list[tuple[int, str]] | None = None) -> dict[str, Any]:
    status = json.loads((output / STATUS).read_text())
    receipt = json.loads((output / RECEIPT).read_text())
    if status != {"closed": True, "installed": True, "failures": []}:
        raise RuntimeError("hidden_probability_status_failed")
    if set(receipt["sha256"]) != {SIDECAR, STATUS}:
        raise RuntimeError("hidden_probability_artifact_set")
    if any(sha(output / name) != digest for name, digest in receipt["sha256"].items()):
        raise RuntimeError("hidden_probability_artifact_changed")
    rows = [json.loads(line) for line in (output / SIDECAR).read_text().splitlines()]
    scopes = expected() if expected_scopes is None else expected_scopes
    if [tuple(item) for item in receipt["expected_scopes"]] != scopes:
        raise RuntimeError("hidden_probability_expected_scope_changed")
    result = summary(rows, scopes)
    if any(receipt.get(key) != value for key, value in result.items()):
        raise RuntimeError("hidden_probability_receipt_mismatch")
    return receipt
