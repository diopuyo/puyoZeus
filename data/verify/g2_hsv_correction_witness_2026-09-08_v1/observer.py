"""実補正scopeのextract→classify→int変換を追加推論なしで保存する。"""
from __future__ import annotations

import contextlib
import functools
import hashlib
import inspect
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any, Callable

from scripts import placement_causal_observation_v1 as P

TASK_DIR = Path(__file__).resolve().parent
PROJECT = TASK_DIR.parents[2]
SNAPSHOT = PROJECT / ".runtime_snapshots/event_first30_observed_context_v5_2026-08-30"
SIDECAR, STATUS, RECEIPT = "hsv_correction_witness.jsonl", "HSV_WITNESS_STATUS.json", "HSV_WITNESS_RECEIPT.json"
REQUIRED = frozenset((SIDECAR, STATUS, RECEIPT))
WINDOWS = ((32640, 32800), (34700, 36000))
HELPER = "_apply_landing_observed_color_correction"
OUTER_CONVERSION_LINE, INNER_CONVERSION_LINE = 609, 930
FIXED = {
    str(PROJECT / "scripts/placement_causal_observation_v1.py"):
        "60626d90968abe2f5ec2c141b47406b699f7125e0f76870550f1414d7eda8415",
    str(SNAPSHOT / "src/recognition_pipeline.py"):
        "6e945d7584025ae9803e14c9ac079b1a468f483d0beac681a1f0e580cdd10e02",
    str(SNAPSHOT / "src/placement_inferrer.py"):
        "412a15120db0d6e9c10f38ad8d5fa8dff1dfa05f05f3d1a14b71f4ce10610b82",
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def guards() -> dict[str, str]:
    for path, digest in FIXED.items():
        if sha(Path(path)) != digest:
            raise RuntimeError("hsv_fixed_source_changed")
    return {**FIXED, **{str(TASK_DIR / name): sha(TASK_DIR / name)
                        for name in ("observer.py", "ASSET_PREFLIGHT.md")}}


def selected(frame: Any, side: Any) -> bool:
    return side == "2P" and type(frame) is int and frame % 2 == 0 and any(a <= frame <= b for a, b in WINDOWS)


def expected() -> list[tuple[int, str]]:
    return [(frame, "2P") for start, end in WINDOWS for frame in range(start, end + 1, 2)]


def board_value(board: Any) -> Any:
    if board is None:
        return None
    return {**P._board_value(board), "grid": board._grid.copy().tolist()}


def scalar(value: Any) -> dict[str, Any]:
    """任意objectのint/reprを再実行せず、既知scalarだけ保存する。"""
    known = type(value) in (int, bool, str, type(None))
    return {"type": type(value).__module__ + "." + type(value).__qualname__,
            "value": value if known else None, "value_serialized": known}


def patch_instance(stack: Any, obj: Any, name: str, value: Any) -> None:
    """継承methodをinstanceに置いた場合は、元の属性不存在まで復元する。"""
    owned, old = name in vars(obj), vars(obj).get(name)
    setattr(obj, name, value)
    stack.callback(setattr, obj, name, old) if owned else stack.callback(delattr, obj, name)


class ConversionTrace:
    """二つの実int(classify)行だけ観測し、元global/local traceを一回ずつ委譲。"""
    def __init__(self, witness: Any, targets: dict[Any, int]) -> None:
        self.witness, self.targets, self.previous = witness, targets, sys.gettrace()

    def observe(self, frame: Any, event: str, arg: Any) -> None:
        try:
            active = self.witness.active
            call = None if active is None else active.get("last_call")
            if call is None or call["frame"] is not frame:
                return
            item = call["item"]
            if event == "exception" and frame.f_lineno == self.targets[frame.f_code]:
                if item["status"] == "exception":
                    item["propagated_exception_identity"] = arg[1] is call["exception"]
                elif item["conversion"] == "pending":
                    item.update(conversion="exception", conversion_exception=type(arg[1]).__name__)
            elif event == "line" and frame.f_lineno > self.targets[frame.f_code] and item["conversion"] == "pending":
                item.update(conversion="returned", converted_value=scalar(frame.f_locals["hsv_color"]))
        except Exception as exc:
            self.witness.observer.fail_scope("conversion_trace", exc)

    def __call__(self, frame: Any, event: str, arg: Any) -> Any:
        prior = self.previous(frame, event, arg) if self.previous is not None else None
        if frame.f_code not in self.targets:
            return prior
        self.observe(frame, event, arg)
        def local(current: Any, kind: str, value: Any) -> Any:
            nonlocal prior
            if prior is not None:
                prior = prior(current, kind, value)
            self.observe(current, kind, value)
            return local
        return local


class Witness:
    """sidecarは旧history emitと分離。閉じた後もstateから一回finish可能。"""
    def __init__(self, output: Path, tracker: Any) -> None:
        self.output, self.active, self.suspended = output, None, False
        self.rows: list[dict[str, Any]] = []
        self.stream = (output / SIDECAR).open("x", encoding="utf-8")
        self.observer = P.PlacementCausalObserver(SimpleNamespace(emit=self.emit, generation_recorder=tracker))
        self.closed, self.installed, self.bindings = False, False, []

    def emit(self, row: dict[str, Any]) -> None:
        value = json.loads(json.dumps({**row, "current_publication_allowed": False,
            "accounting_commit_allowed": False, "physical_identity_certified": False}, allow_nan=False))
        self.rows.append(value)
        self.stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")
        self.stream.flush()

    def close(self) -> None:
        self.stream.close()
        self.closed = True
        write(self.output / STATUS, {"closed": True, "installed": self.installed,
                                    "failures": self.observer.failures})

    def extract(self, original: Any) -> Any:
        @functools.wraps(original)
        def wrapped(frame: Any, region: Any, row: int, col: int) -> Any:
            result = original(frame, region, row, col)
            if self.active is not None and not self.suspended:
                try:
                    record = {"kind": "extract", "sequence": len(self.active["events"]) + 1,
                        "row": row, "col": col, "patch": P._frame_value(result),
                        "frame_is_helper_input": frame is self.active["frame"],
                        "region_is_helper_input": region is self.active["region"]}
                    self.active["events"].append(record)
                    self.active["patch"] = (result, record, False)
                except Exception as exc:
                    self.observer.fail_scope("extract_capture", exc)
            return result
        return wrapped

    def classify(self, original: Any) -> Any:
        @functools.wraps(original)
        def wrapped(patch: Any, *args: Any, **kwargs: Any) -> Any:
            if self.active is None or self.suspended:
                return original(patch, *args, **kwargs)
            item = self.classify_begin(patch, inspect.currentframe().f_back)
            try:
                result = original(patch, *args, **kwargs)
            except BaseException as exc:
                item.update(status="exception", exception=type(exc).__name__, conversion="not_attempted")
                self.active["last_call"]["exception"] = exc
                raise
            item.update(status="returned", returned=scalar(result), conversion="pending")
            return result
        return wrapped

    def classify_begin(self, patch: Any, frame: Any) -> dict[str, Any]:
        prior = self.active.get("patch")
        matched = bool(prior is not None and prior[0] is patch and not prior[2])
        item = {"kind": "classify", "sequence": len(self.active["events"]) + 1,
            "extract_sequence": None if prior is None else prior[1]["sequence"],
            "row": None if prior is None else prior[1]["row"], "col": None if prior is None else prior[1]["col"],
            "argument_is_extract_return": matched, "status": "entered", "conversion": "not_attempted"}
        self.active["events"].append(item)
        self.active["last_call"] = {"frame": frame, "item": item}
        if not matched:
            self.observer.fail_scope("classify_patch_identity", "実extract返却と一回結合できません")
        if prior is not None:
            self.active["patch"] = (prior[0], prior[1], True)
        return item

    def correct(self, original: Any, inner: Any) -> Any:
        observed = P._wrap_observed_color(self.observer, original)
        @functools.wraps(original)
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            if self.observer.scope is None:
                return original(*args, **kwargs)
            if self.active is not None:
                self.observer.fail_scope("correction_reentry", "補正scope再入")
                self.suspended = True
                try:
                    return original(*args, **kwargs)
                finally:
                    self.suspended = False
            return self.call_correct(observed, original, inner, args, kwargs)
        return wrapped

    def call_correct(self, observed: Any, original: Any, inner: Any, args: Any, kwargs: Any) -> Any:
        frame, region = P._arg(args, kwargs, 4, "frame_bgr"), P._arg(args, kwargs, 5, "region")
        reader = P._arg(args, kwargs, 3, "reader")
        classifier = getattr(reader, "_classifier", None)
        classifier = getattr(classifier, "_hsv", classifier)
        events: list[dict[str, Any]] = []
        self.active = {"frame": frame, "region": region, "events": events}
        before = {"input": board_value(P._arg(args, kwargs, 0, "inferred")),
                  "raw": board_value(P._arg(args, kwargs, 2, "cnn_board"))}
        try:
            with contextlib.ExitStack() as stack:
                if classifier is not None and hasattr(classifier, "classify"):
                    patch_instance(stack, classifier, "classify", self.classify(classifier.classify))
                tracer = ConversionTrace(self, {inspect.unwrap(original).__code__: OUTER_CONVERSION_LINE,
                                                inner.__code__: INNER_CONVERSION_LINE})
                sys.settrace(tracer)
                stack.callback(sys.settrace, tracer.previous)
                result = observed(*args, **kwargs)
            self.observer.capture("hsv_correction_witness", lambda: {
                "before": before, "result": board_value(result), "events": events,
                "helper_returned_normally": True, "classify_calls": sum(e["kind"] == "classify" for e in events)})
            return result
        finally:
            self.active = None


def step_functions(function: Any, wanted: set[Any]) -> dict[Any, Any]:
    """実closure/wraps鎖だけを探索し、同filenameだけの一致は採用しない。"""
    found, seen, pending = {}, set(), [function]
    while pending:
        current = pending.pop()
        if not inspect.isfunction(current) or id(current) in seen:
            continue
        seen.add(id(current))
        if len(seen) > 128:
            raise RuntimeError("step_closure_search_limit")
        if current.__code__ in wanted:
            found[current.__code__] = current
        pending.append(getattr(current, "__wrapped__", None))
        pending.extend(cell.cell_contents for cell in (current.__closure__ or ()))
    if set(found) != wanted:
        raise RuntimeError("actual_pending_or_grace_code_missing")
    return found


def connect_resolve(stack: Any, witness: Witness, history: Any, namespace: dict[str, Any],
                    inner: Any, seen_cells: set[int]) -> str:
    """実exportだけmapping、認証済みdetailsだけ内側cellへ設置する。"""
    original = namespace["resolve_after_placement"]
    if original is inner.resolve_after_placement:
        if (not inspect.isfunction(original) or original.__globals__ is not vars(inner)
                or Path(original.__code__.co_filename).resolve() != SNAPSHOT / "src/placement_inferrer.py"):
            raise RuntimeError("nonfrozen_resolve_export")
        P._patch_mapping(stack, namespace, "resolve_after_placement", P._wrap_resolve(witness.observer, original))
        return "frozen_export_mapping"
    cell, resolve = P._details_original_cell(original, history)
    if id(cell) not in seen_cells:
        seen_cells.add(id(cell))
        cell.cell_contents = P._wrap_resolve(witness.observer, resolve)
        stack.callback(P._restore_cell, cell, resolve)
    return "details_original_cell"


def connect_functions(stack: Any, witness: Witness, history: Any, functions: dict[Any, Any]) -> None:
    seen_namespaces: dict[int, int] = {}
    resolve_modes: dict[int, str] = {}
    seen_cells: set[int] = set()
    inner = sys.modules["src.placement_inferrer"]
    if Path(inner.__file__).resolve() != SNAPSHOT / "src/placement_inferrer.py":
        raise RuntimeError("nonfrozen_placement_module")
    P._patch(stack, inner, "_extract_cell_patch_from_frame", witness.extract(inner._extract_cell_patch_from_frame))
    for code, function in functions.items():
        namespace = function.__globals__
        existed = id(namespace) in seen_namespaces
        index = seen_namespaces.setdefault(id(namespace), len(seen_namespaces))
        witness.bindings.append({"code_first_line": code.co_firstlineno,
                                 "code_sha": hashlib.sha256(code.co_code).hexdigest(), "namespace_index": index})
        if existed:
            witness.bindings[-1]["resolve_binding"] = resolve_modes[id(namespace)]
            continue
        original = namespace[HELPER]
        if Path(inspect.unwrap(original).__code__.co_filename).resolve() != SNAPSHOT / "src/recognition_pipeline.py":
            raise RuntimeError("nonfrozen_correction_helper")
        P._patch_mapping(stack, namespace, "infer_placement", P._wrap_infer(witness.observer, namespace["infer_placement"]))
        P._patch_mapping(stack, namespace, HELPER, witness.correct(original, inner.correct_landing_cells_by_observed_color))
        mode = connect_resolve(stack, witness, history, namespace, inner, seen_cells)
        resolve_modes[id(namespace)] = mode
        witness.bindings[-1]["resolve_binding"] = mode


def install(stack: Any, collector: Any, history: Any, state: dict[str, Any]) -> None:
    guards()
    if history.step_code is state["pending_code"]:
        raise RuntimeError("distinct_pending_and_grace_code_required")
    functions = step_functions(collector.RecognitionPipeline._step_side, {history.step_code, state["pending_code"]})
    if functions[state["pending_code"]].__globals__ is not state["pending_globals"]:
        raise RuntimeError("pending_namespace_identity")
    witness = Witness(state["output"], state["tracker"])
    state["hsv_witness"] = witness
    stack.callback(witness.close)
    connect_functions(stack, witness, history, functions)
    original = collector.RecognitionPipeline._step_side
    observed = P._wrap_step(witness.observer, original)
    @functools.wraps(original)
    def step(pipe: Any, side: str, frame: int, clock: float, *args: Any, **kwargs: Any) -> Any:
        if not selected(frame, side):
            return original(pipe, side, frame, clock, *args, **kwargs)
        return observed(pipe, side, frame, clock, *args, **kwargs)
    P._patch(stack, collector.RecognitionPipeline, "_step_side", step)
    witness.installed = True


def summarize(rows: list[dict[str, Any]], scopes: list[tuple[int, str]]) -> dict[str, Any]:
    if [(row["frame_idx"], row["side"]) for row in rows] != scopes:
        raise RuntimeError("witness_scope_coverage")
    corrections = [call for row in rows for call in row["calls"] if call["call_kind"] == "hsv_correction_witness"]
    calls = [event for call in corrections for event in call["events"] if event["kind"] == "classify"]
    if not corrections or any(row.get("instrumentation_errors") for row in rows):
        raise RuntimeError("missing_or_failed_correction_witness")
    if any(not call["argument_is_extract_return"] or call["conversion"] == "pending" for call in calls):
        raise RuntimeError("incomplete_classify_binding")
    return {"scope_count": len(rows), "correction_calls": len(corrections), "classify_calls": len(calls),
        "classifier_exceptions": sum(call["status"] == "exception" for call in calls),
        "conversion_exceptions": sum(call["conversion"] == "exception" for call in calls),
        "current_publication_allowed": False, "accounting_commit_allowed": False, "quality_gate_clear": False}


def finish(state: dict[str, Any]) -> dict[str, Any]:
    witness = state["hsv_witness"]
    if not witness.closed or not witness.installed:
        raise RuntimeError("witness_lifetime_not_closed")
    witness.observer.assert_complete()
    result = summarize(witness.rows, expected())
    result.update(bindings=witness.bindings, sha256={name: sha(witness.output / name) for name in (SIDECAR, STATUS)})
    write(witness.output / RECEIPT, result)
    return result


def verify(output: Path) -> dict[str, Any]:
    status = json.loads((output / STATUS).read_text())
    receipt = json.loads((output / RECEIPT).read_text())
    if status != {"closed": True, "installed": True, "failures": []}:
        raise RuntimeError("witness_status_failed")
    if set(receipt["sha256"]) != {SIDECAR, STATUS}:
        raise RuntimeError("witness_artifact_set")
    for name, digest in receipt["sha256"].items():
        if sha(output / name) != digest:
            raise RuntimeError("witness_artifact_changed")
    rows = [json.loads(line) for line in (output / SIDECAR).read_text().splitlines()]
    result = summarize(rows, expected())
    if any(receipt.get(key) != value for key, value in result.items()):
        raise RuntimeError("witness_receipt_mismatch")
    return receipt
