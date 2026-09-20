"""修復 grace 背景の局所 step だけで固定 B を試す。一般採用ではない。"""
from __future__ import annotations

import argparse
from collections import Counter
import contextlib
import functools
import importlib.util
import inspect
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any

from scripts import diagnose_video38_confirmed_collapse_v1 as base

HOME = Path(__file__).resolve().parent
PRIOR_ROOT = HOME.parent / "g2_resolved_grace_guard_adapter_2026-09-08_v1"
PRIOR_SHA = ("615235f604da82261abbaf50ab07f72054c4de517b4576dbafcec91e6c94ab18",
             "77f4f95e5966ccdaaa895904d539a07fe88307b5e85c92693373abd0df3d3bba",
             "170d4e12f7a6eef2d1210668553a8eeb40871c0b36914ebe1551f6c38cadbcbc")
B_ROOT = HOME.parent / "g2_chigiri_completion_shadow_2026-09-08_v1"
B_SHA = {"shadow.py": "0ed36d8026dd9ccae71c1c5e44a5c1705930bd9036e61e246ec7dff10cdc39e8",
    "completion_probe.py": "3e882424ecb8c5356b2f0ee52fca34ea0ca13ac34877d3ffd94266dbea715383",
    "test_shadow.py": "27d1759100916e5b764677780b536349450e5a5153d8f34a52af1981518313ee",
    "run_cpu.py": "1379ecb08ab30c14c3efc0bffc8586b596b90cda8a57dc7b8b1f1f94cda2a8bd",
    "ASSET_PREFLIGHT.md": "ea57b9c459e7427c10d9150ae955a1a2a71e01baef5e2962521a7ade24f41eb3"}
REFERENCE = HOME.parent / "video38_resolved_grace_guard_live_2026-09-08_v1"
REFERENCE_SHA = "5c17efbb81123d4147fc59bc2bd6e660b7c65264df2b48b5246e7aaf6ee37c84"
FORMAT = "video38-chigiri-completion-shadow/v1"
SIDECAR, REPORT = "chigiri_completion.jsonl", "CHIGIRI_COMPLETION_RECEIPT.json"
DIFF, STATUS = "CHIGIRI_COMPLETION_COMPARISON.json", "CHIGIRI_COMPLETION_STATUS.json"
EXTRAS = {SIDECAR, REPORT, DIFF, STATUS}
FIRST, LAST, STRIDE, FPS = 34468, 34516, 2, 60
RAW_COUNT, NEXT_COUNT = 7248, 2100
BASELINE_COUNT, COMPLETE_COUNT, PARTIAL_COUNT = 50, 52, 51
BASELINE_SHA = "5ac993a875e6ab77a2e8aaacfddcd6431473809af3c19f207a490e644892b495"
COMPLETE_SHA = "4c0531b1a0e4f784509bce8545a9989e24c2074ceacbaa194a257436ab16ecb9"
NEXT_KIND = "next_enqueue_live_invocation"
PREFIX = "chigiri_completion_"
SERIALIZER = HOME.parent / "g2_chigiri_exit_observer_2026-09-08_v1/observer.py"
SERIALIZER_SHA = "b49878718a7c1e213616ab41bf2c856391a56131610ce8ee6f5aeca5cf23298b"


def load(name: str, path: Path) -> Any:
    if name in sys.modules:
        raise RuntimeError("unexpected_loaded_alias")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(str(path))
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    try:
        spec.loader.exec_module(value)
        return value
    except BaseException:
        sys.modules.pop(name, None)
        raise


base.assert_unchanged({str(PRIOR_ROOT / "adapter.py"): PRIOR_SHA[0],
                      **{str(B_ROOT / name): sha for name, sha in B_SHA.items()}})
prior = load("_completion_repaired_adapter", PRIOR_ROOT / "adapter.py")
entry = prior.entry
shadow = load("_completion_fixed_shadow", B_ROOT / "shadow.py")
base.assert_unchanged({str(SERIALIZER): SERIALIZER_SHA})
serializer = load("_completion_serializer_reuse", SERIALIZER)
OLD_ADDITIONAL, OLD_PREPARE = prior.additional_guards, prior.prepare
OLD_INSTRUMENT, OLD_FINISH = prior.instrument, prior.finish
OLD_FORMAT, OLD_EXTRAS = prior.FORMAT, prior.EXTRAS


def selected(frame: int, side: str) -> bool:
    return type(frame) is int and side == "1P" and FIRST <= frame <= LAST and frame % STRIDE == 0


def additional_guards(args: Any) -> dict[str, str]:
    old = argparse.Namespace(**vars(args))
    for name, sha in zip(("script", "test", "launcher"), PRIOR_SHA, strict=True):
        setattr(old, name + "_sha256", sha)
    hashes = OLD_ADDITIONAL(old)
    for name, file in (("script", "adapter.py"), ("test", "test_adapter.py"), ("launcher", "launcher.sh")):
        sha = getattr(args, name + "_sha256")
        if not isinstance(sha, str) or len(sha) != 64:
            raise ValueError("new_sha_required")
        hashes[str(HOME / file)] = sha
    hashes[str(HOME / "ASSET_PREFLIGHT.md")] = base.sha256(HOME / "ASSET_PREFLIGHT.md")
    hashes[str(SERIALIZER)] = SERIALIZER_SHA
    hashes.update({str(B_ROOT / name): sha for name, sha in B_SHA.items()})
    hashes[str(REFERENCE / "COMPLETE")] = REFERENCE_SHA
    base.assert_unchanged(hashes)
    complete = base.read_json(REFERENCE / "COMPLETE")
    if (complete.get("format_version") != OLD_FORMAT or type(complete.get("child_exit_code")) is not int
            or complete["child_exit_code"] != 0 or len(complete["sha256"]) != 16):
        raise ValueError("repaired_reference_contract")
    hashes.update({str(REFERENCE / name): sha for name, sha in complete["sha256"].items()})
    base.assert_unchanged(hashes)
    return hashes


def prepare(args: Any, new_args: Any) -> Any:
    receipt, config = OLD_PREPARE(args, new_args)
    receipt["chigiri_completion_shadow"] = contract()
    return receipt, config


def contract() -> dict[str, Any]:
    return {"side": "1P", "first": FIRST, "last": LAST, "stride": STRIDE,
        "reference_root": str(REFERENCE), "saved_sm_injection": False,
        "expected_grid_is_evaluation_only": True, "physical_hand_certified": False,
        "unresolved_last_scope_is_failure": True, "window_extension_allowed": False,
        "next_kind": NEXT_KIND, "required_raw_rows": RAW_COUNT, "required_next_rows": NEXT_COUNT,
        "quality_gate_clear": False, "general_publication_allowed": False, "general_accounting_allowed": False}


def snapshot(pipe: Any, side: str) -> dict[str, Any]:
    ctx = getattr(pipe, "_sm_" + side.lower()).context
    return {"state": ctx.state.value, "confirmed": base.board_value(ctx.confirmed_board),
        "pending": base.board_value(ctx.pending_board), "context_frame": ctx.frame_idx,
        "context_time": ctx.time_sec, "next_queue": list(ctx.next_queue),
        "accounting": entry.previous.history.accounting_snapshot(pipe, side)}


class Sink(entry.Sidecar):
    def emit(self, row: dict[str, Any]) -> None:
        try:
            frame, clock = row["frame_idx"], row["time_sec"]
            if not selected(frame, row["side"]) or type(clock) not in (int, float):
                raise ValueError("target_scope_required")
            if not math.isfinite(clock) or clock != frame / FPS or not row["kind"].startswith(PREFIX):
                raise ValueError("scope_clock_or_kind")
            value = {**row, "software_epoch": self.epoch(row), "physical_hand_certified": False}
            encoded = json.dumps(value, ensure_ascii=False, allow_nan=False)
            self.stream.write(encoded + "\n")
            self.rows.append(json.loads(encoded))
        except BaseException as exc:
            self.failed(exc, "completion_sink")
            raise

    def summary(self) -> dict[str, Any]:
        self.require_ok()
        groups: dict[int, list[dict[str, Any]]] = {}
        for row in self.rows:
            groups.setdefault(row["frame_idx"], []).append(row)
        if list(groups) != list(range(FIRST, LAST + STRIDE, STRIDE)):
            raise ValueError("scope_coverage")
        for rows in groups.values():
            validate_scope(rows)
        if any(row["software_epoch"] is None for row in self.rows):
            raise ValueError("software_epoch_missing")
        return {"contract": contract(), "scope_count": len(groups), "row_count": len(self.rows),
            "runtime": self.runtime, "local_evaluation": evaluate(list(groups.values())),
            "reasons": dict(Counter(row["reason"] for row in self.rows if row["kind"] == PREFIX + "gate"))}


def validate_scope(rows: list[dict[str, Any]]) -> None:
    if rows[0]["kind"] != PREFIX + "enter" or rows[-1]["kind"] != PREFIX + "return":
        raise ValueError("scope_incomplete")
    gates = rows[1:-1]
    if any(row["kind"] != PREFIX + "gate" for row in gates):
        raise ValueError("scope_order")
    expected = {name: sum(row[name] for row in gates)
                for name in ("preview_calls", "live_transition_calls", "within_calls")}
    if rows[-1]["calls"] != expected or rows[-1]["gate_count"] != len(gates):
        raise ValueError("gate_coverage")
    if rows[-1]["shadow_error"] is not None:
        raise ValueError("shadow_failed")


def evaluate(groups: list[list[dict[str, Any]]]) -> dict[str, Any]:
    initial = groups[0][0]["snapshot"]["confirmed"]
    returned = [rows[-1] for rows in groups]
    accepted = [row for rows in groups for row in rows if row.get("reason") == "accepted"]
    partial = [row["frame_idx"] for row in returned if row["snapshot"]["state"] == "stable"
               and row["confirmed_return"] is not None and row["confirmed_return"]["color"] == PARTIAL_COUNT]
    completed = [row["frame_idx"] for row in returned if row["snapshot"]["state"] == "stable"
                 and expected_grid(row["confirmed_return"], COMPLETE_SHA)]
    reasons = []
    if not expected_grid(initial, BASELINE_SHA):
        reasons.append("initial_50_full_grid_not_observed")
    if not accepted or not completed:
        reasons.append("finite_completion_not_observed")
    if partial:
        reasons.append("partial_51_published")
    if returned[-1]["snapshot"]["state"] != "stable":
        reasons.append("window_end_unresolved")
    if accepted:
        first = next(row for row in returned if row["frame_idx"] == accepted[0]["frame_idx"])
        if not expected_grid(first["confirmed_return"], COMPLETE_SHA):
            reasons.append("first_accepted_final_grid_mismatch")
    return {"pass": not reasons, "failure_reasons": reasons, "initial": initial,
        "partial_stable_frames": partial, "completed_52_frames": completed,
        "accepted_frames": [row["frame_idx"] for row in accepted],
        "general_quality_pass": False, "probabilistic_board_measured": True}


def grid_present(value: Any) -> bool:
    cells = value.get("grid") if isinstance(value, dict) else None
    return (isinstance(cells, list) and len(cells) == shadow.ROWS and
        all(isinstance(row, list) and len(row) == shadow.COLS and
            all(type(cell) is int and cell in shadow.VALID for cell in row) for row in cells))


def expected_grid(value: Any, expected: str) -> bool:
    """評価専用。参照 SHA は SM / gate の入力へ流さない。"""
    return grid_present(value) and base.board_value(value["grid"])["sha256"] == expected


def scoped_step(original: Any, sink: Sink) -> Any:
    @functools.wraps(original)
    def step(pipe: Any, side: str, frame_idx: int, time_sec: float, *args: Any, **kwargs: Any) -> Any:
        if not selected(frame_idx, side):
            return original(pipe, side, frame_idx, time_sec, *args, **kwargs)
        def emit(kind: str, **fields: Any) -> None:
            sink.emit({"kind": PREFIX + kind, "side": side, "frame_idx": frame_idx,
                       "time_sec": time_sec, **fields})
        try:
            emit("enter", snapshot=snapshot(pipe, side))
            sm = getattr(pipe, "_sm_" + side.lower())
            if kwargs.get("sm") is not sm:
                raise RuntimeError("actual_side_sm_identity")
            with shadow.install(type(sm), lambda row: emit("gate", **row)) as controller:
                result = original(pipe, side, frame_idx, time_sec, *args, **kwargs)
            if controller.sticky_error is not None:
                raise RuntimeError("caught_shadow_error:" + controller.sticky_error)
            calls = {name: sum(row[name] for row in controller.records)
                     for name in ("preview_calls", "live_transition_calls", "within_calls")}
            emit("return", snapshot=snapshot(pipe, side), confirmed_return=base.board_value(result.confirmed_board),
                 inferred_return=base.board_value(result.inferred_board), calls=calls,
                 prob_return=serializer.json_value(SimpleNamespace(entry=entry), result.prob_board),
                 gate_count=len(controller.records), shadow_error=controller.sticky_error)
            return result
        except BaseException as exc:
            sink.failed(exc, "completion_step")
            raise
    return step


def close_sink(state: dict[str, Any]) -> None:
    sink = state["completion_sink"]
    try:
        sink.close()
    finally:
        base.write_json(state["output"] / STATUS, {"sticky_error": sink.error, "closed": sink.closed,
                        "installed": state.get("completion_installed", False)})


def instrument(stack: Any, collector: Any, rec: Any, receipt: Any, state: dict[str, Any]) -> None:
    OLD_INSTRUMENT(stack, collector, rec, receipt, state)
    sink = Sink(state["output"] / SIDECAR, rec)
    state["completion_sink"] = sink
    stack.callback(close_sink, state)
    cls, original = collector.RecognitionPipeline, collector.RecognitionPipeline._step_side
    path = Path(inspect.getfile(cls)).resolve()
    if receipt["input_and_code_sha256"].get(str(path)) != base.sha256(path):
        raise ValueError("actual_pipeline_not_guarded")
    base.patch(stack, cls, "_step_side", scoped_step(original, sink))
    state["completion_installed"] = True
    sink.runtime = {"pipeline_path": str(path), "pipeline_sha256": base.sha256(path),
        "outer_step_code": {"path": original.__code__.co_filename, "line": original.__code__.co_firstlineno},
        "shadow_sha256": B_SHA["shadow.py"], "saved_sm_injection": False,
        "original_step_body_unchanged": True, "background": OLD_FORMAT}


def projections(values: dict[str, Any], category: str) -> list[Any]:
    result = []
    for key, item in values.items():
        row = item["row"]
        if category == "raw" and row["kind"] == "frame_side":
            capture = row.get("accounting_capture")
            result.append((key, row.get("raw_captured_this_frame"), None if capture is None else
                           {name: capture.get(name) for name in ("captured_frame", "raw")}))
        elif category == "next" and row["kind"] == NEXT_KIND:
            result.append((key, row["raw"]))
        elif category == "2P" and row.get("side") in ("2P", "p2"):
            result.append((key, item["text"]))
    return result


def strict_invariants(reference: Path, candidate: Path) -> dict[str, Any]:
    old, new = prior.indexed(reference), prior.indexed(candidate)
    result = {}
    for category, required in (("raw", RAW_COUNT), ("next", NEXT_COUNT), ("2P", None)):
        left, right = projections(old, category), projections(new, category)
        count_ok = bool(left) and len(left) == len(right) and (required is None or len(left) == required)
        result[category] = {"equal": left == right, "count_ok": count_ok,
            "reference_rows": len(left), "candidate_rows": len(right), "required": required}
        result[category]["input_structure_ok"] = all(input_present(item["row"], category)
            for values in (old, new) for item in values.values())
    return result


def input_present(row: dict[str, Any], category: str) -> bool:
    if category == "raw" and row["kind"] == "frame_side":
        capture = row.get("accounting_capture")
        return (row.get("raw_captured_this_frame") is True and isinstance(capture, dict)
            and type(capture.get("captured_frame")) is int and capture["captured_frame"] == row["frame_idx"]
            and grid_present(capture.get("raw")))
    if category == "next" and row["kind"] == NEXT_KIND:
        raw = row.get("raw")
        if not isinstance(raw, dict) or not {"main", "slides", "same_return_identity", "capture_error", "extra_detector_calls"} <= raw.keys():
            return False
        main = raw["main"]
        return (isinstance(main, dict) and {"count", "returned", "exception"} <= main.keys()
            and type(main["count"]) is int and main["count"] in (0, 1)
            and isinstance(raw["slides"], dict) and {"1P", "2P"} <= raw["slides"].keys())
    return True


def finish(output: Path, receipt: Any, rec: Any, elapsed: float, state: dict[str, Any]) -> Any:
    report = state["completion_sink"].summary()
    comparisons = {name: prior.compare(REFERENCE / name, output / name, [{"frame_idx": FIRST}])
                   for name in ("frames.jsonl", entry.SIDECAR, prior.SIDECAR)}
    invariants = strict_invariants(REFERENCE / "frames.jsonl", output / "frames.jsonl")
    base.write_json(output / REPORT, report)
    base.write_json(output / DIFF, {"streams": comparisons, "invariants": invariants,
        "post_window_1p_differences_require_independent_review": True, "quality_gate_clear": False})
    if not all(item["pre_mutation_prefix_bit_exact"] for item in comparisons.values()):
        raise ValueError("pre_window_prefix_changed")
    if not all(item["equal"] and item["count_ok"] and item["input_structure_ok"] for item in invariants.values()):
        raise ValueError("noninterference_failed")
    if not report["local_evaluation"]["pass"]:
        raise ValueError("local_window_quality_failed")
    return OLD_FINISH(output, receipt, rec, elapsed, state)


@contextlib.contextmanager
def configured() -> Any:
    with contextlib.ExitStack() as stack:
        for name, value in {"FORMAT": FORMAT, "EXTRAS": OLD_EXTRAS | EXTRAS,
                "additional_guards": additional_guards, "prepare": prepare,
                "instrument": instrument, "finish": finish}.items():
            base.patch(stack, prior, name, value)
        yield


def run(args: Any) -> Any:
    with configured():
        return prior.run(args)


def finalize(output: Path, child_exit: int) -> Any:
    if type(child_exit) is not int or not 0 <= child_exit <= 255:
        raise ValueError("actual_numeric_child_exit_required")
    if child_exit == 0:
        status, report = base.read_json(output / STATUS), base.read_json(output / REPORT)
        if status.get("sticky_error") is not None or status.get("closed") is not True or status.get("installed") is not True:
            raise ValueError("completion_shadow_failed_or_missing")
        if report.get("local_evaluation", {}).get("pass") is not True:
            raise ValueError("local_window_quality_failed")
    with configured():
        return prior.finalize(output, child_exit)


def main() -> int:
    with configured(), prior.configured(), contextlib.ExitStack() as stack:
        base.patch(stack, entry, "run", run)
        base.patch(stack, entry, "finalize", finalize)
        return entry.main()


if __name__ == "__main__":
    raise SystemExit(main())
