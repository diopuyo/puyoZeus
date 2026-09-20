"""同updateの候補・公開値・hard fieldを別sinkへ非変更で捕捉する。"""
from __future__ import annotations

import dataclasses
import functools
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parent
PB_SOURCE = ROOT.parent / "g2_hidden_probability_capture_2026-09-08_v1/observer.py"
SCOPE_SOURCE = ROOT.parent / "g2_current_scope_capture_2026-09-09_v1/observer.py"
CONNECTION = ROOT.parent / "g2_generation_publication_split_2026-09-09_v1/current_connection.py"
FIXED = {str(PB_SOURCE): "cea8f92da47c19210a3c9a36dc877bfddd07d102f57575caecc766c8ee2692e9",
    str(SCOPE_SOURCE): "f38224b5d13a8f45b28fa8ed7f74a6432ad8555aa4ab5d373ecd7ba6993b4d56",
    str(CONNECTION): "af17f8ec4901e0a718feca9b1c739f5688c1e69328eb0ecc3e1fbf42f5e23dbc"}
SIDECAR, STATUS, RECEIPT = "provisional_context.jsonl", "PROVISIONAL_CONTEXT_STATUS.json", "PROVISIONAL_CONTEXT_RECEIPT.json"
REQUIRED = frozenset((SIDECAR, STATUS, RECEIPT))
SCHEMA = "provisional-current-context/v1"
SIDES, WINDOWS, STRIDE = ("1P", "2P"), ((32640, 32800), (34700, 36000)), 2
OUTER_FIRST, OUTER_LAST = 29052, 36298
FIELDS = ("pending_garbage", "effective_rate", "chain_active", "provisional_generated", "provisional_score", "provisional_chain_count")
STATE_KEY = "provisional_context_observer"


def require(condition: bool, reason: str) -> None:
    if not condition:
        raise ValueError(reason)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def copied(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))


def write(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False, indent=2)


def guards() -> dict[str, str]:
    require(all(sha(Path(path)) == value for path, value in FIXED.items()), "context_fixed_source_changed")
    return FIXED | {str(ROOT / name): sha(ROOT / name) for name in
        ("observer.py", "test_observer.py", "run_observer_cpu.py", "OBSERVER_CONTRACT.md")}


def expected() -> list[int]:
    return list(range(OUTER_FIRST, OUTER_LAST + 1, STRIDE))


def slot(value: Any = None, *, observed: bool = False, reason: str = "producer_not_connected") -> dict[str, Any]:
    return {"observed": observed, "value": copied(value), "reason": reason}


def disconnected() -> dict[str, Any]:
    return {"connection": "NOT_CONNECTED", "reason": "producer_not_connected", "sides": {
        side: {name: {"value": None, "availability": "UNKNOWN", "reason": "producer_not_connected"}
               for name in FIELDS} for side in SIDES}}


def probability_equal(left: Any, right: Any) -> bool | None:
    if not left or not right or not left.get("present") or not right.get("present"):
        return None
    return not left["errors"] and not right["errors"] and left["cells"] == right["cells"]


def clock(frame: Any, time_sec: Any) -> None:
    require(type(frame) is int and frame >= 0, "context_frame_type")
    require(type(time_sec) in (int, float) and math.isfinite(time_sec) and time_sec >= 0, "context_clock_type")


def fixed_module(value: Any, path: Path) -> Any:
    module = sys.modules.get(type(value).__module__)
    require(module is not None and Path(module.__file__).resolve() == path, "context_runtime_module")
    return module


class Recorder:
    """私有snapshotのみ更新し、元pipeline/ledgerへは書き戻さない。"""
    def __init__(self, history: Any, state: dict[str, Any], serializer: Any,
                 expected_frames: list[int] | None = None) -> None:
        self.history, self.state, self.serializer = history, state, serializer
        self.pb, self.sm = state["hidden_probability_observer"], state["current_scope_sink"]
        self.connection, self.tracker = state["provisional_current_connection"], state["tracker"]
        self.source_id, self.run_id = self.pb.source_id, self.pb.run_id
        require(all(type(v) is str and v.strip() for v in (self.source_id, self.run_id)), "context_source_run")
        scopes = list(self.pb.expected)
        pb_frames = list(dict.fromkeys(frame for frame, _ in scopes))
        require(scopes == [(frame, side) for frame in pb_frames for side in SIDES]
                and scopes == list(self.sm.expected), "context_parent_scope_mismatch")
        self.expected = expected() if expected_frames is None else list(expected_frames)
        require(self.expected and all(type(f) is int and f >= 0 for f in self.expected)
                and self.expected == sorted(set(self.expected)), "context_expected_schema")
        self.pb_expected = scopes
        self.output = Path(state["output"])
        self.rows: list[dict[str, Any]] = []
        self.errors: list[dict[str, Any]] = []
        self.closed, self.active, self.pipe = False, None, None
        self.last_clock: tuple[int, float] | None = None
        self.installed = False
        self.stream = (self.output / SIDECAR).open("x", encoding="utf-8")

    def fail(self, stage: str, error: BaseException) -> None:
        value = {"stage": stage, "type": type(error).__name__, "message": str(error)}
        self.errors.append(value)
        if self.active is not None:
            self.active["row"]["failures"].append(copied(value))

    def probability(self, value: Any) -> dict[str, Any]:
        return self.serializer.probability_value(value, self.pb.pb_type)

    def side_value(self, result: Any) -> dict[str, Any] | None:
        if result is None:
            return None
        scalar = self.serializer.scalar
        value = {"state": self.serializer.state_name(result.state), "state_value": scalar(result.state)}
        for name in ("confirmed", "cnn", "inferred"):
            value[name] = self.pb.board_value(getattr(result, name + "_board", None))
        value["probability"] = self.probability(getattr(result, "prob_board", None))
        for name in ("next_pair", "dnext_pair", "board_provenance", "board_none_reason"):
            value[name] = scalar(getattr(result, name, None))
        return copied(value)

    def generation(self) -> dict[str, Any]:
        values = {side: dataclasses.asdict(self.tracker.generation(side)) for side in SIDES}
        require(all(values[side]["side"] == side for side in SIDES), "context_generation_side")
        return slot(values, observed=True, reason="actual_software_generation_not_physical_game")

    def begin(self, pipe: Any, frame: int, time_sec: float) -> None:
        clock(frame, time_sec)
        require(self.active is None and not self.closed and not self.errors, "context_reentry_or_failed")
        require(self.tracker._pipeline is pipe and pipe is not None, "context_unbound_or_other_pipeline")
        require(self.pipe is None or self.pipe is pipe, "context_pipeline_changed")
        require(self.last_clock is None or (frame > self.last_clock[0] and time_sec >= self.last_clock[1]), "context_clock_reversed_or_duplicate")
        self.pipe, self.last_clock = pipe, (frame, time_sec)
        index = len(self.rows)
        row = {"schema_version": SCHEMA, "source_id": self.source_id, "run_id": self.run_id,
            "frame_idx": frame, "time_sec": time_sec, "capture_token": f"update:{index}",
            "available_frame": None, "update": {"frame_idx": frame, "time_sec": time_sec,
                "returned_frame_idx": None, "returned_time_sec": None, "call_index": index,
                "pipe_index": 0, "returned": False, "exception": None}, "sides": {}, "failures": [],
            "game": slot(reason="no_game_identity_on_this_pipeline_boundary"),
            "generation": {"before": self.generation(), "after": slot(reason="update_not_returned")},
            "canonical": {"connection": "NOT_CONNECTED", "value": None, "reason": "producer_not_connected"},
            "ledger": disconnected(), "current_publication_allowed": False,
            "accounting_permission": False, "quality_gate_clear": False}
        row["candidate_scope_selected"] = (frame, SIDES[0]) in self.pb_expected
        starts = {"pb": len(self.pb.rows), "sm": len(self.sm.rows), "candidate": len(self.connection.rows)}
        self.active = {"row": row, "objects": {}, "starts": starts}

    def isolate_before(self, side: str, result: Any) -> None:
        require(side in SIDES and side not in self.active["objects"], "context_duplicate_or_unknown_isolate")
        require(getattr(result, "side", None) == side, "context_isolate_side")
        self.active["objects"][side] = {"before": result, "before_pb": getattr(result, "prob_board", None)}
        self.active["row"]["sides"][side] = {"before_hold": self.side_value(result), "after_hold": None,
            "final": None, "pb": None, "sm": None, "next": None, "candidate_row": None,
            "identity": {}, "hold_reasons": []}

    def isolate_after(self, side: str, result: Any) -> None:
        objects = self.active["objects"][side]
        objects.update(after=result, after_pb=getattr(result, "prob_board", None))
        self.active["row"]["sides"][side]["after_hold"] = self.side_value(result)

    def one_row(self, rows: list[dict[str, Any]], side: str, kind: str | None = None) -> Any:
        frame, time_sec = self.active["row"]["frame_idx"], self.active["row"]["time_sec"]
        chosen = [r for r in rows if r.get("side") == side and r.get("frame_idx", r.get("frame")) == frame
                  and (kind is None or r.get("kind") == kind)]
        require(len(chosen) <= 1, "context_duplicate_evidence")
        if not chosen:
            return None
        row = chosen[0]
        require("time_sec" not in row or row["time_sec"] == time_sec, "context_evidence_clock")
        require(all(key not in row or row[key] == value for key, value in
                    (("source_id", self.source_id), ("run_id", self.run_id))), "context_evidence_identity")
        return copied(row)

    def evidence(self, side: str) -> dict[str, Any]:
        starts = self.active["starts"]
        frame = self.active["row"]["frame_idx"]
        next_row = self.connection.next_rows.get((frame, side))
        return {"pb": self.one_row(self.pb.rows[starts["pb"]:], side),
            "sm": self.one_row(self.sm.rows[starts["sm"]:], side, "current_entry_sm_return"),
            "candidate_row": self.one_row(self.connection.rows[starts["candidate"]:], side),
            "next": self.one_row([] if next_row is None else [next_row], side)}

    def finish_side(self, side: str, result: Any) -> None:
        require(side in self.active["objects"], "context_isolate_missing")
        require(result is not None and getattr(result, "side", None) == side, "context_final_side")
        data, objects = self.active["row"]["sides"][side], self.active["objects"][side]
        require("after" in objects, "context_isolate_return_missing")
        data.update(final=self.side_value(result), **self.evidence(side))
        pb = None if data["pb"] is None else data["pb"].get("probability")
        before, after, final = [data[name]["probability"] for name in ("before_hold", "after_hold", "final")]
        data["identity"] = {"isolate_return_is_final": objects["after"] is result,
            "before_probability_matches_pb": probability_equal(before, pb),
            "before_probability_unchanged": self.unchanged(objects, "before", before),
            "after_probability_unchanged": self.unchanged(objects, "after", after),
            "final_probability_matches_after": final == after if objects["after_pb"] is None
                and getattr(result, "prob_board", None) is None else probability_equal(final, after),
            "before_pb_is_after_pb": None if objects["before_pb"] is None else objects["before_pb"] is objects["after_pb"],
            "after_pb_is_final_pb": None if objects["after_pb"] is None else objects["after_pb"] is getattr(result, "prob_board", None)}
        data["hold_reasons"] = side_holds(data)

    def unchanged(self, objects: dict[str, Any], stage: str, stored: dict[str, Any]) -> bool | None:
        original = objects[stage + "_pb"]
        current = getattr(objects[stage], "prob_board", None)
        if current is not original:
            return False
        now = self.probability(current)
        return stored == now if original is None else probability_equal(stored, now)

    def returned(self, result: Any) -> None:
        row, pipe = self.active["row"], self.pipe
        update = row["update"]
        frame, time_sec = getattr(result, "frame_idx", None), getattr(result, "time_sec", None)
        update.update(returned=True, returned_frame_idx=frame, returned_time_sec=time_sec)
        clock(frame, time_sec)
        require((frame, time_sec) == (row["frame_idx"], row["time_sec"]), "context_return_clock")
        require((self.history.frame, self.history.time_sec) == (frame, time_sec)
                and (self.tracker._frame, self.tracker._time) == (frame, time_sec), "context_installed_clock")
        row["available_frame"] = frame
        for field, obj, attr in (("is_match_active", result, "is_match_active"),
                ("match_end_locked", result, "match_end_locked"),
                ("post_match_lockdown_active", pipe, "_post_match_lockdown_active")):
            update[field + "_observed"] = hasattr(obj, attr)
            update[field] = self.serializer.scalar(getattr(obj, attr, None))
        for side in SIDES:
            self.finish_side(side, getattr(result, "p1" if side == "1P" else "p2", None))
        row["generation"]["after"] = self.generation()
        failures = {key: copied(getattr(self.state[key], "failures", [])) for key in
                    ("hidden_probability_observer", "current_scope_sink", "provisional_current_connection")}
        row["upstream_failures"] = failures
        require(not any(failures.values()), "context_upstream_instrumentation_failed")
        row["hold_reasons"] = common_holds(row)

    def end(self, error: BaseException | None) -> None:
        if self.active is None:
            return
        row = self.active["row"]
        if error is not None:
            row["update"]["exception"] = {"type": type(error).__name__, "message": str(error)}
        row["capture_status"] = "FAILED" if row["failures"] or error is not None else "CAPTURED"
        encoded = json.dumps(row, ensure_ascii=False, allow_nan=False)
        self.stream.write(encoded + "\n")
        self.stream.flush()
        self.rows.append(json.loads(encoded))
        self.active = None

    def close(self) -> None:
        if self.active is not None:
            self.fail("close", RuntimeError("unfinished_update"))
        self.stream.close()
        self.closed = True
        write(self.output / STATUS, {"closed": self.closed, "installed": self.installed, "errors": self.errors})


def side_holds(data: dict[str, Any]) -> list[str]:
    reasons = [name + "_missing" for name in ("pb", "sm", "next", "candidate_row") if data[name] is None]
    if data["before_hold"]["state"] != "STABLE":
        reasons.append("non_stable")
    reasons += [name for name, value in data["identity"].items() if value is False and name != "before_pb_is_after_pb"]
    if data["before_hold"]["probability"]["errors"]:
        reasons.append("before_probability_invalid_or_missing")
    return reasons


def common_holds(row: dict[str, Any]) -> list[str]:
    update, reasons = row["update"], []
    for key, expected_value in (("is_match_active", True), ("match_end_locked", False), ("post_match_lockdown_active", False)):
        if update.get(key + "_observed") is not True or update.get(key) is not expected_value:
            reasons.append(key + "_not_ready")
    return reasons


def update_wrapper(original: Any, rec: Recorder) -> Any:
    @functools.wraps(original)
    def wrapped(pipe: Any, frame_idx: int, time_sec: float, *args: Any, **kwargs: Any) -> Any:
        if frame_idx not in rec.expected:
            return original(pipe, frame_idx, time_sec, *args, **kwargs)
        try:
            rec.begin(pipe, frame_idx, time_sec)
        except BaseException as error:
            rec.fail("begin", error)
            raise
        error = None
        try:
            result = original(pipe, frame_idx, time_sec, *args, **kwargs)
            try:
                rec.returned(result)
            except BaseException as capture_error:
                rec.fail("returned", capture_error)
            return result
        except BaseException as original_error:
            error = original_error
            rec.fail("update_exception", original_error)
            raise
        finally:
            try:
                rec.end(error)
            except BaseException as capture_error:
                rec.fail("end", capture_error)
                rec.active = None
    return wrapped


def isolate_wrapper(original: Any, rec: Recorder) -> Any:
    @functools.wraps(original)
    def wrapped(side: str, value: Any) -> Any:
        if rec.active is None:
            return original(side, value)
        started = False
        try:
            rec.isolate_before(side, value)
            started = True
        except BaseException as error:
            rec.fail("isolate_before", error)
        try:
            result = original(side, value)
        except BaseException as error:
            rec.fail("isolate_exception", error)
            raise
        if started:
            try:
                rec.isolate_after(side, result)
            except BaseException as error:
                rec.fail("isolate_after", error)
        return result
    return wrapped


def install(stack: Any, collector: Any, history: Any, state: dict[str, Any], *,
            expected_frames: list[int] | None = None) -> Recorder:
    guards()
    require(STATE_KEY not in state, "context_install_reentry")
    pb = state["hidden_probability_observer"]
    serializer = fixed_module(pb, PB_SOURCE)
    fixed_module(state["current_scope_sink"], SCOPE_SOURCE)
    fixed_module(state["provisional_current_connection"], CONNECTION)
    rec = Recorder(history, state, serializer, expected_frames)
    pending, cls = state["pending_rec"], collector.RecognitionPipeline
    old_update, old_isolate = cls.update, pending.isolate_side_result
    had_own, own = "isolate_side_result" in vars(pending), vars(pending).get("isolate_side_result")
    stack.callback(rec.close)
    stack.callback(setattr, cls, "update", old_update)
    if had_own:
        stack.callback(setattr, pending, "isolate_side_result", own)
    else:
        stack.callback(delattr, pending, "isolate_side_result")
    cls.update, pending.isolate_side_result = update_wrapper(old_update, rec), isolate_wrapper(old_isolate, rec)
    rec.installed = True
    state[STATE_KEY] = rec
    return rec


def coverage(rows: list[dict[str, Any]], frames: list[int]) -> dict[str, Any]:
    require(rows and [row["frame_idx"] for row in rows] == frames, "context_scope_coverage")
    for index, row in enumerate(rows):
        update = row["update"]
        clock(row["frame_idx"], row["time_sec"])
        clock(update["returned_frame_idx"], update["returned_time_sec"])
        require(row["schema_version"] == SCHEMA and row["capture_status"] == "CAPTURED" and not row["failures"], "context_failed_row")
        require(type(update["call_index"]) is int and update["call_index"] == index
                and type(update["pipe_index"]) is int and update["pipe_index"] == 0, "context_call_identity")
        require(row["capture_token"] == f"update:{index}" and update["returned"] is True
                and update["exception"] is None, "context_update_incomplete")
        require(row["available_frame"] == row["frame_idx"] == update["returned_frame_idx"]
                and row["time_sec"] == update["returned_time_sec"], "context_saved_clock")
        require(set(row["sides"]) == set(SIDES) and all(row["sides"][s]["after_hold"] is not None for s in SIDES), "context_side_coverage")
        require(row["ledger"] == disconnected() and row["canonical"] ==
            {"connection": "NOT_CONNECTED", "value": None, "reason": "producer_not_connected"}, "context_connection_registration")
    return {"update_count": len(rows), "side_count": len(rows) * len(SIDES),
        "candidate_window_updates": sum(row["candidate_scope_selected"] is True for row in rows),
        "outside_candidate_window_updates": sum(row["candidate_scope_selected"] is False for row in rows),
        "held_updates": sum(bool(row["hold_reasons"] or any(s["hold_reasons"] for s in row["sides"].values())) for row in rows)}


def finish(state: dict[str, Any]) -> dict[str, Any]:
    rec = state[STATE_KEY]
    require(rec.closed and rec.installed and not rec.errors and rec.active is None, "context_lifetime_or_error")
    result = coverage(rec.rows, rec.expected)
    result.update(source_id=rec.source_id, run_id=rec.run_id, schema_version=SCHEMA,
        expected_frames=rec.expected, candidate_expected_scopes=rec.pb_expected, guards=guards(),
        sha256={name: sha(rec.output / name) for name in (SIDECAR, STATUS)},
        current_publication_allowed=False, accounting_permission=False, quality_gate_clear=False)
    write(rec.output / RECEIPT, result)
    return result


def verify(output: Path, *, expected_frames: list[int] | None = None) -> dict[str, Any]:
    status = json.loads((output / STATUS).read_text())
    receipt = json.loads((output / RECEIPT).read_text())
    require(status == {"closed": True, "installed": True, "errors": []}, "context_saved_status")
    require(set(receipt["sha256"]) == {SIDECAR, STATUS}, "context_artifact_set")
    require(all(sha(output / name) == digest for name, digest in receipt["sha256"].items()), "context_artifact_changed")
    require(receipt["guards"] == guards(), "context_saved_guards")
    frames = expected() if expected_frames is None else expected_frames
    require(receipt["expected_frames"] == frames, "context_saved_expected_scope")
    rows = [json.loads(line) for line in (output / SIDECAR).read_text().splitlines()]
    require(all(row["source_id"] == receipt["source_id"] and row["run_id"] == receipt["run_id"] for row in rows), "context_saved_provenance")
    result = coverage(rows, frames)
    require(all(receipt.get(key) == value for key, value in result.items()), "context_saved_summary")
    require(all(receipt[key] is False for key in ("current_publication_allowed", "accounting_permission", "quality_gate_clear")), "context_permission")
    return receipt
