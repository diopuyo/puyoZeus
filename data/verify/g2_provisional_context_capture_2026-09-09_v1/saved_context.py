"""完了した同runの実保存へ暫定binderを接続する。部分runの救済はしない。"""
from __future__ import annotations
import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any, Iterator
import binding as B

ROOT = Path(__file__).resolve().parent


def unique(pairs: list[Any]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        B.require(key not in result, "duplicate_json_key")
        result[key] = value
    return result


def decode(value: str) -> Any:
    return json.loads(value, object_pairs_hook=unique, parse_constant=lambda value: B.require(False, "nonfinite_json"))


def read(path: Path) -> Any:
    return decode(path.read_text(encoding="utf-8"))


def rows(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            yield decode(line)


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def observer() -> Any:
    spec = importlib.util.spec_from_file_location("_saved_provisional_context_observer", ROOT / "observer.py")
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def verify_complete(output: Path) -> tuple[dict[str, str], dict[str, Any]]:
    complete = read(output / "COMPLETE")
    B.require(type(complete.get("child_exit_code")) is int and complete["child_exit_code"] == 0, "real_child_exit")
    O = observer()
    required = O.REQUIRED | {"PLAN.json", "CHILD_EXIT.json", "frames.jsonl", "hidden_probability.jsonl",
                             "current_scope.jsonl", "provisional_current.jsonl"}
    index = complete.get("sha256")
    B.require(type(index) is dict and required <= set(index), "complete_required_missing")
    bound = {str(output / "COMPLETE"): sha(output / "COMPLETE")}
    for name, expected in index.items():
        path = (output / name).resolve()
        B.require(path.is_relative_to(output.resolve()) and sha(path) == expected, "complete_artifact_changed")
        bound[str(path)] = expected
    child = read(output / "CHILD_EXIT.json").get("child_exit_code")
    B.require(type(child) is int and child == 0, "child_record_failed")
    receipt = O.verify(output)
    for name, expected in receipt["guards"].items():
        B.require(sha(Path(name)) == expected, "context_source_changed")
        bound[name] = expected
    return bound, receipt


def registration(output: Path, receipt: dict[str, Any]) -> dict[str, Any]:
    plan = read(output / "PLAN.json")
    source = plan["original_source"]
    video_sha = plan["input_and_code_sha256"][plan["video_path"]]
    B.require(video_sha == source["source_video_sha256"], "plan_source_sha")
    B.require(receipt["source_id"] == "sha256:" + video_sha and receipt["run_id"] == output.resolve().as_posix(),
              "run_or_source_registration")
    return {"source_id": receipt["source_id"], "run_id": receipt["run_id"], "source_sha256": video_sha,
        "time_base_numerator": source["time_base_numerator"], "time_base_denominator": source["time_base_denominator"],
        "ledger_connection": "NOT_CONNECTED", "context_receipt_sha256": sha(output / "PROVISIONAL_CONTEXT_RECEIPT.json"),
        "plan_sha256": sha(output / "PLAN.json"), "complete_sha256": sha(output / "COMPLETE")}


def indices(output: Path) -> dict[str, dict[Any, Any]]:
    result = {}
    specs = (("pb", "hidden_probability.jsonl", None),
             ("sm", "current_scope.jsonl", "current_entry_sm_return"),
             ("next", "frames.jsonl", "next_enqueue_live_decision"),
             ("candidate_row", "provisional_current.jsonl", None))
    for key, filename, kind in specs:
        index = {}
        for row in rows(output / filename):
            if kind is not None and row.get("kind") != kind:
                continue
            clock = row.get("frame_idx", row.get("frame")), row.get("side")
            B.require(clock not in index, "duplicate_saved_evidence")
            index[clock] = row
        result[key] = index
    return result


def saved_join(row: dict[str, Any], index: dict[str, dict[Any, Any]]) -> None:
    for side in B.SIDES:
        clock = row["frame_idx"], side
        for key in ("pb", "sm", "next", "candidate_row"):
            B.require(B.exact(row["sides"][side].get(key), index[key].get(clock)), "same_run_join:" + key)
