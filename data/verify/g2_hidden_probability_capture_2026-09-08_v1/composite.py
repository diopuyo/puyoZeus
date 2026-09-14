"""旧W・修復保存入口とPB観測を合成する。元runtimeは再製造しない。"""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys
from types import SimpleNamespace
from typing import Any

ROOT = Path(__file__).resolve().parent
CONNECTION = ROOT.parent / "g2_receipt_driver_connection_2026-09-08_v1/connection.py"
CONNECTION_SHA = "96e176d9ee93e1cbd0153c23d423cc741a6f0c0211e9e006d3835f5340b7c7d8"
HEX_SHA = re.compile(r"[0-9a-f]{64}\Z")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate_json_key")
        result[key] = value
    return result


def reject_constant(value: str) -> None:
    raise ValueError("nonfinite_json")


def read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique_object,
                      parse_constant=reject_constant)


def source_ids(state: dict[str, Any]) -> tuple[str, str]:
    output = state["output"].resolve()
    plan = read(output / "PLAN.json")
    video = plan.get("video_path")
    if type(video) is not str or not Path(video).is_absolute():
        raise ValueError("plan_video_absolute_required")
    digest = plan["input_and_code_sha256"].get(video)
    if type(digest) is not str or not HEX_SHA.fullmatch(digest):
        raise ValueError("plan_video_sha_required")
    # 実動画のSHA照合自体は既存prepare/runtimeが行う。この関数で認証を増やさない。
    return "sha256:" + digest, output.as_posix()


def merge_guards(*groups: dict[str, str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for group in groups:
        for path, digest in group.items():
            if path in result and result[path] != digest:
                raise ValueError("composite_guard_conflict")
            result[path] = digest
    return result


def combine(old: Any, new: Any) -> Any:
    if old.REQUIRED & new.REQUIRED:
        raise ValueError("observer_artifact_collision")
    def guards() -> dict[str, str]:
        return merge_guards(old.guards(), new.guards(), {str(Path(__file__)): sha(Path(__file__))})
    def install(stack: Any, collector: Any, history: Any, state: dict[str, Any]) -> None:
        source, run = source_ids(state)
        old.install(stack, collector, history, state)
        new.install(stack, collector, history, state, source_id=source, run_id=run)
    def finish(state: dict[str, Any]) -> None:
        old.finish(state)
        new.finish(state)
    def verify(output: Path) -> None:
        old.verify(output)
        new.verify(output)
    return SimpleNamespace(REQUIRED=old.REQUIRED | new.REQUIRED, guards=guards,
                           install=install, finish=finish, verify=verify)


def load_connection() -> Any:
    if sys.flags.optimize:
        raise RuntimeError("optimized_python_not_supported")
    alias = "_hidden_capture_receipt_connection"
    if alias in sys.modules or sha(CONNECTION) != CONNECTION_SHA:
        raise RuntimeError("connection_source_or_alias_changed")
    spec = importlib.util.spec_from_file_location(alias, CONNECTION)
    module = importlib.util.module_from_spec(spec)
    sys.modules[alias] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(alias, None)
        raise
    return module
