"""元runのfinally完了後だけENGINEを一回保存する私有adapter。"""
from __future__ import annotations

import contextlib
import copy
import hashlib
from pathlib import Path
from types import CodeType, FunctionType
from typing import Any

ROOT = Path(__file__).resolve().parent
RUNTIME = ROOT.parent / "g2_split_runtime_evidence_adapter_2026-09-08_v1/runner.py"
RUNTIME_SHA = "78bdc5fdc58e2b2300ddff2215bc0276be4d810ce36bfa87aeff7505924fd93e"
STATUS = "CURRENT_ENTRY_STATUS.json"
FORBIDDEN = frozenset(("COMPLETE", "CHILD_EXIT.json", "SPLIT_ENGINE.json"))
OWN = ("adapter.py", "test_adapter.py", "run_cpu.py")
_INSTALLED: set[Any] = set()


def require(condition: bool, reason: str) -> None:
    if not condition:
        raise RuntimeError(reason)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def guards() -> dict[str, str]:
    require(sha(RUNTIME) == RUNTIME_SHA, "fixed_runtime_changed")
    return {str(RUNTIME): RUNTIME_SHA, **{str(ROOT / name): sha(ROOT / name) for name in OWN}}


def validate_runtime(runtime: Any) -> None:
    require(Path(runtime.__file__).resolve() == RUNTIME.resolve(), "runtime_path")
    guards()
    code = compile(RUNTIME.read_bytes(), str(runtime.__file__), "exec")
    for name in ("run", "write"):
        expected = next(item for item in code.co_consts if isinstance(item, CodeType) and item.co_name == name)
        actual = getattr(runtime, name)
        require(type(actual) is FunctionType and actual.__code__ == expected
            and actual.__globals__ is vars(runtime), "runtime_function_identity")
    require(type(runtime.REQUIRED) is frozenset and STATUS in runtime.REQUIRED, "runtime_required")


def validate_hashes(values: Any) -> None:
    require(type(values) is dict and bool(values), "engine_hashes")
    for name, digest in values.items():
        require(type(name) is str and Path(name).name == name and name not in FORBIDDEN
            and "/" not in name and "\\" not in name, "engine_artifact_name")
        require(type(digest) is str and len(digest) == 64
            and all(char in "0123456789abcdef" for char in digest), "engine_artifact_digest")


class State:
    """ENGINE候補だけを私有保持し、保存権は元writeへ戻す。"""
    def __init__(self, runtime: Any) -> None:
        self.runtime, self.original_write, self.original_run = runtime, runtime.write, runtime.run
        self.required = runtime.REQUIRED
        self.pending: dict[str, Any] | None = None
        self.output: Path | None = None
        self.attempted = self.active = self.closed = False
        self.failure: str | None = None

    def write(self, path: Path, value: Any) -> Any:
        if Path(path).name != self.runtime.ENGINE:
            return self.original_write(path, value)
        try:
            require(self.active and self.output is not None, "engine_outside_active_run")
            require(Path(path).resolve() == self.output / self.runtime.ENGINE, "engine_output_scope")
            require(self.pending is None, "engine_duplicate_write")
            require(type(value) is dict and value.get("format_version") == self.runtime.FORMAT
                and value.get("awaiting_actual_child_exit") is True
                and value.get("quality_gate_clear") is False, "engine_payload")
            validate_hashes(value.get("sha256"))
            self.pending = copy.deepcopy(value)
        except BaseException as error:
            self.failure = repr(error)
            raise
        return None

    def run(self, args: Any) -> Any:
        try:
            require(not self.attempted, "engine_run_repeated")
            self.attempted = True
            self.output = Path(args.output_root).resolve()
            require(not self.output.exists(), "exclusive_new_output_required")
            self.active = True
            returned = self.original_run(args)
            require(self.failure is None, "sticky_engine_failure")
            self.save_after_finally()
            return returned
        except BaseException as error:
            self.failure = repr(error)
            raise
        finally:
            self.active = False

    def save_after_finally(self) -> None:
        require(self.pending is not None and self.output is not None, "engine_not_requested")
        require(self.runtime.REQUIRED == self.required, "required_changed_during_run")
        hashes = self.pending["sha256"]
        self.runtime.M.base.assert_unchanged({str(self.output / name): digest for name, digest in hashes.items()})
        status = self.runtime.read(self.output / STATUS)
        require(type(status) is dict and status.get("closed") is True
            and "sticky_error" in status and status["sticky_error"] is None
            and type(status.get("row_count")) is int and status["row_count"] >= 0,
            "current_entry_status_failed")
        final_hashes = {**hashes, STATUS: sha(self.output / STATUS)}
        require(self.required <= final_hashes.keys(), "required_artifact_missing")
        files = {path.name for path in self.output.iterdir() if path.is_file()}
        require(files == final_hashes.keys(), "unexpected_post_finish_artifact")
        plan = self.runtime.read(self.output / "PLAN.json")
        inputs = plan["input_and_code_sha256"]
        require(all(inputs.get(path) == digest for path, digest in guards().items()), "adapter_prepare_guard_missing")
        self.runtime.M.base.assert_unchanged(inputs)
        self.runtime.M.base.assert_unchanged({str(self.output / n): h for n, h in final_hashes.items()})
        self.original_write(self.output / self.runtime.ENGINE, {**self.pending, "sha256": final_hashes})


def install(stack: contextlib.ExitStack, runtime: Any) -> State:
    """finish確定後、entryへruntime.runを束縛する前に一回設置する。"""
    require(runtime not in _INSTALLED, "engine_adapter_reentry")
    validate_runtime(runtime)
    state = State(runtime)
    _INSTALLED.add(runtime)
    def restore() -> None:
        runtime.write, runtime.run = state.original_write, state.original_run
        state.closed = True
        _INSTALLED.remove(runtime)
    stack.callback(restore)
    runtime.write, runtime.run = state.write, state.run
    return state
