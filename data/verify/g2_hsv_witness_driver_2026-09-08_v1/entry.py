"""凍結runtimeと診断finisherへ、新観測だけを追加接続する入口。"""
from __future__ import annotations

import contextlib
import hashlib
import importlib.util
from pathlib import Path
import sys
from typing import Any, Iterator

ROOT = Path(__file__).resolve().parent
RUNTIME = ROOT.parent / "g2_split_runtime_evidence_adapter_2026-09-08_v1/runner.py"
FINISHER = ROOT.parent / "g2_split_diagnostic_finisher_2026-09-08_v1/finisher.py"
RUNTIME_SHA = "78bdc5fdc58e2b2300ddff2215bc0276be4d810ce36bfa87aeff7505924fd93e"
FINISHER_SHA = "96541c1e7df8b555e26bd5cca742d16060ba46f06c925e675adaacb5decc0702"
ADAPTER_SHA = "f24bd02f9bbf36c5a32028ede54ce2cc736e07657b1210ec4ee65279863a60d1"
OWN_FILES = ("entry.py", "test_entry.py", "run_cpu.py", "PLAN.md")


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_fixed(name: str, path: Path, digest: str) -> Any:
    if name in sys.modules or sha(path) != digest:
        raise RuntimeError("unexpected_module_or_source")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def load_runtime() -> tuple[Any, Any]:
    """既存live_mainと同じmodule/bindを使う。factory/collectは起動しない。"""
    runtime = load_fixed("_hsv_driver_split_runtime", RUNTIME, RUNTIME_SHA)
    finisher = load_fixed("_hsv_driver_finisher", FINISHER, FINISHER_SHA)
    runtime.M = load_fixed("_hsv_driver_evidence", RUNTIME.parent / "adapter.py", ADAPTER_SHA)
    latest = load_fixed("_hsv_driver_latest", runtime.LATEST / "adapter.py", runtime.OLD_SHA[0])
    runtime.M.bind(latest)
    runtime.OLD_ADDITIONAL = latest.additional_guards
    return runtime, finisher


def own_guards() -> dict[str, str]:
    return {str(ROOT / name): sha(ROOT / name) for name in OWN_FILES}


def validate_addon(addon: Any) -> None:
    """欠測observerをno-opとして走らせない。"""
    if addon is None or not all(callable(getattr(addon, name, None)) for name in
                               ("guards", "install", "finish", "verify")):
        raise ValueError("observer_not_connected")
    names = getattr(addon, "REQUIRED", None)
    if type(names) is not frozenset or not names:
        raise ValueError("observer_artifacts_required")
    if any(type(name) is not str or Path(name).name != name or name in
           ("COMPLETE", "CHILD_EXIT.json", "PLAN.json", "SPLIT_ENGINE.json") for name in names):
        raise ValueError("invalid_observer_artifact")


def guard_extension(runtime: Any, finisher: Any, addon: Any, original: Any) -> Any:
    def extra(args: Any) -> dict[str, str]:
        result = dict(original(args))
        for extension in (finisher.guards(), own_guards(), addon.guards()):
            if not extension:
                raise ValueError("empty_additional_guard")
            for path, digest in extension.items():
                if path in result and result[path] != digest:
                    raise ValueError("conflicting_additional_guard")
                result[path] = digest
        runtime.M.base.assert_unchanged(result)
        return result
    return extra


def install_extension(original: Any, addon: Any) -> Any:
    def instrument(stack: Any, collector: Any, history: Any, receipt: Any, state: Any) -> None:
        original(stack, collector, history, receipt, state)
        addon.install(stack, collector, history, state)
    return instrument


def finish_extension(original: Any, addon: Any) -> Any:
    def finish(output: Path, receipt: Any, rec: Any, elapsed: float, state: Any) -> Any:
        addon.finish(state)
        addon.verify(output)
        return original(output, receipt, rec, elapsed, state)
    return finish


def finalize_extension(runtime: Any, original: Any, addon: Any) -> Any:
    def finalize(output: Path, code: int) -> Any:
        if type(code) is not int or not 0 <= code <= 255:
            raise ValueError("actual_numeric_child_exit_required")
        if code == 0:
            if (output / "COMPLETE").exists() or (output / "CHILD_EXIT.json").exists():
                raise FileExistsError("exclusive_finalization")
            try:
                addon.verify(output)
            except BaseException:
                runtime.write(output / "CHILD_EXIT.json", {"child_exit_code": code,
                    "source": "actual_wait", "observer_verification_failed": True})
                raise
        return original(output, code)
    return finalize


@contextlib.contextmanager
def configured(runtime: Any, finisher: Any, addon: Any) -> Iterator[None]:
    """旧guard/必須artifactに追加し、例外時も全属性identityを復元する。"""
    validate_addon(addon)
    with contextlib.ExitStack() as stack:
        patch = runtime.M.base.patch
        patch(stack, runtime, "extra", guard_extension(runtime, finisher, addon, runtime.extra))
        patch(stack, runtime.M, "instrument", install_extension(runtime.M.instrument, addon))
        patch(stack, runtime, "finish", finish_extension(runtime.finish, addon))
        patch(stack, runtime, "finalize", finalize_extension(runtime, runtime.finalize, addon))
        patch(stack, runtime, "REQUIRED", runtime.REQUIRED | addon.REQUIRED)
        finisher.install(stack, runtime)
        yield


def live_main(addon: Any) -> int:
    """CPU結合検収後だけ呼ぶ。新observerなしのGPU入口を設けない。"""
    validate_addon(addon)
    runtime, finisher = load_runtime()
    latest = runtime.M.A
    with configured(runtime, finisher, addon), runtime.configured(), latest.configured(), \
            latest.prior.configured(), contextlib.ExitStack() as stack:
        runtime.M.base.patch(stack, latest.entry, "run", runtime.run)
        runtime.M.base.patch(stack, latest.entry, "finalize", runtime.finalize)
        return latest.entry.main()
