"""既存C6/W/PB実入口へ公開holdと別current streamを追加する限定CLI。"""
from __future__ import annotations
from contextlib import contextmanager
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any, Iterator

import current_connection as K

ROOT = Path(__file__).resolve().parent
OLD = ROOT.parent / "g2_hidden_probability_capture_2026-09-08_v1"
sys.path.append(str(OLD))
FINISHER = ROOT.parent / "g2_finisher_torch_version_2026-09-08_v1/adapter.py"
FINISHER_SHA = "cb20535bfc3b48b3e9f9caee12e03d0e0106adb7e6e5f5e80cb62a79612e6daf"
OWN = ("live_cli.py", "current_connection.py", "current_policy.py", "runtime_policy.py", "adapter.py", "launcher.sh", "preflight.py")
STATUS, RECEIPT = "PROVISIONAL_CURRENT_STATUS.json", "PROVISIONAL_CURRENT_RECEIPT.json"
REQUIRED = {K.SIDECAR, STATUS, RECEIPT}


def write(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def guards() -> dict[str, str]:
    if K.A.sha(FINISHER) != FINISHER_SHA:
        raise RuntimeError("fixed_torch_finisher_changed")
    return K.A.guards() | {str(path): K.A.sha(path) for path in
        (*(ROOT / name for name in OWN), *K.POLICY_FILES, FINISHER)}


def finish(state: dict[str, Any]) -> None:
    connection, hold = state["provisional_current_connection"], state["generation_publication_hold"]
    status = {"closed": connection.closed, "failures": connection.failures, "hold": hold.report()}
    if not connection.closed or connection.failures or not hold.closed or status["hold"]["failures"]:
        raise RuntimeError("provisional_current_capture_failed")
    expected = [(row["frame_idx"], row["side"]) for row in state["hidden_probability_observer"].rows]
    actual = [(row["frame"], row["side"]) for row in connection.rows]
    if not expected or expected != actual:
        raise RuntimeError("provisional_current_coverage")
    write(state["output"] / STATUS, status)
    write(state["output"] / RECEIPT, {"expected_scopes": expected,
        "candidate_count": sum(row["current_candidate"] is not None for row in connection.rows),
        "sha256": {name: K.A.sha(state["output"] / name) for name in (K.SIDECAR, STATUS)},
        "quality_gate_clear": False, "production_permission": False})


def verify(output: Path) -> None:
    receipt = json.loads((output / RECEIPT).read_text())
    status = json.loads((output / STATUS).read_text())
    if status["closed"] is not True or status["failures"] or status["hold"]["failures"]:
        raise RuntimeError("saved_current_failed")
    if set(receipt["sha256"]) != {K.SIDECAR, STATUS}:
        raise RuntimeError("saved_current_index")
    if any(K.A.sha(output / name) != digest for name, digest in receipt["sha256"].items()):
        raise RuntimeError("saved_current_changed")
    rows = [json.loads(line) for line in (output / K.SIDECAR).read_text().splitlines()]
    if [[row["frame"], row["side"]] for row in rows] != receipt["expected_scopes"]:
        raise RuntimeError("saved_current_coverage")


def bootstrap() -> tuple[Any, Any, Any]:
    spec = importlib.util.spec_from_file_location("_generation_previous_live_cli", OLD / "live_cli.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    prior_connection, driver, old = module.bootstrap()
    def install(stack: Any, collector: Any, history: Any, state: Any) -> None:
        old.install(stack, collector, history, state)
        K.install(stack, collector, history, state)
    def combined_finish(state: Any) -> None:
        old.finish(state)
        finish(state)
    def combined_verify(output: Path) -> None:
        old.verify(output)
        verify(output)
    addon = SimpleNamespace(REQUIRED=old.REQUIRED | REQUIRED, install=install,
        finish=combined_finish, verify=combined_verify, guards=lambda: module.C.merge_guards(old.guards(), guards()))
    return prior_connection, driver, addon


@contextmanager
def configuration(prior: Any, driver: Any) -> Iterator[None]:
    with prior.configuration(driver):
        old_path, old_sha = driver.FINISHER, driver.FINISHER_SHA
        driver.FINISHER, driver.FINISHER_SHA = FINISHER, FINISHER_SHA
        try:
            yield
        finally:
            driver.FINISHER, driver.FINISHER_SHA = old_path, old_sha


def main() -> int:
    prior, driver, addon = bootstrap()
    with configuration(prior, driver):
        return driver.live_main(addon)


if __name__ == "__main__":
    raise SystemExit(main())
