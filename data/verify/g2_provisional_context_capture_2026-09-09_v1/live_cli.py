"""最新修復背景へ同update暫定context観測だけを加える。"""
from __future__ import annotations
import contextlib
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any

ROOT = Path(__file__).resolve().parent
PRIOR = ROOT.parent / "g2_current_scope_capture_2026-09-09_v1"


def load(name: str, path: Path) -> Any:
    if name in sys.modules:
        raise RuntimeError("context_module_alias_collision")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


L = load("_context_previous_cli", PRIOR / "live_cli.py")
O = load("_context_outer_observer", ROOT / "observer.py")


def guards() -> dict[str, str]:
    return {str(ROOT / name): O.sha(ROOT / name) for name in ("live_cli.py", "launcher.sh", "preflight.py")}


def bootstrap() -> tuple[Any, Any, Any, Any]:
    prior, driver, old, engine = L.bootstrap()
    def install(stack: Any, collector: Any, history: Any, state: Any) -> None:
        old.install(stack, collector, history, state)
        O.install(stack, collector, history, state)
    def finish(state: Any) -> None:
        O.finish(state)
        old.finish(state)
    def verify(output: Path) -> None:
        O.verify(output)
        old.verify(output)
    addon = SimpleNamespace(REQUIRED=old.REQUIRED | O.REQUIRED, install=install,
        finish=finish, verify=verify, guards=lambda: old.guards() | O.guards() | guards())
    return prior, driver, addon, engine


def main() -> int:
    prior, driver, addon, engine = bootstrap()
    driver.validate_addon(addon)
    with L.L.configuration(prior, driver):
        runtime, finisher = driver.load_runtime()
        latest = runtime.M.A
        with driver.configured(runtime, finisher, addon), contextlib.ExitStack() as stack:
            engine.install(stack, runtime)
            with runtime.configured(), latest.configured(), latest.prior.configured():
                runtime.M.base.patch(stack, latest.entry, "run", runtime.run)
                runtime.M.base.patch(stack, latest.entry, "finalize", runtime.finalize)
                return latest.entry.main()


if __name__ == "__main__":
    raise SystemExit(main())
