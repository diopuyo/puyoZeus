"""両側SM追加観測とfinally後索引を旧修復背景へ合成する入口。"""
from __future__ import annotations
import contextlib
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any

ROOT = Path(__file__).resolve().parent
PRIOR = ROOT.parent / "g2_generation_publication_split_2026-09-09_v1"
ENGINE = ROOT.parent / "g2_engine_after_finally_2026-09-09_v1/adapter.py"
sys.path.insert(0, str(PRIOR))


def load(name: str, path: Path) -> Any:
    if name in sys.modules:
        raise RuntimeError("unexpected_scope_module_alias")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


L = load("_scope_previous_cli", PRIOR / "live_cli.py")
O = load("_scope_extended_observer", ROOT / "observer.py")


def guards() -> dict[str, str]:
    paths = [ROOT / name for name in ("live_cli.py", "observer.py", "launcher.sh", "preflight.py")]
    paths += [O.HOOK, ENGINE]
    return {str(path): O.sha(path) for path in paths}


def bootstrap() -> tuple[Any, Any, Any, Any]:
    prior, driver, old = L.bootstrap()
    engine = load("_scope_after_finally_engine", ENGINE)
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
        finish=finish, verify=verify, guards=lambda: old.guards() | guards() | engine.guards())
    return prior, driver, addon, engine


def main() -> int:
    prior, driver, addon, engine = bootstrap()
    driver.validate_addon(addon)
    with L.configuration(prior, driver):
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
