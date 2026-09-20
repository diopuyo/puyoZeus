"""既存実走へ観測窓設定を加える。下流公開の変更は含めない。"""
from __future__ import annotations
import contextlib
import importlib.util
import sys
from types import SimpleNamespace
from typing import Any
import scope as O

ROOT = O.ROOT
PATH = ROOT.parent / 'g2_provisional_context_capture_2026-09-09_v1/live_cli.py'
SPEC = importlib.util.spec_from_file_location('_full_scope_previous_cli', PATH)
L = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = L
SPEC.loader.exec_module(L)


def bootstrap() -> tuple[Any, Any, Any, Any]:
    prior, driver, old, engine = L.bootstrap()
    def finish(state: Any) -> None:
        old.finish(state)
        O.finish(state)
    def verify(output: Any) -> None:
        old.verify(output)
        O.verify(output)
    addon = SimpleNamespace(REQUIRED=old.REQUIRED | {O.RECEIPT}, install=old.install,
        finish=finish, verify=verify, guards=lambda: old.guards() | O.guards())
    return prior, driver, addon, engine


@contextlib.contextmanager
def configuration(prior: Any, driver: Any) -> Any:
    with O.configured(), L.L.L.configuration(prior, driver):
        yield


def main() -> int:
    prior, driver, addon, engine = bootstrap()
    driver.validate_addon(addon)
    with configuration(prior, driver):
        runtime, finisher = driver.load_runtime()
        latest = runtime.M.A
        with driver.configured(runtime, finisher, addon), contextlib.ExitStack() as stack:
            engine.install(stack, runtime)
            with runtime.configured(), latest.configured(), latest.prior.configured():
                runtime.M.base.patch(stack, latest.entry, 'run', runtime.run)
                runtime.M.base.patch(stack, latest.entry, 'finalize', runtime.finalize)
                return latest.entry.main()


if __name__ == '__main__':
    raise SystemExit(main())
