"""全域観測＋元保存契約を維持し、限定2P退出だけを修復する。"""
from __future__ import annotations
import contextlib
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any
import runtime_hook as H

ROOT = H.ROOT
PRIOR = ROOT.parent / 'g2_full_observation_scope_2026-09-09_v1'
sys.path.insert(0, str(PRIOR))
SPEC = importlib.util.spec_from_file_location('_floating_full_scope_cli', PRIOR / 'live_cli.py')
L = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = L
SPEC.loader.exec_module(L)


def bootstrap() -> tuple[Any, Any, Any, Any]:
    # 全frame不変という観測専用finishは使わず、元C6/raw/NEXTの比較契約を保持。
    prior, driver, old, engine = L.L.bootstrap()
    def install(stack: Any, collector: Any, history: Any, state: Any) -> None:
        old.install(stack, collector, history, state)
        H.install(stack, collector, history, state)
    def finish(state: Any) -> None:
        old.finish(state)
        H.finish(state)
    def verify(output: Path) -> None:
        old.verify(output)
        H.verify(output)
    addon = SimpleNamespace(REQUIRED=old.REQUIRED | H.REQUIRED, install=install,
        finish=finish, verify=verify, guards=lambda: old.guards() | L.O.guards() | H.guards())
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
                runtime.M.base.patch(stack, latest.entry, 'run', runtime.run)
                runtime.M.base.patch(stack, latest.entry, 'finalize', runtime.finalize)
                return latest.entry.main()


if __name__ == '__main__':
    raise SystemExit(main())
