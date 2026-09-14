"""着地修復・全域観測・実公開と、限定採否保存契約を合成する。"""
from __future__ import annotations
import contextlib
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any

ROOT = Path(__file__).resolve().parent
OWN = ('live_cli.py', 'preflight.py', 'launcher.sh', 'test_runtime.py', 'run_cpu.py', 'CONTRACT.md')


def load(name: str, path: Path, *, restore_path: bool = True) -> Any:
    if name in sys.modules:
        raise ValueError('combined_private_alias')
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    previous = list(sys.path)
    try:
        sys.path.insert(0, str(path.parent))
        spec.loader.exec_module(module)
    finally:
        if restore_path:
            sys.path[:] = previous
    return module


L = load('_combined_floating_cli', ROOT.parent / 'g2_floating_single_runtime_2026-09-09_v1/live_cli.py', restore_path=False)
A = load('_combined_publication_adapter', ROOT.parent / 'g2_publication_consumer_runtime_2026-09-09_v1/runtime_adapter.py')
F = load('_combined_prefix_finish', ROOT.parent / 'g2_publication_collector_finish_2026-09-09_v1/finish.py')


def guards() -> dict[str, str]:
    return F.P.guards() | {str(ROOT / name): F.sha(ROOT / name) for name in OWN}


def bootstrap() -> tuple[Any, Any, Any, Any]:
    prior, driver, old, engine = L.bootstrap()
    runtime_box = []
    def bind(runtime: Any) -> None:
        F.require(not runtime_box, 'combined_runtime_rebind')
        runtime_box.append(runtime)
    def install(stack: Any, collector: Any, history: Any, state: Any) -> None:
        old.install(stack, collector, history, state)
        A.install(stack, collector, history, state)
        F.require(F.P.KEY not in state, 'combined_collector_reentry')
        state[F.P.KEY] = collector
    def finish(state: Any) -> None:
        old.finish(state)
        A.finish(state)
        F.require(len(runtime_box) == 1, 'combined_runtime_unbound')
        output = state['output']
        F.P.prove(output, runtime_box[0].comparisons(output, state), state)
    def verify(output: Path) -> None:
        old.verify(output)
        A.verify(output)
        F.P.verify(output)
    addon = SimpleNamespace(REQUIRED=old.REQUIRED | A.REQUIRED | F.P.REQUIRED, install=install,
        finish=finish, verify=verify, guards=lambda: old.guards() | A.guards() | guards(), bind=bind)
    return prior, driver, addon, engine


@contextlib.contextmanager
def configured(runtime: Any, finisher: Any, driver: Any, addon: Any, engine: Any) -> Any:
    addon.bind(runtime)
    with contextlib.ExitStack() as stack:
        F.install(stack, runtime, enabled=True)
        stack.enter_context(driver.configured(runtime, finisher, addon))
        engine.install(stack, runtime)
        yield


def main() -> int:
    prior, driver, addon, engine = bootstrap()
    driver.validate_addon(addon)
    with L.L.configuration(prior, driver):
        runtime, finisher = driver.load_runtime()
        latest = runtime.M.A
        with configured(runtime, finisher, driver, addon, engine):
            with contextlib.ExitStack() as stack, runtime.configured(), latest.configured(), latest.prior.configured():
                runtime.M.base.patch(stack, latest.entry, 'run', runtime.run)
                runtime.M.base.patch(stack, latest.entry, 'finalize', runtime.finalize)
                return latest.entry.main()


if __name__ == '__main__':
    raise SystemExit(main())
