"""実公開の修復、採録metadata観測、限定保存契約を私有合成する。"""
from __future__ import annotations
import contextlib
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any

ROOT = Path(__file__).resolve().parent
OWN = ('live_cli.py', 'preflight.py', 'launcher.sh', 'test_runtime.py', 'run_cpu.py', 'CONTRACT.md', 'resource_guard.py')


def load(name: str, path: Path) -> Any:
    if name in sys.modules:
        raise ValueError('metadata_combined_alias')
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    sys.path.insert(0, str(path.parent))
    spec.loader.exec_module(module)
    return module


C = load('_metadata_prior_combined_cli', ROOT.parent / 'g2_floating_publication_runtime_2026-09-09_v1/live_cli.py')
M = load('_bounded_metadata_observer', ROOT.parent / 'g2_collector_metadata_bounded_2026-09-09_v1/bounded.py')
D = load('_metadata_downstream_finisher', ROOT.parent / 'g2_publication_downstream_finisher_2026-09-09_v1/adapter.py')
P = load('_bounded_pending_current', ROOT.parent / 'g2_pending_fresh_current_release_2026-09-09_v1/adapter.py')
F = C.F
configuration = C.L.L.configuration


def guards() -> dict[str, str]:
    return C.guards() | M.guards() | D.guards() | P.guards() | {str(ROOT / n): F.sha(ROOT / n) for n in OWN}


def current_release(state: dict[str, Any]) -> Any:
    publisher = state[C.A.STATE_KEY].publisher
    module = sys.modules[type(publisher).__module__]
    F.require(Path(module.__file__).resolve() == C.A.ROOT / 'connected.py'
        and type(publisher) is module.Publisher, 'bounded_actual_publisher_identity')
    F.require(type(publisher).wrap.__globals__ is vars(module), 'bounded_publisher_globals')
    return module.R


def bootstrap() -> tuple[Any, Any, Any, Any]:
    prior, driver, old, engine = C.bootstrap()
    def install(stack: Any, collector: Any, history: Any, state: Any) -> None:
        old.install(stack, collector, history, state)
        P.install(stack, current_release(state), state, enabled=True)
        M.install(stack, collector, history, state, enabled=True)
    def finish(state: Any) -> None:
        old.finish(state)
        M.finish(state)
        P.finish(state)
    def verify(output: Path) -> None:
        old.verify(output)
        M.verify(output)
        P.verify(output)
    addon = SimpleNamespace(REQUIRED=old.REQUIRED | M.REQUIRED | P.REQUIRED, install=install, finish=finish,
        verify=verify, guards=lambda: old.guards() | guards(), bind=old.bind)
    return prior, driver, addon, engine


@contextlib.contextmanager
def configured(runtime: Any, finisher: Any, driver: Any, addon: Any, engine: Any) -> Any:
    addon.bind(runtime)
    with contextlib.ExitStack() as stack:
        F.install(stack, runtime, enabled=True)
        D.install(stack, runtime, finisher, F.P, enabled=True)
        stack.enter_context(driver.configured(runtime, finisher, addon))
        engine.install(stack, runtime)
        yield


def main() -> int:
    prior, driver, addon, engine = bootstrap()
    driver.validate_addon(addon)
    with configuration(prior, driver):
        runtime, finisher = driver.load_runtime()
        latest = runtime.M.A
        with configured(runtime, finisher, driver, addon, engine):
            with contextlib.ExitStack() as stack, runtime.configured(), latest.configured(), latest.prior.configured():
                runtime.M.base.patch(stack, latest.entry, 'run', runtime.run)
                runtime.M.base.patch(stack, latest.entry, 'finalize', runtime.finalize)
                return latest.entry.main()


if __name__ == '__main__':
    raise SystemExit(main())
