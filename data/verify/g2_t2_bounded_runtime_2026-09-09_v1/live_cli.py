"""成功済みbounded公開へT2条件修復だけを生成前に合成する。"""
from __future__ import annotations
import contextlib
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any

ROOT = Path(__file__).resolve().parent
OWN = ('live_cli.py', 'preflight.py', 'launcher.sh', 'test_runtime.py', 'run_cpu.py', 'CONTRACT.md')


def load(name: str, path: Path) -> Any:
    if name in sys.modules:
        raise RuntimeError('t2_combined_alias')
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    sys.path.insert(0, str(path.parent))
    spec.loader.exec_module(module)
    return module


B = load('_t2_bounded_prior_cli', ROOT.parent / 'g2_bounded_publication_runtime_2026-09-09_v1/live_cli.py')
T = load('_t2_fresh_landing_guard', ROOT.parent / 'g2_t2_fresh_landing_guard_2026-09-09_v1/adapter.py')
C, F, configuration = B.C, B.F, B.configuration


def guards() -> dict[str, str]:
    return B.guards() | T.guards() | {str(ROOT / name): F.sha(ROOT / name) for name in OWN}


def bootstrap() -> tuple[Any, Any, Any, Any]:
    prior, driver, old, engine = B.bootstrap()
    addon = SimpleNamespace(REQUIRED=old.REQUIRED, install=old.install, finish=old.finish,
        verify=old.verify, guards=lambda: old.guards() | guards(), bind=old.bind)
    return prior, driver, addon, engine


@contextlib.contextmanager
def configured(runtime: Any, finisher: Any, driver: Any, addon: Any, engine: Any) -> Any:
    with contextlib.ExitStack() as stack:
        stack.enter_context(B.configured(runtime, finisher, driver, addon, engine))
        original = runtime.M.instrument
        def instrument(inner: Any, collector: Any, history: Any, receipt: Any, state: Any) -> None:
            T.install(inner, runtime.M, enabled=True)
            original(inner, collector, history, receipt, state)
        runtime.M.base.patch(stack, runtime.M, 'instrument', instrument)
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
