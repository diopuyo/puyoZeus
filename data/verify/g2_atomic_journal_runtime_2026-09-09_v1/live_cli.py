"""最新T2成功背景へ同callの在庫journal観測だけを追加する。"""
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
        raise RuntimeError('journal_runtime_alias_collision')
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    sys.path.insert(0, str(path.parent))
    try:
        spec.loader.exec_module(module)
    except BaseException:
        del sys.modules[name]
        raise
    return module


B = load('_atomic_journal_t2_cli', ROOT.parent / 'g2_t2_bounded_runtime_2026-09-09_v1/live_cli.py')
F, configuration = B.F, B.configuration


def observer() -> Any:
    return load('_atomic_journal_observer', ROOT.parent / 'g2_atomic_journal_capture_2026-09-09_v1/observer.py')


def guards() -> dict[str, str]:
    return B.guards() | {str(ROOT / name): F.sha(ROOT / name) for name in OWN}


def bootstrap() -> tuple[Any, Any, Any, Any]:
    prior, driver, old, engine = B.bootstrap()
    journal = observer()
    def install(stack: Any, collector: Any, history: Any, state: Any) -> None:
        old.install(stack, collector, history, state)
        journal.install(stack, collector, history, state)
    def finish(state: Any) -> None:
        old.finish(state)
        journal.finish(state)
    def verify(output: Path) -> None:
        old.verify(output)
        journal.verify(output)
    addon = SimpleNamespace(REQUIRED=old.REQUIRED | journal.REQUIRED, install=install,
        finish=finish, verify=verify, guards=lambda: old.guards() | guards() | journal.guards(), bind=old.bind)
    return prior, driver, addon, engine


@contextlib.contextmanager
def configured(runtime: Any, finisher: Any, driver: Any, addon: Any, engine: Any) -> Any:
    with B.configured(runtime, finisher, driver, addon, engine):
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
