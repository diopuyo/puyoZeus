"""固定palette候補の証拠を先に閉じ、互換保存契約を後段へ接続する。"""
from __future__ import annotations
import contextlib
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any

ROOT = Path(__file__).resolve().parent
OWN = ('live_cli.py', 'test_runtime.py', 'preflight.py', 'run_cpu.py', 'launcher.sh', 'CONTRACT.md')


def load(name: str, path: Path) -> Any:
    if name in sys.modules:
        raise RuntimeError('palette_finish_alias_collision')
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


B = load('_palette_finish_prior_cli', ROOT.parent / 'g2_palette_veto_runtime_2026-09-09_v1/live_cli.py')
F, configuration = B.F, B.configuration


def proof_adapter() -> Any:
    return load('_palette_finish_evidence_proof', ROOT.parent / 'g2_palette_finish_proof_2026-09-09_v1/adapter.py')


def guards() -> dict[str, str]:
    return B.guards() | {str(ROOT / name): F.sha(ROOT / name) for name in OWN}


def bootstrap() -> tuple[Any, Any, Any, Any]:
    prior, driver, old, engine = B.B.bootstrap()
    palette, proof = B.observer(), proof_adapter()
    def install(stack: Any, collector: Any, history: Any, state: Any) -> None:
        old.install(stack, collector, history, state)
        palette.install(stack, collector, history, state, enabled=True)
    def finish(state: Any) -> None:
        palette.finish(state)
        old.finish(state)
    def verify(output: Path) -> None:
        old.verify(output)
        palette.verify(output)
        proof.verify(output)
    addon = SimpleNamespace(REQUIRED=old.REQUIRED | palette.REQUIRED | proof.REQUIRED,
        install=install, finish=finish, verify=verify, bind=old.bind,
        guards=lambda: old.guards() | palette.guards() | proof.guards() | guards(), proof=proof)
    return prior, driver, addon, engine


@contextlib.contextmanager
def configured(runtime: Any, finisher: Any, driver: Any, addon: Any, engine: Any) -> Any:
    with B.B.configured(runtime, finisher, driver, addon, engine):
        with contextlib.ExitStack() as stack:
            addon.proof.install(stack, F.P, enabled=True)
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
