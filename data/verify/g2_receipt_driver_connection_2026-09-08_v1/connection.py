"""既存driverの設定hookだけを使い、修復保存と既存HSV観測を併用する。"""
from __future__ import annotations

import contextlib
import hashlib
import importlib.util
from pathlib import Path
import sys
from typing import Any, Iterator

ROOT = Path(__file__).resolve().parent
DRIVER_ROOT = ROOT.parent / "g2_hsv_witness_driver_2026-09-08_v1"
REPAIR = ROOT.parent / "g2_finisher_receipt_repair_2026-09-08_v1/adapter.py"
REPAIR_SHA = "e82d1cac690f77bbc819cecad29e0de294bab8fa7b94d2167ceb3b40498692a6"
LEGACY = ROOT.parent / "g2_split_diagnostic_finisher_2026-09-08_v1/finisher.py"
LEGACY_SHA = "96541c1e7df8b555e26bd5cca742d16060ba46f06c925e675adaacb5decc0702"
CLI = DRIVER_ROOT / "live_cli.py"
CLI_SHA = "147678be2878ec239ac5d5bdbec4a46cc473096f84b157e90cc4227eec1a9b08"
ENTRY_SHA = "78212a3b45ce7f09c8f38c2618aba468836f66f8eb20633f1a7f7c0d6c22d000"
OWN = ("connection.py", "test_connection.py", "run_cpu.py", "PLAN.md")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_cli() -> Any:
    if sys.flags.optimize:
        raise RuntimeError("optimized_python_not_supported")
    alias = "_receipt_connection_original_cli"
    if alias in sys.modules or sha(CLI) != CLI_SHA:
        raise RuntimeError("unexpected_cli_alias_or_source")
    spec = importlib.util.spec_from_file_location(alias, CLI)
    module = importlib.util.module_from_spec(spec)
    sys.modules[alias] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(alias, None)
        raise
    return module


def guards() -> dict[str, str]:
    fixed = {str(CLI): CLI_SHA, str(DRIVER_ROOT / "entry.py"): ENTRY_SHA,
             str(REPAIR): REPAIR_SHA, str(LEGACY): LEGACY_SHA}
    if any(sha(Path(path)) != digest for path, digest in fixed.items()):
        raise RuntimeError("fixed_connection_source_changed")
    return {**fixed, **{str(ROOT / name): sha(ROOT / name) for name in OWN}}


@contextlib.contextmanager
def configuration(driver: Any) -> Iterator[None]:
    """旧定数とguardをscope内だけ差替える。二重設定や別driverを拒否する。"""
    guards()
    if Path(driver.__file__).resolve() != DRIVER_ROOT / "entry.py":
        raise ValueError("unexpected_driver_identity")
    if driver.FINISHER != LEGACY or driver.FINISHER_SHA != LEGACY_SHA:
        raise ValueError("unexpected_finisher_or_reentry")
    original = driver.own_guards
    def extended() -> dict[str, str]:
        result = dict(original())
        for path, digest in guards().items():
            if path in result and result[path] != digest:
                raise ValueError("connection_guard_conflict")
            result[path] = digest
        return result
    with contextlib.ExitStack() as stack:
        for name, value in (("FINISHER", REPAIR), ("FINISHER_SHA", REPAIR_SHA), ("own_guards", extended)):
            old = getattr(driver, name)
            setattr(driver, name, value)
            stack.callback(setattr, driver, name, old)
        yield


def main() -> int:
    """将来の診断入口。起動は接続/prepare検収と新証拠取得の必要性確認後のみ。"""
    cli = load_cli()
    driver = cli.bootstrap()
    with configuration(driver):
        return driver.live_main(cli.addon(driver))
