"""検収済みwitnessを固定SHAで接続する診断専用CLI。本番権は発行しない。"""
from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[2]
ENTRY_SHA = "78212a3b45ce7f09c8f38c2618aba468836f66f8eb20633f1a7f7c0d6c22d000"
OBSERVER_ROOT = ROOT.parent / "g2_hsv_correction_witness_2026-09-08_v1"
OBSERVER_SHA = "7b3407ccbc46bd5ad48fdcc1fc15810ab3dce2c39866ab42b0d5da3d1084642d"
EXECUTION_FILES = ("live_cli.py", "launcher.sh", "preflight.py", "test_runtime_connection.py",
                   "run_connection_cpu.py")


def bootstrap() -> Any:
    """assertを含む凍結旧検査の最適化実行を入口で拒否する。"""
    if sys.flags.optimize:
        raise RuntimeError("optimized_python_not_supported")
    sys.path.insert(0, str(PROJECT))
    path = ROOT / "entry.py"
    if hashlib.sha256(path.read_bytes()).hexdigest() != ENTRY_SHA:
        raise RuntimeError("driver_source_changed")
    spec = importlib.util.spec_from_file_location("_hsv_live_driver", path)
    module = importlib.util.module_from_spec(spec)
    if spec.name in sys.modules:
        raise RuntimeError("driver_alias_already_loaded")
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def addon(driver: Any) -> Any:
    observer = driver.load_fixed("_hsv_live_observer", OBSERVER_ROOT / "observer.py", OBSERVER_SHA)
    def guards() -> dict[str, str]:
        result = observer.guards()
        paths = [ROOT / name for name in EXECUTION_FILES]
        paths.extend(OBSERVER_ROOT / name for name in ("test_observer.py", "run_cpu.py"))
        result.update({str(path): driver.sha(path) for path in paths})
        return result
    return SimpleNamespace(REQUIRED=observer.REQUIRED, guards=guards, install=observer.install,
                           finish=observer.finish, verify=observer.verify)


def main() -> int:
    driver = bootstrap()
    return driver.live_main(addon(driver))


if __name__ == "__main__":
    raise SystemExit(main())
