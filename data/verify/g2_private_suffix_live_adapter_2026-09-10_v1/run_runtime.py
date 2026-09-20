"""既存実wait launcherが呼ぶ薄いCPU入口。"""
from __future__ import annotations
import importlib.util
from pathlib import Path


def main() -> int:
    path = Path(__file__).resolve().parent/'run_cpu.py'
    spec = importlib.util.spec_from_file_location('_private_commit_cpu_entry', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.main()

if __name__ == '__main__':
    raise SystemExit(main())
