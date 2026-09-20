"""原CPU実waitのコードを出力rootのみ替えて再利用する。"""
from __future__ import annotations
import importlib.util
from pathlib import Path
from types import FunctionType

ROOT = Path(__file__).resolve().parent


def main() -> int:
    path = ROOT.parent/'g2_private_suffix_fusion_2026-09-10_v1/launch.py'
    spec = importlib.util.spec_from_file_location('_private_commit_launcher', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return FunctionType(module.main.__code__, dict(vars(module), ROOT=ROOT))()


if __name__ == '__main__':
    raise SystemExit(main())
