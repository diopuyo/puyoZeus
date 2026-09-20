"""旧prepareへだけ委譲する。実動画/GPUの開始権限は持たない。"""
from __future__ import annotations
from contextlib import ExitStack
from pathlib import Path
import importlib.util
import sys

ROOT = Path(__file__).resolve().parent


def main() -> int:
    assert sys.argv.count('--mode') == 1, 'private_prepare_mode_required'
    position = sys.argv.index('--mode')+1
    assert position < len(sys.argv) and sys.argv[position] == 'prepare', 'private_prepare_only'
    with ExitStack() as stack:
        alias = '_private_prepare_adapter'
        assert alias not in sys.modules
        spec = importlib.util.spec_from_file_location(alias, ROOT/'adapter.py')
        module = importlib.util.module_from_spec(spec)
        sys.modules[alias] = module
        stack.callback(sys.modules.pop, alias, None)
        spec.loader.exec_module(module)
        return module.configured(stack)()


if __name__ == '__main__':
    raise SystemExit(main())
