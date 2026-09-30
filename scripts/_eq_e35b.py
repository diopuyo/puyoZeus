"""E35b同値性: 作業ツリーの新実装でE36bと同じ入力・オプションを再生する（出力は別ディレクトリ）。"""
from __future__ import annotations

import sys
from pathlib import Path

from scripts import d5_runtime, run_e36, run_e36b

OUT = Path('logs/e35b/eq_e36b')


def main() -> None:
    """d5_runtimeの0f28a04固定を無効化し、作業ツリーのsrcを読ませる。"""
    d5_runtime.install = lambda: None
    run_e36.OUT, run_e36.options = OUT, run_e36b.options
    run_e36.worker(sys.argv[1])


if __name__ == '__main__':
    main()
