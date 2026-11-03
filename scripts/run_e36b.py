"""E36b: E36と同一構成にD5b (単発死亡証明の取消を生存枝のみに限定) を足して再生する。"""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

from scripts import run_d5
from scripts import run_e36

OUT = Path('logs/e36b')
E36_OPTIONS = run_e36.options  # 差し替え前の元関数 (自己再帰を避ける)


def options() -> dict:
    """E36のONにD5bを足す。"""
    return dict(E36_OPTIONS(), single_death_proof_negative_only=True)


def main() -> None:
    """run_e36の再生器を出力先だけ差し替えて、記録ごとに1プロセスで順次起動する。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', choices=run_d5.SOURCES)
    args = parser.parse_args()
    assert (OUT/'PREREGISTRATION.md').exists()
    if args.source:
        run_e36.OUT, run_e36.options = OUT, options
        run_e36.worker(args.source)
        return
    order = ('q_7gc4TgFig', *[s for s in run_d5.SOURCES if s != 'q_7gc4TgFig'])
    for source in order:
        with (OUT/f'{source}.log').open('a') as stream:
            subprocess.run([sys.executable, '-B', '-m', 'scripts.run_e36b', '--source', source],
                           stdout=stream, stderr=subprocess.STDOUT, check=True)


if __name__ == '__main__':
    main()
