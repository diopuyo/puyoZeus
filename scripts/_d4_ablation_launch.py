"""D4境界介入を直列で実行する。"""
from __future__ import annotations
import subprocess
import sys
from scripts._d4_loss import OUT
from scripts._d4_counterfactual import KINDS


def main() -> None:
    """一条件ごとの結果を保存して安全に再開する。"""
    for signal in KINDS:
        dest = OUT/'completion_ablation'/signal
        dest.mkdir(parents=True, exist_ok=True)
        if (dest/'RESULT.json').exists():
            continue
        with (dest/'runner.log').open('w') as stream:
            subprocess.run([sys.executable, '-B', '-m', 'scripts._d4_counterfactual',
                '--signal', signal], stdout=stream, stderr=subprocess.STDOUT, check=True)


if __name__ == '__main__':
    main()
