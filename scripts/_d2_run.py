"""D2短区間診断を独立プロセスで順次実行する。"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys

OUT = Path('logs/d2')
SHORT_SECONDS = 10
NICE_LEVEL = 10


def main() -> None:
    """WSLでdetachし、診断結果だけを指定フォルダへ保存する。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--jobs', nargs='+', default=['q_new=new', 'q_old=old'])
    parser.add_argument('--source', default='q_7gc4TgFig')
    parser.add_argument('--seconds', type=int, default=SHORT_SECONDS)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE='1',
                       PYTHONUNBUFFERED='1', PYTHONPATH='.', OMP_NUM_THREADS='1',
                       OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    if not args.worker:
        with (OUT/'runner.log').open('a') as stream:
            child = subprocess.Popen([sys.executable, '-m', 'scripts._d2_run',
                '--worker', '--source', args.source, '--seconds', str(args.seconds),
                '--jobs', *args.jobs],
                env=environment, stdout=stream, stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL, start_new_session=True)
        print(child.pid)
        return
    os.nice(NICE_LEVEL)
    for job in args.jobs:
        name, variant = job.split('=', 1)
        source = args.source
        if '@' in name:
            name, source = name.split('@', 1)
        with (OUT/f'{name}.log').open('w') as stream:
            subprocess.run([sys.executable, '-m', 'scripts._d2_probe',
                '--source', source, '--variant', variant, '--name', name,
                '--seconds', str(args.seconds)],
                env=environment, stdout=stream, stderr=subprocess.STDOUT, check=True)


if __name__ == '__main__':
    main()
