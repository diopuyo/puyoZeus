"""run_limited経由で診断子プロセスだけを最大三本まで実行する。"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import os
import shlex
import subprocess
import sys

from scripts._diag_dv2d_loss_inventory import OUT, ROOT, read

WORKERS = 3


def run(job: tuple[str, str]) -> dict:
    """既存出力を拒否し、資源制限起動口を各子に適用する。"""
    mode, source = job
    logfile = OUT/f'{mode}_{source}_r2.log'
    assert not logfile.exists()
    command = ['nice', '-n', '10', sys.executable, '-B', '-m',
               'scripts._diag_dv2d_loss_trace', source, '--mode', mode]
    shell = f'OUT={shlex.quote(str(OUT))}; source logs/resource_falsedeath.sh; '
    shell += f'run_limited {shlex.quote(str(logfile))} {shlex.join(command)}'
    result = subprocess.run(['bash', '-c', shell], cwd=ROOT, check=False)
    print(f'{mode}/{source}: {result.returncode}', flush=True)
    return dict(mode=mode, source=source, returncode=result.returncode)


def main() -> None:
    """制御器を除外し、実計算子は共有上限でも数える。"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='trace')
    parser.add_argument('sources', nargs='*')
    args = parser.parse_args()
    os.environ['FALSEDEATH_CONTROLLER_MODULES'] = 'scripts._diag_dv2d_loss_runner'
    sources = args.sources or list(dict.fromkeys(r['record'] for r in read(OUT/'EVENTS.json')))
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        results = list(pool.map(run, [(mode, source) for mode in args.mode.split(',') for source in sources]))
    assert all(r['returncode'] == 0 for r in results), results


if __name__ == '__main__':
    main()
