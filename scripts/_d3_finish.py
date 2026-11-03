"""D3診断の後半を、先行処理の終了後に一つずつ実行する。"""
from __future__ import annotations
import argparse
import os
import subprocess
import sys
import time
from scripts._d3_inventory import OUT

NICE = 19


def main() -> None:
    """プロセス終了後も継続する直列の診断ランナー。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--refinalize', action='store_true')
    args = parser.parse_args()
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1',
        OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    if not args.worker:
        with (OUT/'finish.log').open('a') as stream:
            extra = ['--refinalize'] if args.refinalize else []
            child = subprocess.Popen([sys.executable, '-B', '-m', 'scripts._d3_finish', '--worker', *extra],
                env=environment, stdin=subprocess.DEVNULL, stdout=stream,
                stderr=subprocess.STDOUT, start_new_session=True)
        print(child.pid)
        return
    os.nice(NICE)
    while not (OUT/'zenchi_formula_scan.json').exists():
        time.sleep(5)
    jobs = ([] if args.refinalize else [('_d3_onsets', [])])
    jobs += [('_d3_refresh', []), ('_d3_observe', ['--worker']), ('_d3_measure', [])]
    if args.refinalize:
        jobs += [('_d3_d1_audit', []), ('_d3_report', [])]
    for script, options in jobs:
        print('START', script, flush=True)
        subprocess.run([sys.executable, '-B', '-m', f'scripts.{script}', *options],
            env=environment, check=True)
        print('END', script, flush=True)
    (OUT/'diagnosis_complete.json').write_text('{"completed": true}')


if __name__ == '__main__':
    main()
