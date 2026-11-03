"""診断入力を最大2プロセスで取得し、全出力の同一性を確認する。"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import subprocess
import sys

from scripts.gfe_phase_stage1_20261002 import memory_guard
from scripts.gfe_phase_stage2_replay_20261002 import SET1, SET2

OUT = Path('logs/gfe_phase_stage2_diag_20261002')
WORKERS = 2


def run(source: str) -> None:
    """別プロセスでパッチを隔離し、完了済みの区間には触れない。"""
    if (OUT/'capture'/source/'DONE.json').exists():
        return
    memory_guard()
    with (OUT/f'capture_{source}.log').open('w') as stream:
        subprocess.run([sys.executable, '-B', '-m', 'scripts._diag_gfe_stage2_capture', source],
                       stdout=stream, stderr=subprocess.STDOUT, check=True)
    print('complete', source, flush=True)


def main() -> None:
    """優先度19と6GiB以上の余裕を維持する。"""
    os.nice(max(0,19-os.nice(0)))
    memory_guard()
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        list(pool.map(run, (*SET1, *SET2)))


if __name__ == '__main__':
    main()
