"""負荷条件を満たす場合だけ、B1本測定を逐次実行する。"""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import time

OUTPUT = Path('logs/live_b4_baseline')
LOAD_LIMIT = 1.0


def main() -> None:
    """各条件直前の負荷・nice・終了状態を保存する。"""
    OUTPUT.mkdir(parents=True, exist_ok=True)
    records = []
    for condition in ('gpu', 'cpu'):
        load = os.getloadavg()
        row = dict(condition=condition, loadavg=load, nice=os.nice(0), time=time.time())
        if load[0] >= LOAD_LIMIT:
            row['status'] = 'skipped_load'
        else:
            command = [sys.executable, '-m', 'scripts.measure_realtime_breakdown_20260928',
                       '--start-sec', '2580', '--end-sec', '2700', '--parallel-count', '1',
                       '--no-render', '--output', str(OUTPUT / condition)]
            env = dict(os.environ, PYTHONPATH='.')
            if condition == 'cpu':
                env['CUDA_VISIBLE_DEVICES'] = ''
            with (OUTPUT / f'{condition}.log').open('w') as log:
                row['exitcode'] = subprocess.run(command, env=env, stdout=log,
                                                stderr=subprocess.STDOUT).returncode
            row['status'] = 'completed' if row['exitcode'] == 0 else 'failed'
            row['command'] = command
        records.append(row)
        (OUTPUT / 'conditions.json').write_text(json.dumps(records, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
