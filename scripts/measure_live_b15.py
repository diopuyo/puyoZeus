"""一度だけ60分を実時間運転し、終了後に全フレーム参照と自動比較する。"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np

from scripts.live_b15_plan import START, END, DURATION, OUTPUT, prepare
from scripts.measure_live_b6 import save, wait_idle, MAX_WAIT_SECONDS
from scripts.measure_live_b9 import owned_process, cleanup_audit
from scripts.measure_live_b12 import monitor
from scripts.analyze_live_b14 import source_hashes

REFERENCE_TIMEOUT_FACTOR = 3


def command(root: Path) -> list[str]:
    from scripts.diagnose_live_b8 import case_command
    args = case_command(('realtime', True, START, END, 1, 10), root)
    config_path = Path(args[args.index('--config')+1])
    config = json.loads(config_path.read_text())
    config.update(runtime_audit=True, coalesce_features=True, event_priority=True,
                  adaptive_evaluation=False, cpu_isolation=False)
    save(config_path, config)
    return args+['--realtime']


def realtime(root: Path) -> None:
    path = root/'realtime'
    path.mkdir(parents=True, exist_ok=False)
    args = command(root)
    with (path/'run.log').open('w') as log, owned_process(args, log) as child:
        save(root/'status.json', dict(state='running', phase='realtime', pid=os.getpid(), child=child.pid))
        save(path/'launch.json', dict(pid=child.pid, command=args, start=START, end=END))
        monitor(child, path, DURATION)
        child.wait()
        if child.returncode:
            raise RuntimeError(f'実時間運転に失敗: {child.returncode}')
    cleanup = cleanup_audit(child.pid)
    save(path/'cleanup.json', cleanup)
    if not cleanup['no_children_remain']:
        raise RuntimeError('所有プロセスが残りました')


def reference(root: Path) -> None:
    from scripts.run_live_pipeline_20260928 import asset_hashes
    metrics = json.loads((root/'realtime/metrics.json').read_text())
    if metrics['assets'] != asset_hashes():
        raise ValueError('実時間運転後にコード・モデル・較正資産が変わりました')
    save(root/'status.json', dict(state='running', phase='offline_reference', pid=os.getpid()))
    args = [sys.executable, '-m', 'scripts.profile_live_b14', '--output', str(root/'reference'),
            '--start', str(START), '--end', str(END)]
    with (root/'reference.log').open('w') as log:
        subprocess.run(args, stdout=log, stderr=subprocess.STDOUT, check=True,
                       timeout=DURATION*REFERENCE_TIMEOUT_FACTOR)
    metadata = json.loads((root/'reference/recognition.json').read_text())
    if metadata['dropped']:
        raise ValueError('全フレーム参照に捨てがあります')
    with np.load(root/'reference/recognition.npz') as data:
        if data['t_sec'][-1] < END-1:
            raise RuntimeError('全フレーム参照が終端へ届いていません')
    configs = [json.loads((path/'recognition_config.json').read_text())
               for path in (root/'realtime', Path('logs/live_b13/reference'))]
    if configs[0] != configs[1]:
        raise ValueError('全フレーム参照と実時間運転の認識設定が異なります')


def run(root: Path, commit: str) -> None:
    if (root/'launch.json').exists() or (root/'realtime').exists():
        raise ValueError('既存実行を上書き・再実行しません')
    plan = prepare(root)
    hashes = source_hashes()
    save(root/'launch.json', dict(pid=os.getpid(), commit=commit, started_unix=time.time(),
        start=START, end=END, expected_games=plan['expected_games'], source_hashes=hashes,
        cpu_threads=1, device='cpu', selected='c', realtime_runs=1))
    try:
        if not wait_idle(time.monotonic()+MAX_WAIT_SECONDS, root, 'B15 60min'):
            return
        if source_hashes() != hashes:
            raise ValueError('待機中に認識コードが変わりました')
        realtime(root)
        from scripts.analyze_live_b15 import runtime_report, quality_report
        runtime_report(root)
        if not wait_idle(time.monotonic()+MAX_WAIT_SECONDS, root, 'B15 offline reference'):
            return
        if source_hashes() != hashes:
            raise ValueError('実時間運転と参照の認識コードが異なります')
        reference(root)
        quality_report(root)
        save(root/'status.json', dict(state='finished', pid=os.getpid()))
    except Exception as error:
        save(root/'status.json', dict(state='failed', pid=os.getpid(), error=str(error)))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--commit', required=True)
    options = parser.parse_args()
    run(options.output, options.commit)
