"""B16のまとめ更新とB17の逐次更新を同一5分区間で直列実走する。"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
from pathlib import Path
import pickle
import sys
import time
from typing import Any
from unittest.mock import patch

from scripts.measure_live_b6 import save, wait_idle, MAX_WAIT_SECONDS

START, DURATION = 5880.566, 300.0
OUTPUT = Path('logs/live_b17/live')
STARTUP_TIMEOUT, STARTUP_POLL = 120.0, 0.1
END_TOLERANCE = 1.0


def batched_update(self: Any, *args: Any) -> None:
    """f9d029fの更新方式を検証時だけ復元する。"""
    from src.phase_j.live_spool import DiskRows
    self.t_sec, game = args[3:5]
    if game != self.game:
        self.journal.close()
        self.journal = DiskRows(self.directory/'evaluation-journal.pickle')
        self.game = game
    blob = pickle.dumps(('update', args), protocol=pickle.HIGHEST_PROTOCOL)
    self.journal.append(blob)
    self.pending.append(blob)


def worker(mode: str) -> None:
    from scripts.run_live_pipeline_20260928 import parse_args
    from src.phase_j.live_lifetime import join_evaluation
    options = parse_args()
    options.output.mkdir(parents=True, exist_ok=False)
    # 本番同様、CPU環境設定後に評価をspawnし、BLAS設定も継承する。
    child = mp.get_context('spawn').Process(target=evaluation,
        args=(mode, options, os.getpid()), name='evaluation-worker')
    child.start()
    join_evaluation(child)
    if child.exitcode:
        raise SystemExit(child.exitcode)


def evaluation(mode: str, options: argparse.Namespace, parent: int) -> None:
    from scripts.run_live_pipeline_20260928 import run_live
    from src.phase_j.live_eval_supervisor import SupervisedOverlay
    from src.phase_j.live_lifetime import protect_parent
    protect_parent(parent)
    if mode == 'baseline':
        with patch.object(SupervisedOverlay, 'update', batched_update):
            run_live(options)
    else:
        run_live(options)


def execute(root: Path, mode: str) -> None:
    from scripts.measure_live_b12 import command, monitor
    from scripts.measure_live_b9 import owned_process, cleanup_audit
    from scripts.analyze_live_b9 import windows
    from scripts.analyze_live_b12 import report
    args = command(mode, root, DURATION, 'c')
    config_path = Path(args[args.index('--config')+1])
    config = json.loads(config_path.read_text())
    save(config_path, dict(config, start_sec=START, end_sec=START+DURATION))
    args[2] = 'scripts.measure_live_b17'
    args[3:3] = ['--worker', mode]
    path = root/mode
    save(root/'status.json', dict(state='running', mode=mode, pid=os.getpid()))
    # run_liveの監査出力先はworker自身が作成し、monitorは生成後に開く。
    with (root/(mode+'.log')).open('w') as log, owned_process(args, log) as child:
        deadline = time.monotonic()+STARTUP_TIMEOUT
        while not path.exists() and child.poll() is None:
            if time.monotonic() >= deadline:
                raise TimeoutError('計測workerの起動待ち上限')
            time.sleep(STARTUP_POLL)
        if not path.exists():
            raise RuntimeError(f'{mode}: 起動失敗')
        save(path/'launch.json', dict(command=args, pid=child.pid, start=START, end=START+DURATION))
        monitor(child, path, DURATION)
        child.wait()
        if child.returncode:
            raise RuntimeError(f'{mode}: exit={child.returncode}')
    cleanup = cleanup_audit(child.pid)
    if not cleanup['no_children_remain']:
        raise RuntimeError('所有プロセスが残りました')
    save(path/'comparison.json', dict(windows=windows(path, START, START+DURATION),
        contamination=report(path, START, START+DURATION)['windows'], cleanup=cleanup))


def compare_runs(root: Path) -> None:
    """同じ資産・認識設定の完走だけを前後比較として保存する。"""
    import numpy as np
    runs, configs, results = {}, {}, {}
    for mode in ('baseline', 'candidate'):
        path = root/mode
        runs[mode] = json.loads((path/'metrics.json').read_text())
        configs[mode] = json.loads((path/'recognition_config.json').read_text())
        errors = path/'evaluation_errors.jsonl'
        if errors.exists() and errors.read_text().strip():
            raise ValueError(f'{mode}: 評価エラーがあります')
        with np.load(path/'recognition.npz') as data:
            if data['t_sec'][-1] < START+DURATION-END_TOLERANCE:
                raise ValueError(f'{mode}: 終端へ届いていません')
        results[mode] = json.loads((path/'comparison.json').read_text())
    if runs['baseline']['assets'] != runs['candidate']['assets'] or configs['baseline'] != configs['candidate']:
        raise ValueError('前後で資産または認識設定が変わりました')
    save(root/'report.json', dict(start=START, end=START+DURATION, runs=results,
                                  assets=runs['baseline']['assets'], evaluation_errors=0))


def main() -> None:
    if '--worker' in sys.argv:
        index = sys.argv.index('--worker')
        mode = sys.argv[index+1]
        del sys.argv[index:index+2]
        worker(mode)
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    try:
        for mode in ('baseline', 'candidate'):
            if (args.output/mode).exists():
                raise ValueError('既存計測を上書きしません')
            if not wait_idle(time.monotonic()+MAX_WAIT_SECONDS, args.output, 'B17 '+mode):
                return
            execute(args.output, mode)
        compare_runs(args.output)
    except Exception as error:
        save(args.output/'status.json', dict(state='failed', pid=os.getpid(), error=str(error)))
        raise
    save(args.output/'status.json', dict(state='finished', pid=os.getpid()))


if __name__ == '__main__':
    main()
