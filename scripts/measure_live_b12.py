"""B12: 所有した人工CPU負荷と縮退案を直列比較し、longを常時監査する。"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

from scripts.measure_live_b6 import save, wait_idle, MAX_WAIT_SECONDS
from scripts.measure_live_b9 import owned_process, cleanup_audit, STARTUP_TIMEOUT_SEC
from scripts.live_b9_resources import sample
from src.phase_j.live_load import CpuSampler, SAMPLE_SECONDS

START = 2580.566
LONG_END = 3412.0
RESOURCE_PERIOD = 5.0
CASE_FLAGS = {'baseline': (), 'a': ('cpu_isolation',), 'b': ('adaptive_evaluation',),
              'c': ('event_priority',), 'ab': ('cpu_isolation', 'adaptive_evaluation'),
              'ac': ('cpu_isolation', 'event_priority'), 'bc': ('adaptive_evaluation', 'event_priority'),
              'abc': ('cpu_isolation', 'adaptive_evaluation', 'event_priority')}
MODULE = 'scripts.measure_live_b12'
STRESS_BATCH = 10000
LCG_MULTIPLIER, LCG_INCREMENT, LCG_MASK = 1664525, 1013904223, 0xffffffff


def command(name: str, root: Path, duration: float, selected: str) -> list[str]:
    from scripts.diagnose_live_b8 import case_command
    args = case_command((name, True, START, START+duration, 1, 10), root)
    config_path = Path(args[args.index('--config')+1])
    config = json.loads(config_path.read_text())
    config.update(runtime_audit=True, coalesce_features=True)
    config.update({flag: flag in CASE_FLAGS[selected] for flag in CASE_FLAGS['abc']})
    save(config_path, config)
    return args


def stress_worker(seconds: float, parent: int) -> None:
    from src.phase_j.live_lifetime import protect_parent
    protect_parent(parent)
    deadline, value = time.monotonic()+seconds, 1
    while time.monotonic() < deadline:
        for _ in range(STRESS_BATCH):
            value = (value*LCG_MULTIPLIER+LCG_INCREMENT) & LCG_MASK


def monitor(child: subprocess.Popen[Any], path: Path, duration: float, workers: list | None = None) -> None:
    sampler = CpuSampler(child.pid, controlled_groups=tuple(p.pid for p in workers or []))
    deadline = time.monotonic()+STARTUP_TIMEOUT_SEC+duration*2
    last_resource, next_sample = float('-inf'), time.monotonic()+SAMPLE_SECONDS
    with (path/'cpu_load.jsonl').open('w') as load, (path/'resources.jsonl').open('w') as resource:
        while child.poll() is None:
            if any(p.poll() is not None for p in workers or []):
                raise RuntimeError('人工負荷processが計測中に終了しました')
            time.sleep(max(0, next_sample-time.monotonic()))
            row = sampler.sample()
            load.write(json.dumps(row)+'\n')
            load.flush()
            next_sample = row['end']+SAMPLE_SECONDS
            if row['end']-last_resource >= RESOURCE_PERIOD:
                resource.write(json.dumps(sample(child.pid))+'\n')
                resource.flush()
                last_resource = row['end']
            if time.monotonic() >= deadline:
                raise TimeoutError('計測processが制限時間を超えました')


def execute(name: str, root: Path, duration: float, selected: str,
            stress: int, cpus: str | None) -> dict:
    args = command(name, root, duration, selected)
    path = root/name
    path.mkdir(parents=True, exist_ok=True)
    prefix = ['taskset', '-c', cpus] if cpus else []
    save(root/'status.json', dict(state='running', case=name, pid=os.getpid()))
    with ExitStack() as stack:
        log = stack.enter_context((path/'run.log').open('w'))
        workers = [stack.enter_context(owned_process(prefix+[sys.executable, '-m', MODULE,
            '--stress-worker', str(STARTUP_TIMEOUT_SEC+duration*2), '--parent', str(os.getpid())], log))
            for _ in range(stress)]
        child = stack.enter_context(owned_process(prefix+args, log))
        save(path/'launch.json', dict(pid=child.pid, command=prefix+args, start=START, end=START+duration,
             stress_pids=[p.pid for p in workers], stress_count=stress, cpus=cpus, flags=CASE_FLAGS[selected]))
        monitor(child, path, duration, workers)
        child.wait()
        if child.returncode:
            raise RuntimeError(f'{name}: exit={child.returncode}')
    result = dict(returncode=child.returncode, **cleanup_audit(child.pid))
    result['stress_cleanup'] = [cleanup_audit(p.pid) for p in workers]
    from scripts.analyze_live_b12 import report
    result['contamination'] = report(path, START, START+duration)
    result['unplanned_contamination'] = report(path, START, START+duration, unplanned=True)
    if name == 'long':
        from scripts.analyze_live_b9 import report as acceptance
        result['acceptance'] = acceptance(path, START, START+duration)
    save(path/'result.json', result)
    if not result['no_children_remain'] or not all(r['no_children_remain'] for r in result['stress_cleanup']):
        raise RuntimeError('所有processの残留を検出しました。result.jsonを確認してください')
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('logs/live_b12'))
    parser.add_argument('--cases', nargs='+', choices=CASE_FLAGS, default=list(CASE_FLAGS))
    parser.add_argument('--duration', type=float, default=120)
    parser.add_argument('--stress', type=int, default=3)
    parser.add_argument('--cpus', default='0,2,4,6')
    parser.add_argument('--long', action='store_true')
    parser.add_argument('--selected', choices=CASE_FLAGS, default='c')
    parser.add_argument('--commit')
    parser.add_argument('--wait-idle', action='store_true')
    parser.add_argument('--stress-worker', type=float)
    parser.add_argument('--parent', type=int)
    parser.add_argument('--reference', type=Path, default=Path('logs/live_b11/baseline'))
    options = parser.parse_args()
    if options.stress_worker:
        stress_worker(options.stress_worker, options.parent)
        return
    run(options)


def run(options: argparse.Namespace) -> None:
    if os.nice(0) != 0 or options.duration <= 0 or options.stress < 0:
        raise ValueError('nice 0、正の尺、非負の負荷本数が必要です')
    if options.long and not options.commit:
        raise ValueError('longの起動には--commitが必要です')
    root = options.output
    root.mkdir(parents=True, exist_ok=True)
    save(root/'launch.json', dict(pid=os.getpid(), commit=options.commit, started_unix=time.time(),
        cases=['long'] if options.long else options.cases, long=options.long, selected=options.selected,
        stress=0 if options.long else options.stress, cpus=None if options.long else options.cpus,
        cpu_threshold_cores=1, sample_seconds=SAMPLE_SECONDS))
    if (options.long or options.wait_idle) and not wait_idle(time.monotonic()+MAX_WAIT_SECONDS, root, 'B12'):
        return
    try:
        cases = ['long'] if options.long else options.cases
        for case in cases:
            execute(case, root, LONG_END-START if options.long else options.duration,
                    options.selected if options.long else case,
                    0 if options.long else options.stress, None if options.long else options.cpus)
            if not options.long:
                from scripts.analyze_live_b12 import compare
                compare(options.reference, root/case, START, START+options.duration)
        save(root/'status.json', dict(state='finished', pid=os.getpid(), cases=cases))
    except Exception as error:
        save(root/'status.json', dict(state='failed', pid=os.getpid(), error=str(error)))
        raise


if __name__ == '__main__':
    main()
