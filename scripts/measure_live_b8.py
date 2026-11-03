"""低負荷60秒を待ち、認識優先のCPU条件を一度だけdetach測定する。"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import time

from scripts.measure_live_b6 import IDLE_SECONDS, MAX_WAIT_SECONDS, command, save, wait_idle
from scripts.diagnose_live_b8 import sample_system, POLL_SEC

OUTPUT = Path('logs/live_b8_measurement')
CPU_THREADS = 1
EVALUATION_NICE = 10
P95_TARGET_MS = 500.


def measurement_command(output: Path) -> list[str]:
    return command('cpu', output)+['--cpu-threads', str(CPU_THREADS),
        '--evaluation-nice', str(EVALUATION_NICE), '--recognition-audit']


def run_once(output: Path) -> dict:
    args = measurement_command(output)
    with (output/'cpu.log').open('w') as log, (output/'runtime_load.jsonl').open('w') as history:
        child = subprocess.Popen(args, env=dict(os.environ, CUDA_VISIBLE_DEVICES=''),
                                 stdout=log, stderr=subprocess.STDOUT)
        while child.poll() is None:
            history.write(json.dumps(sample_system())+'\n')
            history.flush()
            time.sleep(POLL_SEC)
    if child.returncode:
        return dict(state='failed', returncode=child.returncode, command=args)
    from scripts.analyze_live_b8 import summarize_case, board_comparison, load_audit, physics_comparison
    result = dict(state='complete', command=args, **summarize_case(output/'cpu'))
    result['runtime_verified'] = verify_runtime(result)
    result['p95_target_met'] = result['capture_to_sse']['P95'] <= P95_TARGET_MS
    if not result['runtime_verified']:
        result['state'] = 'invalid_runtime'
    reference = Path('logs/live_b8_diagnosis/full')
    if (reference/'recognition.npz').exists():
        impact = dict(boards=board_comparison(load_audit(reference/'recognition.npz'),
                                             load_audit(output/'cpu/recognition.npz')),
            physics=physics_comparison(reference/'events.jsonl', output/'cpu/events.jsonl'))
        save(output/'impact.json', impact)
    save(output/'results.json', {'cpu': result})
    return result


def verify_runtime(result: dict) -> bool:
    """CPU限定・指定スレッド数・認識への優先度継承が実測でも成立したか確認する。"""
    if result['gpu'] is not None:
        return False
    for role, nice in (('recognition', 0), ('evaluation', EVALUATION_NICE)):
        runtime = result['runtime'].get(role) or {}
        if runtime.get('nice') != nice or any(runtime.get(key) != CPU_THREADS for key in
                ('torch_threads', 'torch_interop_threads', 'opencv_threads')):
            return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--commit', required=True)
    options = parser.parse_args()
    output = options.output
    output.mkdir(parents=True, exist_ok=True)
    if os.nice(0) != 0:
        raise RuntimeError('認識へnice 0を継承するため、起動元もnice 0にしてください')
    deadline = time.monotonic()+MAX_WAIT_SECONDS
    save(output/'launch.json', dict(pid=os.getpid(), started_unix=time.time(), commit=options.commit,
        max_wait_seconds=MAX_WAIT_SECONDS, idle_seconds=IDLE_SECONDS, nice=os.nice(0),
        parallelism=1, command=measurement_command(output), cwd=str(Path.cwd())))
    try:
        result = run_once(output) if wait_idle(deadline, output, 'cpu_priority') else dict(
            state='skipped', reason='低負荷60秒が3時間以内に成立しなかった')
    except Exception as error:
        result = dict(state='failed', error=str(error))
        save(output/'results.json', {'cpu': result})
        save(output/'status.json', dict(pid=os.getpid(), state='failed', results={'cpu': result}))
        raise
    save(output/'results.json', {'cpu': result})
    save(output/'status.json', dict(pid=os.getpid(), state='finished', results={'cpu': result}))


if __name__ == '__main__':
    main()
