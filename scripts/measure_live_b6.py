"""低負荷が連続60秒続いてからGPU/CPUライブ測定を直列実行する。"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Callable

LOAD_LIMIT = 1.0
IDLE_SECONDS = 60.0
MAX_WAIT_SECONDS = 3*60*60
POLL_SECONDS = 5.0
START_SEC, END_SEC = 2580, 2700
OUTPUT = Path('logs/live_b6_measurement')


def save(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)


def wait_idle(deadline: float, output: Path, condition: str,
              load: Callable[[], float] = lambda: os.getloadavg()[0],
              clock: Callable[[], float] = time.monotonic,
              sleep: Callable[[float], None] = time.sleep) -> bool:
    """連続低負荷時間は高負荷でリセット。3時間後は待たずに理由を保存する。"""
    since = None
    while True:
        now, value = clock(), load()
        since = (now if since is None else since) if value < LOAD_LIMIT else None
        idle = 0.0 if since is None else now-since
        status = dict(pid=os.getpid(), condition=condition, state='waiting', loadavg=value,
                      continuous_idle_seconds=idle, remaining_seconds=max(0, deadline-now))
        with (output / 'wait_history.jsonl').open('a', encoding='utf-8') as history:
            history.write(json.dumps(dict(status, monotonic=now, unix=time.time()))+'\n')
        if now >= deadline:
            save(output / 'status.json', dict(status, state='skipped', reason='3時間以内に低負荷60秒が成立しなかった'))
            return False
        if idle >= IDLE_SECONDS:
            save(output / 'status.json', dict(status, state='running'))
            return True
        save(output / 'status.json', status)
        sleep(min(POLL_SECONDS, deadline-now))


def command(condition: str, output: Path) -> list[str]:
    return [sys.executable, '-m', 'scripts.run_live_pipeline_20260928',
        '--source', 'video', '--config', 'config/live_video.example.json', '--realtime',
        '--start-sec', str(START_SEC), '--end-sec', str(END_SEC), '--warmup-sec', '1',
        '--cnn-device', 'cpu' if condition == 'cpu' else 'auto', '--mc-rollouts', '30',
        '--output', str(output / condition), '--port', '0']


def run_condition(condition: str, output: Path) -> dict:
    env = dict(os.environ)
    if condition == 'cpu':
        env['CUDA_VISIBLE_DEVICES'] = ''
    else:
        env.pop('CUDA_VISIBLE_DEVICES', None)
    args = command(condition, output)
    with (output / f'{condition}.log').open('w') as stream:
        result = subprocess.run(args, env=env, stdout=stream, stderr=subprocess.STDOUT)
    if result.returncode:
        return dict(state='failed', returncode=result.returncode, command=args)
    metrics = json.loads((output / condition / 'metrics.json').read_text())
    valid_device = bool(metrics['gpu']) if condition == 'gpu' else metrics['gpu'] is None
    return dict(state='complete' if valid_device else 'invalid_device', gpu=metrics['gpu'], command=args,
        capture_to_sse_ms=metrics['latency_ms']['capture_to_sse'],
        dropped=metrics['capture_dropped_in_measured_window'], expected=metrics['expected_frames'],
        dropped_fraction=metrics['capture_drop_fraction'], gated_frames=metrics['gated_frames'],
        queue_maximum=metrics['evaluation_queue']['maximum_including_batch'],
        queue_distribution=metrics['evaluation_queue']['distribution'],
        queue_end=metrics['evaluation_queue']['end'], loadavg_start=metrics['loadavg_start'],
        loadavg_end=metrics['loadavg_end'])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    options = parser.parse_args()
    output = options.output
    output.mkdir(parents=True, exist_ok=True)
    # WSLのプロセス群は同一sessionで一つずつ起動する。
    if os.nice(0) != 0:
        raise RuntimeError('nice 0で起動してください')
    deadline, results = time.monotonic()+MAX_WAIT_SECONDS, {}
    save(output / 'launch.json', dict(pid=os.getpid(), started_unix=time.time(),
        nice=os.nice(0), max_wait_seconds=MAX_WAIT_SECONDS, idle_seconds=IDLE_SECONDS,
        parallelism=1, cwd=str(Path.cwd()), python=sys.executable))
    for condition in ('gpu', 'cpu'):
        if not wait_idle(deadline, output, condition):
            results[condition] = dict(state='skipped', reason='低負荷待機の3時間上限')
        else:
            results[condition] = run_condition(condition, output)
        save(output / 'results.json', results)
    save(output / 'status.json', dict(pid=os.getpid(), state='finished', results=results))


if __name__ == '__main__':
    main()
