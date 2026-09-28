"""低負荷待機後、長時間→異常入力→終了/再起動検収を直列実行する。"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Any, Iterator

from scripts.measure_live_b6 import wait_idle, save, MAX_WAIT_SECONDS, IDLE_SECONDS
from scripts.live_b9_resources import sample, group_members

OUTPUT = Path('logs/live_b9')
START_SEC, END_SEC = 2580.566, 3427.166
FAULT_START, FAULT_END, FAULT_AT = 2600., 2680., 2620.
SAMPLE_SEC, POLL_SEC, CLEANUP_SEC = 5., 1., 20.
STARTUP_TIMEOUT_SEC = 180.
CASE_TIMEOUT_FACTOR = 2.
CASES = ('long', 'stall', 'resolution', 'black', 'other', 'repeat',
         'interrupt', 'restart_interrupt', 'kill_evaluation', 'restart_kill')


def make_other_video(path: Path) -> None:
    """外部素材に依存しない、動く非ぷよ動画を別ファイルとして生成する。"""
    import cv2
    import numpy as np
    fps, seconds, width, height = 30, 22, 320, 180
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'MJPG'), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError('故障用動画を生成できません')
    try:
        for i in range(fps*seconds):
            image = np.zeros((height, width, 3), np.uint8)
            image[:] = (80, 40, 20)
            cv2.circle(image, (i % width, height//2), 25, (0, 220, 200), -1)
            cv2.putText(image, 'NON-PUYO VIDEO', (10, 35), cv2.FONT_HERSHEY_SIMPLEX, .6, (255, 255, 255), 1)
            writer.write(image)
    finally:
        writer.release()


def case_config(name: str, root: Path) -> tuple[list[str], float, float]:
    from src.phase_j.live_device import DeviceConfig
    config = json.loads(Path('config/live_video.example.json').read_text())
    device, seed = DeviceConfig('B9-'+name, 0), DeviceConfig(config['name'], config['index'])
    calibration = json.loads(seed.calibration_path.read_text())
    calibration['device'] = dict(name=device.name, index=0)
    save(device.calibration_path, calibration)
    start, end = (START_SEC, END_SEC) if name == 'long' else (FAULT_START, FAULT_END)
    if name.startswith('restart'):
        end = start+15.
    config.update(name=device.name, start_sec=start, end_sec=end, port=0, cnn_device='cpu',
        runtime_audit=True, recognition_audit=True, realtime=True)
    if name in ('stall', 'resolution', 'black', 'other', 'repeat'):
        config.update(video_fault=name, fault_at_sec=FAULT_AT)
        if name == 'other':
            config['fault_video'] = str(root/'non_puyo.avi')
    path = root/(name+'.json')
    save(path, config)
    return [sys.executable, '-m', 'scripts.run_live_pipeline_20260928',
        '--config', str(path), '--output', str(root/name)], start, end


def inject_exit(name: str, child: subprocess.Popen, path: Path) -> dict | None:
    from scripts.analyze_live_b9 import lines, samples, available
    runtime = lines(path/'runtime.jsonl')
    if not runtime or not any(available(r) for r in samples(path)):
        return None
    if runtime[-1]['progress'].get('t_sec', 0) < FAULT_START+10:
        return None
    target = child.pid if name == 'interrupt' else runtime[-1]['evaluation_pid']
    members = group_members(child.pid)
    if target not in {r['pid'] for r in members}:
        raise RuntimeError('終了対象が検収process group外です')
    sig = signal.SIGINT if name == 'interrupt' else signal.SIGKILL
    os.kill(target, sig)
    return dict(target=target, signal=sig, at=time.perf_counter(), members=members)


def cleanup_audit(group: int) -> dict:
    """自己groupの残留だけを調べ、異常時の後始末を合格扱いにしない。"""
    deadline = time.monotonic()+CLEANUP_SEC
    while True:
        remaining = group_members(group)
        if not remaining or time.monotonic() >= deadline:
            break
        time.sleep(POLL_SEC)
    if remaining:
        try:
            os.killpg(group, signal.SIGKILL)
        except ProcessLookupError:
            pass
    return dict(no_children_remain=not remaining, remaining_before_forced_cleanup=remaining)


@contextmanager
def owned_process(args: list[str], log: Any) -> Iterator[subprocess.Popen]:
    """監視器側の例外でも次のケースへ孤児を持ち越さない。"""
    child = subprocess.Popen(args, start_new_session=True, env=dict(os.environ, CUDA_VISIBLE_DEVICES=''),
                             stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
    try:
        yield child
    except BaseException:
        try:
            os.killpg(child.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        raise
    finally:
        if child.poll() is None:
            os.killpg(child.pid, signal.SIGKILL)
        child.wait()


def execute(name: str, root: Path) -> dict:
    args, start, end = case_config(name, root)
    path = root/name
    path.mkdir(parents=True, exist_ok=True)
    save(root/'status.json', dict(pid=os.getpid(), state='running', case=name))
    deadline = time.monotonic()+STARTUP_TIMEOUT_SEC+(end-start)*CASE_TIMEOUT_FACTOR
    termination, last_sample, timed_out = None, float('-inf'), False
    with (path/'run.log').open('w') as log, (path/'resources.jsonl').open('w') as resources, owned_process(args, log) as child:
        save(path/'launch.json', dict(pid=child.pid, command=args, start=start, end=end))
        while child.poll() is None:
            if time.monotonic()-last_sample >= SAMPLE_SEC:
                resources.write(json.dumps(sample(child.pid))+'\n')
                resources.flush()
                last_sample = time.monotonic()
            if name in ('interrupt', 'kill_evaluation') and termination is None:
                termination = inject_exit(name, child, path)
            if time.monotonic() >= deadline:
                timed_out = True
                os.killpg(child.pid, signal.SIGKILL)
                break
            time.sleep(POLL_SEC)
        child.wait()
    result = dict(returncode=child.returncode, timed_out=timed_out, termination=termination,
                  **cleanup_audit(child.pid))
    expected_exit = name in ('interrupt', 'kill_evaluation')
    result['passed'] = not timed_out and result['no_children_remain'] and (
        termination is not None and child.returncode != 0 if expected_exit else child.returncode == 0)
    if child.returncode == 0:
        from scripts.analyze_live_b9 import report, samples, available
        if name.startswith('restart'):
            result['display_restored'] = any(available(r) for r in samples(path))
            result['passed'] &= result['display_restored']
        else:
            result['analysis'] = report(path, start, end, name != 'long')
            if name != 'long':
                result['passed'] &= result['analysis']['passed']
            else:
                result['needs_review'] = [key for key, value in result['analysis']['trends'].items()
                                          if value['strictly_increasing']]
                result['passed'] &= (result['analysis']['games']['count_matches'] and
                    all(row['first_probability_delay_sec'] is not None for row in result['analysis']['games']['rows'])
                    and not result['needs_review'])
    save(path/'result.json', result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--commit', required=True)
    parser.add_argument('--cases', nargs='+', choices=CASES, default=list(CASES))
    options = parser.parse_args()
    root = options.output
    root.mkdir(parents=True, exist_ok=True)
    if os.nice(0) != 0:
        raise RuntimeError('認識nice 0のためnice 0で起動してください')
    save(root/'launch.json', dict(pid=os.getpid(), commit=options.commit, cases=options.cases,
        started_unix=time.time(), max_wait_seconds=MAX_WAIT_SECONDS, idle_seconds=IDLE_SECONDS))
    if not wait_idle(time.monotonic()+MAX_WAIT_SECONDS, root, 'B9_serial'):
        return
    results = {}
    try:
        make_other_video(root/'non_puyo.avi')
        for name in options.cases:
            try:
                results[name] = execute(name, root)
            except Exception as error:
                results[name] = dict(passed=False, error=str(error))
            save(root/'results.json', results)
    except Exception as error:
        save(root/'status.json', dict(state='failed', error=str(error), results=results))
        raise
    finally:
        (root/'non_puyo.avi').unlink(missing_ok=True)
    save(root/'status.json', dict(pid=os.getpid(), state='finished',
        passed=all(r.get('passed', False) for r in results.values()), results=results))


if __name__ == '__main__':
    main()
