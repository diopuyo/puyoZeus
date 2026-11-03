"""B8の同区間認識比較とCPU配分の一因子診断を直列実行する。"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from scripts.measure_live_b6 import save

OUTPUT = Path('logs/live_b8_diagnosis')
POLL_SEC = 1.0
CASES = (
    ('realtime', True, 2580, 2700, 0, 0),
    ('full', False, 2580, 2700, 0, 0),
    ('base40', True, 2600, 2640, 0, 0),
    ('nice40', True, 2600, 2640, 0, 10),
    ('threads40', True, 2600, 2640, 1, 0),
    ('priority40', True, 2600, 2640, 1, 10),
)


def case_command(case: tuple, output: Path) -> list[str]:
    """検証専用profileを各条件へ同じ内容で種付けし、較正の持ち越しを避ける。"""
    from src.phase_j.live_device import DeviceConfig
    name, realtime, start, end, threads, nice = case
    seed = DeviceConfig('zenchi-video', 0, True).calibration_path
    device = DeviceConfig('B8-'+name, 0, True)
    profile = json.loads(seed.read_text(encoding='utf-8'))
    profile['device'] = dict(name=device.name, index=0)
    save(device.calibration_path, profile)
    config = dict(source='video', name=device.name, index=0, verification_only=True,
        realtime=realtime, start_sec=start, end_sec=end, warmup_sec=1,
        cpu_threads=threads, evaluation_nice=nice, cnn_device='cpu', mc_rollouts=30,
        recognition_audit=True, port=0)
    path = output/(name+'_config.json')
    save(path, config)
    return [sys.executable, '-m', 'scripts.run_live_pipeline_20260928',
            '--config', str(path), '--output', str(output/name)]


def sample_system() -> dict:
    """外部負荷の変動も記録し、前後比較だけで因果を断定しない。"""
    lines = Path('/proc/stat').read_text().splitlines()
    cpu = [int(v) for v in lines[0].split()[1:]]
    return dict(unix=time.time(), loadavg=os.getloadavg(), cpu_ticks=cpu)


def run_case(case: tuple, output: Path) -> None:
    name = case[0]
    args = case_command(case, output)
    save(output/(name+'_command.json'), dict(command=args))
    save(output/'status.json', dict(pid=os.getpid(), state='running', case=name))
    with (output/(name+'.log')).open('w') as log, (output/(name+'_load.jsonl')).open('w') as load:
        process = subprocess.Popen(args, stdout=log, stderr=subprocess.STDOUT)
        while process.poll() is None:
            load.write(json.dumps(sample_system())+'\n')
            load.flush()
            time.sleep(POLL_SEC)
    if process.returncode:
        raise RuntimeError(f'{name} failed: {process.returncode}')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    options = parser.parse_args()
    options.output.mkdir(parents=True, exist_ok=True)
    save(options.output/'launch.json', dict(pid=os.getpid(), started_unix=time.time(), cases=CASES))
    try:
        for case in CASES:
            run_case(case, options.output)
    except Exception as error:
        save(options.output/'status.json', dict(state='failed', error=str(error)))
        raise
    save(options.output/'status.json', dict(state='complete'))


if __name__ == '__main__':
    main()
