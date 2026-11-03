"""完成ZIPの別展開先で保存映像を再生し、公開DTOと段別時間を保存する。"""
from __future__ import annotations

import sys as _public_release_sys
from pathlib import Path as _PublicReleasePath
_public_release_root = str(_PublicReleasePath(__file__).resolve().parents[0])
if _public_release_root not in _public_release_sys.path:
    _public_release_sys.path.insert(0, _public_release_root)
from src.public_dev_paths import resolve_text as _public_dev_text

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
from urllib.request import urlopen

OUT = Path('D:/puyo_analyzer/packaging/build_20261019a_release')
BUNDLE = Path(_public_dev_text('<work>/build_20261019a_stage/smoke_copy_r2/PuyoLive'))
VERIFY = Path(_public_dev_text('<work>/build_20261019a_stage/verification_r2'))
VIDEO = Path(_public_dev_text('<work>/build_20261019a_stage/smoke_game.avi'))
PORT, DURATION, TIMEOUT = 18819, 120, 300
LOW_SPEC_RATIO = 1.6473854230


def make_video() -> None:
    """保存済み素材から試験区間を作る。配布Pythonは動画デコーダ不要。"""
    import cv2
    VERIFY.mkdir(exist_ok=True)
    assert not VIDEO.exists()
    cap = cv2.VideoCapture('C:/puyo1_tools/soak/media/video_d04.mp4')
    assert cap.isOpened()
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.set(cv2.CAP_PROP_POS_MSEC, 600000)
    writer = cv2.VideoWriter(str(VIDEO), cv2.CAP_OPENCV_MJPEG,
        cv2.VideoWriter_fourcc(*'MJPG'), fps, (1920, 1080))
    assert writer.isOpened()
    for index in range(round(fps*DURATION)):
        ok, frame = cap.read()
        assert ok
        writer.write(cv2.resize(frame, (1920, 1080)))
        if index % round(fps*10) == 0:
            print('video', index/fps, flush=True)
    writer.release()
    cap.release()


def benchmark_guard() -> None:
    """他作業のベンチマークがあれば5分間隔で待つ。自身の検索は除外する。"""
    while True:
        win = subprocess.check_output(['powershell', '-NoProfile', '-Command',
            "Get-Process | Where-Object ProcessName -like '*ama_plain_bench*' | Select-Object -ExpandProperty Id"], text=True)
        linux = subprocess.run(['wsl', '-d', 'Ubuntu', '--', 'pgrep', '-af',
            '[a]ma_plain_benchmark'], capture_output=True, text=True)
        record = dict(time=time.time(), windows=win.strip(), linux=linux.stdout.strip())
        with (VERIFY/'benchmark_guard.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(record)+'\n')
        if not win.strip() and linux.returncode == 1:
            return
        if linux.returncode not in (0, 1):
            raise RuntimeError(linux.stderr)
        print('Waiting for ama benchmark; next check in 5 minutes', flush=True)
        time.sleep(300)


def settings() -> Path:
    values = json.loads((BUNDLE/'puyo_live.json').read_text(encoding='utf-8'))
    assert values['live_v49f'] and values['b3_async'] and values['live_stage_timing']
    values.update(source='video', video_path=str(VIDEO), start_sec=0, end_sec=DURATION,
        port=PORT, output_dir=str(VERIFY/'output'),
        b3_model=str(BUNDLE/'app/models/b3/model.joblib'),
        b3_timing_table=str(BUNDLE/'app/models/b3/timing_table.json'))
    path = VERIFY/'smoke_config.json'
    path.write_text(json.dumps(values, indent=2), encoding='utf-8')
    return path


def observe(process: subprocess.Popen, started: float) -> dict:
    count, available, states = 0, 0, set()
    with (VERIFY/'polls.jsonl').open('x', encoding='utf-8', buffering=1) as stream:
        while process.poll() is None and time.monotonic()-started < TIMEOUT:
            if (OUT/'RESOURCE_STOP.txt').exists():
                raise RuntimeError('空きメモリ6GiB条件を下回ったため停止')
            try:
                with urlopen(f'http://127.0.0.1:{PORT}/latest', timeout=1) as response:
                    value = json.load(response)
                stream.write(json.dumps(dict(at=time.perf_counter(), snapshot=value), ensure_ascii=False)+'\n')
                count += 1
                trial = value.get('evaluations', {}).get('v49f', {})
                available += int(trial.get('p1') is not None)
                states.add(value.get('display', {}).get('status'))
            except (OSError, ValueError):
                pass
            time.sleep(.1)
    return dict(polls=count, available_polls=available, display_states=list(states))


def main() -> None:
    VERIFY.mkdir(exist_ok=True)
    benchmark_guard()
    config = settings()
    command = [str(BUNDLE/'python/python.exe'), '-m', 'src.phase_j.launcher',
        '--config', str(config), '--require-manifest', '--dev-allow-video']
    start = time.monotonic()
    with (VERIFY/'smoke.log').open('x', encoding='utf-8') as log:
        child = subprocess.Popen(command, cwd=BUNDLE/'app',
            env=dict(os.environ, PYTHONIOENCODING='utf-8'), stdout=log, stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NO_WINDOW)
        try:
            result = observe(child, start)
        finally:
            if child.poll() is None:
                subprocess.run(['taskkill', '/PID', str(child.pid), '/T', '/F'], capture_output=True)
            child.wait(timeout=30)
    result.update(exit_code=child.returncode, elapsed_sec=time.monotonic()-start, command=command)
    (VERIFY/'smoke_result.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result), flush=True)
    assert child.returncode == 0 and result['polls'] > 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--make-video', action='store_true')
    if parser.parse_args().make_video:
        make_video()
    else:
        main()
