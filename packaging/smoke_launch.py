"""配布版を実際に起動し、/latest を監視して起動段階ごとの経過時間を測る (スモーク)。

    python packaging/smoke_launch.py <PuyoLive> <puyo_live.json> [--seconds 120]

起動 (Popen) 時点を 0 として、次の時刻を秒で記録する:
  http_ready       : /latest が最初に JSON を返した (OBS ブラウザソースが繋がる状態)
  first_hold       : display.status が初めて観測できた (入力確認中/較正中の表示)
  first_available  : evaluations.practical.availability が 'available' になった (判定が出た)
入力段階 (display.input_status / message) の遷移列も保存する。標準ライブラリのみで動く。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import time
from urllib.error import URLError
from urllib.request import urlopen

POLL_SEC = 0.1
HTTP_TIMEOUT_SEC = 2.0
DEFAULT_SECONDS = 120.0
AFTER_AVAILABLE_SEC = 3.0


def fetch_latest(url: str) -> dict | None:
    try:
        with urlopen(url, timeout=HTTP_TIMEOUT_SEC) as response:
            return json.loads(response.read().decode('utf-8'))
    except (URLError, OSError, ValueError):
        return None


def observe(snapshot: dict, started: float, result: dict, last: list) -> None:
    """1 回分の観測を result へ反映する。遷移 (input_status, message, availability) が変わったときだけ記録。"""
    now = round(time.perf_counter() - started, 2)
    display = snapshot.get('display', {})
    availability = snapshot.get('evaluations', {}).get('practical', {}).get('availability')
    result.setdefault('http_ready', now)
    if display.get('status') is not None:
        result.setdefault('first_hold', now)
    if availability == 'available':
        result.setdefault('first_available', now)
    key = (display.get('input_status'), display.get('message'), availability)
    if key != last[0]:
        last[0] = key
        result['transitions'].append(dict(t=now, input_status=key[0], message=key[1], availability=key[2]))


def stop_tree(process: subprocess.Popen) -> None:
    """子 process (spawn した評価・認識) ごと止める。"""
    subprocess.run(['taskkill', '/PID', str(process.pid), '/T', '/F'], capture_output=True, check=False)
    process.wait(timeout=30)


def run(bundle: Path, config: Path, seconds: float) -> dict:
    command = [str(bundle / 'python' / 'python.exe'), '-m', 'src.phase_j.launcher',
               '--config', str(config), '--require-manifest']
    settings = json.loads(config.read_text(encoding='utf-8'))
    port = settings.get('port', 8765)
    if settings.get('source') == 'video':  # 配布版は video を受け付けない。開発用フラグ (動画デコーダのある環境でのみ動く)
        command.append('--dev-allow-video')
    url = f'http://127.0.0.1:{port}/latest'
    result: dict = dict(command=command, transitions=[])
    last: list = [None]
    started = time.perf_counter()
    log_path = config.parent / 'smoke_stdout.log'  # PIPE だと読み手が居らず満杯で子が止まるためファイルへ
    log = log_path.open('w', encoding='utf-8')
    process = subprocess.Popen(command, cwd=bundle, stdout=log, stderr=subprocess.STDOUT)
    try:
        while time.perf_counter() - started < seconds and process.poll() is None:
            snapshot = fetch_latest(url)
            if snapshot:
                observe(snapshot, started, result, last)
            if 'first_available' in result and time.perf_counter() - started > result['first_available'] + AFTER_AVAILABLE_SEC:
                break
            time.sleep(POLL_SEC)
        result['exit_code_before_stop'] = process.poll()
    finally:
        if process.poll() is None:
            stop_tree(process)
        log.close()
        result['output_tail'] = log_path.read_text(encoding='utf-8', errors='replace')[-2000:]
    result['elapsed_sec'] = round(time.perf_counter() - started, 2)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('bundle', type=Path)
    parser.add_argument('config', type=Path)
    parser.add_argument('--seconds', type=float, default=DEFAULT_SECONDS)
    options = parser.parse_args()
    print(json.dumps(run(options.bundle, options.config, options.seconds), ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
