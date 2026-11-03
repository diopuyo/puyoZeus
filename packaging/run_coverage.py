"""通し試験の被覆: 複数シナリオを配布版で流し、実行時に open されたファイルを全部記録する。

    python packaging/run_coverage.py <PuyoLive> <シナリオ JSON> <出力ディレクトリ> [並列数]

シナリオ JSON: [{"name": "z", "video": "...mp4", "start_sec": 226, "end_sec": 340}, ...]
各シナリオは非 realtime の既存 CLI (`scripts.run_live_pipeline_20260928`) を配布版 Python で実行する。
監査フック (audit_sitecustomize.py) を python/Lib/site-packages/sitecustomize.py へ一時配置し、
終了時 (失敗時も) 必ず除去する。集計は audit_opens.py <出力>/cov <app> <MANIFEST.json>。
配布物 (MANIFEST の対象) は変更しない。
"""
from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import shutil
import subprocess
import sys
import time

PACKAGING_DIR = Path(__file__).resolve().parent
HOOK_NAME = 'sitecustomize.py'
SITE_PACKAGES = Path('Lib/site-packages')
TAIL_CHARS = 1500
BASE_PORT = 8800
WARMUP_SEC = 1


def scenario_command(python: Path, scenario: dict, output: Path, index: int) -> list[str]:
    return [str(python), '-m', 'scripts.run_live_pipeline_20260928', '--video', scenario['video'],
            '--start-sec', str(scenario['start_sec']), '--end-sec', str(scenario['end_sec']),
            '--warmup-sec', str(WARMUP_SEC), '--output', str(output / scenario['name']),
            '--no-split-evaluation', '--no-async-counter', '--port', str(BASE_PORT + index)]


def run_scenario(bundle: Path, scenario: dict, out: Path, index: int) -> dict:
    """1 シナリオを実行し、終了コード・所要秒・ログ末尾を返す。"""
    env = dict(os.environ, PUYO_AUDIT_LOG=str(out / 'audit' / f'cov.{scenario["name"]}'),
               **scenario.get('env', {}))  # シナリオ固有の環境変数 (例: PUYO_CNN_BACKEND=onnx)
    log_path = out / f'{scenario["name"]}.log'
    started = time.perf_counter()
    with log_path.open('w', encoding='utf-8') as log:
        code = subprocess.run(scenario_command(bundle / 'python' / 'python.exe', scenario, out, index),
                              cwd=bundle / 'app', env=env, stdout=log, stderr=subprocess.STDOUT).returncode
    tail = log_path.read_bytes()[-TAIL_CHARS:].decode('utf-8', 'replace')
    return dict(name=scenario['name'], exit_code=code, seconds=round(time.perf_counter() - started, 1),
                start_sec=scenario['start_sec'], end_sec=scenario['end_sec'], log_tail=tail)


def main() -> None:
    bundle, scenarios_path, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    scenarios = json.loads(scenarios_path.read_text(encoding='utf-8'))
    (out / 'audit').mkdir(parents=True, exist_ok=True)
    hook = bundle / 'python' / SITE_PACKAGES / HOOK_NAME
    shutil.copy2(PACKAGING_DIR / 'audit_sitecustomize.py', hook)
    try:
        workers = int(sys.argv[4]) if len(sys.argv) > 4 else 1
        with ThreadPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(lambda item: run_scenario(bundle, item[1], out, item[0]), enumerate(scenarios)))
    finally:
        hook.unlink(missing_ok=True)
    (out / 'coverage_summary.json').write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding='utf-8')
    print(json.dumps([{k: v for k, v in r.items() if k != 'log_tail'} for r in results], ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
