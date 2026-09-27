"""E12bと同じ保存入力を独立プロセスで再生し、全指標を集計する。"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
from pathlib import Path
import subprocess
import sys

from scripts import run_e10c_exchange_replay_20260927 as runner
from scripts.replay_exchange_event_20260926 import replay
from scripts.run_e3_exchange_eval_20260926 import SOURCES, save_json

OUT = Path('logs/e13/final')
MAX_WORKERS = 2
MODULE_NAME = 'scripts.run_e13_exchange_replay_20260927'
SOURCE_FILES = ('src/exchange_event_tracker.py', 'src/exchange_event_overlay.py',
                'src/exchange_event_landing.py', 'scripts/visualize_advantage_overlay.py',
                'scripts/review_data_panel.py')


def worker(source: str) -> None:
    """保存観測だけを評価器へ渡し、認識処理を再実行しない。"""
    runner.OUT = OUT
    if source == 'zenchi':
        print(replay(Path('logs/review_zenchi_part3/on_e10c/inputs.jsonl.gz'), OUT / source))
    else:
        runner.worker(source)


def launch(source: str) -> None:
    """動画ごとに終了コードとログを分離する。"""
    with (OUT / f'{source}.log').open('w') as log:
        result = subprocess.run([sys.executable, '-m', MODULE_NAME, '--source', source],
                                stdout=log, stderr=subprocess.STDOUT)
    (OUT / f'{source}.exit').write_text(str(result.returncode))
    result.check_returncode()


def main() -> None:
    """描画用の1枠を残し、再生完了後だけ全指標を保存する。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', choices=['zenchi', *SOURCES])
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.source:
        worker(args.source)
        return
    save_json(OUT / 'source_sha256.json',
              {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in SOURCE_FILES})
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        list(pool.map(launch, ['zenchi', *SOURCES]))
    runner.OUT = OUT
    runner.aggregate()
    save_json(OUT / 'complete.json', dict(complete=True))


if __name__ == '__main__':
    main()
