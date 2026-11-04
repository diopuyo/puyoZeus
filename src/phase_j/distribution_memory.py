"""配布アプリ自身のプロセス木だけを5分ごとに軽量なWindows APIで記録する。"""
from __future__ import annotations

import json
from pathlib import Path
import threading
import time

from .live_process_stats import process_tree

MEMORY_INTERVAL_SECONDS = 5 * 60
MAX_MEMORY_LOG_BYTES = 4 * 1024 * 1024


def record_memory(path: Path) -> None:
    rows = process_tree(include_obs=False)
    processes = [dict(pid=row['pid'], parent=row['parent'], name=row['name'],
                      private_bytes=row.get('private_bytes')) for row in rows]
    record = dict(time=time.time(), processes=processes,
                  missing=sum(row['private_bytes'] is None for row in processes))
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = 'w' if path.exists() and path.stat().st_size >= MAX_MEMORY_LOG_BYTES else 'a'
    with path.open(mode, encoding='utf-8') as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + '\n')


def watch_memory(logs: Path, finished: threading.Event) -> None:
    """診断の欠測で配信を止めない。終了イベントで待機を即座に解除する。"""
    while not finished.is_set():
        try:
            record_memory(logs / 'process_memory.jsonl')
        except (OSError, ValueError) as error:
            print(f'process_memory: 欠測 / unavailable: {error}', flush=True)
        if finished.wait(MEMORY_INTERVAL_SECONDS):
            return
