"""mia7原票の構造を低優先度で読み取り確認する。"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'logs/eval_set/falsedeath_fix'
OUT = BASE / 'diag_mia7_2026-10-02'
MIN_MEMORY = 6 * 1024**3
START, END = 680.0, 706.0


def guard() -> int:
    """指定資源条件を実行前に確認する。"""
    info = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
    available = int(info['MemAvailable'].split()[0]) * 1024
    assert available >= MIN_MEMORY, available
    assert os.getpriority(os.PRIO_PROCESS, 0) == 10
    return available


def save(name: str, value: Any) -> None:
    """許可された診断出力先だけへ保存する。"""
    OUT.mkdir(exist_ok=True)
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=1), encoding='utf-8')


def main() -> None:
    """対象の入力件数と資源条件を読み取り確認する。"""
    guard()
    from src.exchange_event_record import read_records
    count = sum(row['kind'] == 'update' and START <= row['args'][3] <= END
                for row in read_records(BASE / 'collect/records/mia8KCjr52g.jsonl.gz'))
    print(json.dumps(dict(updates=count, nice=10, available_bytes=guard())))


if __name__ == '__main__':
    main()
