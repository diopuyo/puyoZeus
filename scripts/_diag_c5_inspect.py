"""c5原票の構造を低優先度で読み取り確認する。"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'logs/eval_set/falsedeath_fix'
OUT = BASE / 'diag_c5_2026-10-02'
MIN_MEMORY = 6 * 1024**3
START, END = 1720.0, 1740.0


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
    """元記録の構造と初回死亡証明を表示する。"""
    guard()
    from src.exchange_event_record import read_records
    from scripts._diag_set2_falsedeath_replay import plain
    capture = BASE / 'collect/capture/c5'
    print((capture / 'COMMAND.json').read_text())
    for name in ('settled', 'display'):
        with np.load(capture / f'{name}.npz', allow_pickle=True) as data:
            print(name, [(k, data[k].shape, str(data[k].dtype)) for k in data.files])
    rows = json.loads((BASE / 'v2c/short/c5/post_counter_bound_audit.json').read_text())['rows']
    print('FIRST_DEAD', json.dumps(next(r for r in rows if r['dead']), ensure_ascii=False))
    for row in read_records(BASE / 'collect/records/c5.jsonl.gz'):
        if row['kind'] == 'header':
            print('HEADER', row)
        if row['kind'] == 'update' and row['args'][3] >= 1727.433:
            print('INPUT', json.dumps(plain(row), ensure_ascii=False, default=str))
            break
    for name in ('placement_signal_reconcile.jsonl', 'events.jsonl'):
        with (capture / name).open() as stream:
            row = json.loads(next(stream))
            print(name, str(row)[:6500])


if __name__ == '__main__':
    main()
