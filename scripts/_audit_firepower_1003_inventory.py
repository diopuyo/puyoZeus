"""火力監査用の読み取り専用インベントリ。"""
from __future__ import annotations

import gzip
import json
import os
from pathlib import Path
from typing import Any

ROOT = Path('/mnt/d/puyo_analyzer/wt_evalset/logs/eval_set/falsedeath_fix')
OUT = Path('/mnt/d/puyo_analyzer/review_2026-10-03/fire_audit')
MIN_KIB = 6 * 1024**2


def resources() -> dict[str, Any]:
    """優先度と空きメモリを確認する。"""
    mem = dict(x.split(':', 1) for x in Path('/proc/meminfo').read_text().splitlines())
    result = dict(nice=os.getpriority(os.PRIO_PROCESS, 0),
                  available_kib=int(mem['MemAvailable'].split()[0]))
    assert result['nice'] == 10 and result['available_kib'] >= MIN_KIB, result
    return result


def main() -> None:
    """長大な原票は先頭構造だけ表示する。"""
    import numpy as np
    OUT.mkdir(parents=True, exist_ok=True)
    print(resources())
    for name in ('c1', 's0'):
        folder = ROOT / 'replay/v2h2b' / name
        event = json.loads(next((folder / 'events.jsonl').open()))
        print(name, 'event', {k: v for k, v in event.items() if k != 'values'})
        print('firstvalues', event['values'][:2])
        with np.load(folder / 'display.npz') as data:
            print('display', {k: (v.shape, str(v.dtype), v.reshape(-1)[:3].tolist()) for k, v in data.items()})
        with gzip.open(ROOT / 'collect/records' / f'{name}.jsonl.gz', 'rt') as stream:
            print('header', next(stream).strip())
            print('record', next(stream).strip()[:13000])
        for fname in ('prefire_audit.json', 'midchain_audit.json'):
            value = json.loads((folder / fname).read_text())
            print(fname, str(value)[:2500])
    for p in Path('logs/eval_set').rglob('*games*.json'):
        print('games', p, p.read_text()[:500])


if __name__ == '__main__':
    main()

