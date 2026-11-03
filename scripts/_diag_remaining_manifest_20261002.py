"""診断成果物の全一覧と最終リソース状態を新規保存する。"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from scripts._diag_remaining_20261002 import ROOT, OUT, MIN_MEMORY, save


def main() -> None:
    """指定先外に生成物がないことをパス一覧で確認する。"""
    memory = dict(line.split(':',1) for line in Path('/proc/meminfo').read_text().splitlines())
    available = int(memory['MemAvailable'].split()[0])*1024
    assert os.getpriority(os.PRIO_PROCESS,0)==19 and available>=MIN_MEMORY
    save('resources_final.json', dict(nice=19, available_bytes=available))
    scripts = sorted((ROOT/'scripts').glob('_diag_remaining_*20261002.py'))
    files = sorted(p for p in OUT.rglob('*') if p.is_file()) + scripts
    rows = ['path\tbytes\tsha256']
    for path in files:
        raw = path.read_bytes()
        rows.append(f'{path.relative_to(ROOT).as_posix()}\t{len(raw)}\t{hashlib.sha256(raw).hexdigest()}')
    rows.append(f'{(OUT/"FILES.tsv").relative_to(ROOT).as_posix()}\t\t')
    with (OUT/'FILES.tsv').open('x',encoding='utf-8') as stream:
        stream.write('\n'.join(rows)+'\n')
    counts: dict[str,int] = {}
    for path in files:
        key = path.parent.name if path.parent != OUT else 'top_level'
        counts[key] = counts.get(key,0)+1
    print(json.dumps(dict(files=len(files)+1, counts=counts, available_gib=available/1024**3),ensure_ascii=False))


if __name__ == '__main__':
    main()
