"""レビュー指摘4〜7の原記録を診断領域だけへ抽出する。"""
from __future__ import annotations

import csv
import gzip
import json
import os
from pathlib import Path
import subprocess
from typing import Any

ROOT = Path('/mnt/d/puyo_analyzer/review_2026-10-03')
OUT = ROOT / 'diag2'
CASES = ((4, 1, 43, 2748., 2763.), (5, 2, 11, 4240., 4247.),
         (6, 2, 11, 4261., 4273.), (7, 2, 41, 6049., 6061.))
MIN_KIB = 6 * 1024 * 1024
PS = '/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe'


def resources() -> dict[str, int]:
    """ホスト・WSLの空きメモリとniceを検証する。"""
    mem = dict(s.split(':', 1) for s in Path('/proc/meminfo').read_text().splitlines())
    host = subprocess.check_output([PS, '-NoProfile', '-Command',
        '(Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory'], text=True)
    result = dict(linux_kib=int(mem['MemAvailable'].split()[0]),
                  host_kib=int(host.strip()), nice=os.nice(0))
    assert min(result['linux_kib'], result['host_kib']) >= MIN_KIB, result
    assert result['nice'] == 10, result
    return result


def save(name: str, value: Any) -> None:
    """UTF-8診断原票を出力する。"""
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def records(path: Path) -> list[dict[str, Any]]:
    """JSONLを読み込む。"""
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt', encoding='utf-8') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def extract(case: tuple, rows: list[dict[str, str]]) -> dict[str, Any]:
    """対象時刻のCSVと関連イベントを抽出する。"""
    number, set_no, game, start, end = case
    selected = [r for r in rows if int(r['set']) == set_no and int(r['game']) == game
                and start <= float(r['t_sec']) <= end]
    with (OUT / f'case{number}_frames.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(selected)
    work = ROOT / 'work' / f'set{set_no}_game{game:02}'
    events = records(work / 'events.jsonl')
    relevant = [e for e in events if any(start <= v['t_sec'] <= end for v in e['values'])]
    save(f'case{number}_events.json', relevant)
    old = Path('/mnt/d/puyo_analyzer/wt_evalset/logs/eval_set/falsedeath_fix/replay/prod')
    old_events = records(old / selected[0]['source_record'] / 'events.jsonl')
    save(f'case{number}_old_events.json', [e for e in old_events
         if any(start <= v['t_sec'] <= end for v in e['values'])])
    print(number, 'frames', len(selected), 'event ids', [e['exchange_id'] for e in relevant])
    for event in relevant:
        print('chains', [{k: v for k, v in c.items() if 'board' not in k} for c in event['chains']])
        values = [v for v in event['values'] if start <= v['t_sec'] <= end]
        print('value keys', sorted(set(k for v in values for k in v)))
    return dict(case=number, frames=len(selected), events=len(relevant), start=start, end=end)


def main() -> None:
    """リソース条件を検証して抽出する。"""
    OUT.mkdir(exist_ok=True)
    before = resources()
    with (ROOT / 'frames_comparison.csv').open(encoding='utf-8-sig') as stream:
        rows = list(csv.DictReader(stream))
    counts = [extract(case, rows) for case in CASES]
    save('extraction.json', dict(cases=counts, before=before, after=resources(),
                                source_frames=len(rows), commit='802efbc'))


if __name__ == '__main__':
    main()
