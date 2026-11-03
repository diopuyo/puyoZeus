"""レビュー指摘区間の保存記録を無変更で抽出する診断専用CLI。"""
from __future__ import annotations

import csv
import gzip
import json
import os
from pathlib import Path
import subprocess
from typing import Any

ROOT = Path('/mnt/d/puyo_analyzer/review_2026-10-03')
OUT = ROOT / 'diag'
WINDOWS = ((2, 210.0, 224.0), (13, 943.0, 950.0), (26, 1725.0, 1735.0))
MIN_KIB = 6 * 1024 * 1024
PS = '/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe'


def resources() -> dict[str, int]:
    """両OSの空きメモリと優先度を検証する。"""
    mem = dict(s.split(':', 1) for s in Path('/proc/meminfo').read_text().splitlines())
    host = subprocess.check_output([PS, '-NoProfile', '-Command',
        '(Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory'], text=True)
    result = dict(linux_kib=int(mem['MemAvailable'].split()[0]),
                  host_kib=int(host.strip()), nice=os.nice(0))
    assert min(result['linux_kib'], result['host_kib']) >= MIN_KIB
    assert result['nice'] == 10
    return result


def save(name: str, value: Any) -> None:
    """診断用ディレクトリだけへUTF-8で保存する。"""
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def records(path: Path) -> list[dict]:
    """圧縮・非圧縮のJSONLを読む。"""
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt', encoding='utf-8') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def extract(game: int, start: float, end: float, rows: list[dict]) -> dict:
    """対象フレームと関連イベント・盤面入力を原値のまま保存する。"""
    selected = [r for r in rows if int(r['set']) == 1 and int(r['game']) == game
                and start <= float(r['t_sec']) <= end]
    prefix = f'game{game:02}'
    with (OUT / f'{prefix}_frames.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(selected)
    work = ROOT / 'work' / f'set1_game{game:02}'
    events = records(work / 'events.jsonl')
    relevant = [e for e in events if any(start <= v['t_sec'] <= end for v in e['values'])]
    save(f'{prefix}_events.json', relevant)
    inputs = records(work / 'inputs.jsonl.gz')
    save(f'{prefix}_input_schema.json', inputs[:1])
    print(prefix, 'frames', len(selected), 'events', len(relevant), 'input keys', list(inputs[0]))
    for event in relevant:
        print('event', {k: v for k, v in event.items() if k not in ('values', 'chains', 'layers')})
        print('chain keys', [list(c) for c in event['chains']][:1])
        transitions = []
        last = None
        for value in event['values']:
            key = (value['source'], round(value['p1'], 4))
            if start <= value['t_sec'] <= end and key != last:
                transitions.append({k: v for k, v in value.items() if k in (
                    'source', 't_sec', 'p1', 'base_p1', 'gfe_p1', 'incoming',
                    'score_totals', 'predicted_final_scores', 'state_safety',
                    'completion_certain', 'completion_sides', 'death_incoming', 'resolving_send')})
                last = key
        print(json.dumps(transitions, ensure_ascii=False))
    return dict(game=game, start=start, end=end, frames=len(selected), events=len(relevant))


def main() -> None:
    """CSVの3区間を抽出し、根拠記録のスキーマを確認する。"""
    OUT.mkdir(exist_ok=True)
    before = resources()
    with (ROOT / 'frames_comparison.csv').open(encoding='utf-8') as stream:
        rows = list(csv.DictReader(stream))
    counts = [extract(*window, rows) for window in WINDOWS]
    save('extraction.json', dict(windows=counts, before=before, after=resources(),
                                source_frames=len(rows), commit='802efbc'))


if __name__ == '__main__':
    main()
