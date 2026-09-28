"""3動画の新規・前倒し死亡判定を、保存された死亡観測と固定勝敗へ照合する。"""
from __future__ import annotations

import csv
import json
from pathlib import Path

from scripts.run_e3_exchange_eval_20260926 import SOURCES, save_json
from src.exchange_event_record import read_records

OUT = Path('logs/e23')


def observed_deaths() -> dict:
    """ばたんきゅー信号の初時刻を保持し、未観測を生存と決めつけない。"""
    observed = {}
    for source in SOURCES:
        for item in read_records(Path('logs/e16/records')/f'{source}.jsonl.gz'):
            if item['kind'] != 'update':
                continue
            result, _, _, t_sec, game_idx, *_ = item['args']
            for side in getattr(result, 'confirmed_dead_sides', ()):
                observed.setdefault((source, game_idx, side), t_sec)
    return observed


def main() -> None:
    """原票一覧と判定不能件数を出し、最終勝者だけの判定との違いも残す。"""
    audit = json.loads((OUT/'DEATH_AUDIT.json').read_text())
    observed, cases = observed_deaths(), []
    for row in audit['three_videos_changed']:
        side = row['side']
        other = '2P' if side == '1P' else '1P'
        own = observed.get((row['source'], row['game_idx'], side))
        opponent = observed.get((row['source'], row['game_idx'], other))
        died = True if own is not None else False if opponent is not None else (
            row['winner'] != side if row['winner'] is not None else None)
        cases.append(dict(source=row['source'], game_idx=row['game_idx'], exchange_id=row['exchange_id'],
            side=side, change=row['change'], first_sec=row['first_sec'],
            baseline_first_sec=row['baseline_first_sec'], winner=row['winner'],
            observed_death_sec=own, observed_opponent_death_sec=opponent,
            actually_died=died, error=died is False,
            evidence='terminal_signal' if own is not None or opponent is not None else 'fixed_winner'))
    cases.sort(key=lambda r: (SOURCES.index(r['source']), r['first_sec'], r['side']))
    summary = dict(cases=len(cases), new=sum(r['change'] == 'new' for r in cases),
        earlier=sum(r['change'] == 'earlier' for r in cases),
        actually_survived=sum(r['actually_died'] is False for r in cases),
        unresolved=sum(r['actually_died'] is None for r in cases))
    save_json(OUT/'NEW_DEATH_CASES.json', dict(summary=summary, cases=cases))
    if cases:
        with (OUT/'NEW_DEATH_CASES.csv').open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(cases[0]))
            writer.writeheader()
            writer.writerows(cases)
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == '__main__':
    main()
