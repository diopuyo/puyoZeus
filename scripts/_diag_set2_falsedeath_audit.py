"""再生原票を固定勝敗と照合し、正しい確定の消失・遅延も記録する。"""
from __future__ import annotations

import json
import argparse
import gzip
from pathlib import Path
from typing import Any

import numpy as np

from scripts._diag_set2_falsedeath_batch import ROOT, OUT, SOURCES

CERTAIN = ('unavoidable_death', 'confirmed_death')
SETS = (
    ('set1', [f'c{i}' for i in range(1, 7)] + ['zenchi'], 'labels.json', 'replay_cli/prod'),
    ('set2', [f's{i}' for i in range(7)], 'set2/labels.json', 'set2/replay/prod'))


def read(path: Path) -> Any:
    """UTF-8 BOM付き原票にも対応する。"""
    return json.loads(path.read_text(encoding='utf-8-sig'))


def events(path: Path) -> list[dict]:
    """JSONLを読む。"""
    raw = path.read_text() if path.exists() else gzip.decompress(path.with_suffix('.jsonl.gz').read_bytes()).decode()
    return [json.loads(line) for line in raw.splitlines()]


def scan(paths: list[Path], games: list[dict]) -> dict:
    """既報と同じイベント母数と、担当表示区間の誤表示を分けて数える。"""
    cases, wrong_frames, all_confirmed = {}, [], set()
    for directory in paths:
        with np.load(directory / 'display.npz') as data:
            begin, end = float(data['t_sec'][0]), float(data['t_sec'][-1])
            for game in games:
                mask = (data['t_sec'] >= game['start']) & (data['t_sec'] < game['end'])
                certain = mask & np.isin(data['source'], CERTAIN)
                losing = data['display_p1'] < .5 if game['winner'] == '1P' else data['display_p1'] > .5
                wrong = data['t_sec'][certain & losing]
                if len(wrong):
                    wrong_frames.append(dict(game=game['game'], count=len(wrong), first=float(wrong[0])))
                if np.any(certain & ~losing):
                    all_confirmed.add(game['game'])
        for event in events(directory / 'events.jsonl'):
            for value in event['values']:
                stamp = value['t_sec']
                if value['source'] != 'unavoidable_death':
                    continue
                game = next((g for g in games if g['start'] <= stamp < g['end']), None)
                if game is None:
                    continue
                for side in value['dead_sides']:
                    key = (game['game'], side)
                    row = dict(game=key[0], side=side, first_sec=stamp, false=side == game['winner'],
                               in_display_span=begin <= stamp <= end)
                    if key not in cases or stamp < cases[key]['first_sec']:
                        cases[key] = row
    return dict(rows=list(cases.values()), cases=len(cases), false=sum(r['false'] for r in cases.values()),
                false_frames=sum(r['count'] for r in wrong_frames), wrong_frames=wrong_frames,
                correct_confirmed_games=sorted(all_confirmed))


def difference(before: dict, after: dict) -> dict:
    """試合・側を固定し、単純な総数差で入れ替わりを隠さない。"""
    keys = lambda rows: {(r['game'], r['side']): r for r in rows}
    old, new = keys(before['rows']), keys(after['rows'])
    true = {k for k, r in old.items() if not r['false']}
    retained = true & new.keys()
    return dict(before=before, after=after, correct_before=len(true), correct_retained=len(retained),
                correct_lost=[old[k] for k in sorted(true - new.keys())],
                added=[new[k] for k in sorted(new.keys() - old.keys())],
                delays=[dict(game=k[0], side=k[1], seconds=new[k]['first_sec']-old[k]['first_sec'])
                        for k in sorted(retained) if new[k]['first_sec'] != old[k]['first_sec']])


def existing(mode: str) -> dict:
    """既存の固定ラベル監査器を再利用する。"""
    from scripts import eval_set2_regression_20261001 as regression
    from scripts import report_e35 as auditor
    from src.exchange_event_record import read_records
    labels = regression.fixed_labels()
    auditor.events = lambda source: events(OUT / mode / source / 'events.jsonl')
    auditor.read_records = lambda path: read_records(regression.EXEV / 'logs/pending_expiry/full/records' / path.name)
    rows = [r for source in SOURCES for r in auditor.deaths(source, labels)]
    old = read(ROOT / 'logs/eval_set/set2/REGRESSION.json')['rows']
    result = {}
    for source in SOURCES:
        before = dict(rows=[r for r in old if r['source'] == source])
        after = dict(rows=[r for r in rows if r['source'] == source])
        result[source] = difference(before, after)
    return dict(sources=result, false=sum(r['false'] for r in rows),
                unresolved=sum(r['unresolved'] for r in rows), cases=len(rows))


def prepare_baselines() -> dict:
    """修正案の完了を待たず、基準側の母数を固定する。"""
    path = OUT / 'BASELINE_AUDIT.json'
    if path.exists() and read(path).get('audit_version') == 2:
        return read(path)
    result = dict(audit_version=2)
    for name, parts, labels, reference in SETS:
        games = read(ROOT / 'logs/eval_set' / labels)['games']
        result[name] = scan([ROOT / 'logs/eval_set' / reference / p for p in parts], games)
    result['existing5'] = read(ROOT / 'logs/eval_set/set2/REGRESSION.json')
    path.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding='utf-8')
    return result


def main() -> None:
    """完了済みの全対象だけを集計し、欠測を明示する。"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='attributed')
    mode, result = parser.parse_args().mode, {}
    baseline = prepare_baselines()
    for name, parts, labels, reference in SETS:
        missing = [p for p in parts if not (OUT / mode / p / 'status.json').exists()]
        if missing:
            result[name] = dict(missing=missing)
            continue
        games = read(ROOT / 'logs/eval_set' / labels)['games']
        before = baseline[name]
        after = scan([OUT / mode / p for p in parts], games)
        result[name] = difference(before, after)
    if all((OUT / mode / s / 'status.json').exists() for s in SOURCES):
        result['existing5'] = existing(mode)
    (OUT / f'COUNTERFACTUAL_{mode}.json').write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
