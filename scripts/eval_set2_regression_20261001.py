"""E19構成を既存5記録の固定ラベルと回帰基準で監査する。"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scripts import report_e35 as auditor
from scripts import report_e36b
from scripts import report_e10b_exchange_20260927 as labels
from scripts import aggregate_e3_exchange_eval_20260926 as metrics
from scripts.eval_set2_score_20261001 import OUT, events, save
from src.exchange_event_record import read_records

EXEV = Path('/mnt/c/Users/ryouj/.codex/worktrees/exev/puyo_analyzer')
SOURCES = ('q_7gc4TgFig','fcXG83vInDY','mia8KCjr52g','review','zenchi')
EXPECTED_CASES = 37
SCENE_START, SCENE_END, DEADLINE, THRESHOLD = 2750., 2775., 2766., .95


def directory(source: str) -> Path:
    """独立CLI再生の出力先。"""
    return OUT/'regression/e19'/source


def fixed_labels() -> list[dict]:
    """従来のWIN・独立判定・補完ラベルを同じ規則で使用する。"""
    games = labels.read(EXEV/'logs/review_zenchi_part3/official_games.json')
    rows = []
    for source in SOURCES:
        if source in ('review','zenchi'):
            windows = games
        else:
            frames, _ = metrics.outcomes(source)
            windows = [dict(start=w['start']/metrics.FPS, end=w['end']/metrics.FPS,
                            winner=w['winner']) for w in frames]
        rows.extend(labels.death_rows(directory(source), source, windows))
    panel = labels.read(EXEV/'logs/e10b/panel_outcomes.json')
    for row in rows:
        if row['winner'] is None:
            match = next((r for r in panel if r['source']==row['source']
                          and r['game_idx']==row['game_idx']), None)
            if match is not None and match['winner'] is not None:
                row.update(winner=match['winner'], false_positive=match['winner']==row['side'])
    return rows


def scene() -> dict:
    """平滑化ONの実表示と、従来採点器のEMA再適用値を併記する。"""
    from scripts.measure_e23_display_20260928 import displayed
    with np.load(directory('review')/'display.npz') as data:
        time, probability = data['t_sec'], data['display_p1']
        hits = time[(time >= SCENE_START)&(time <= SCENE_END)&(probability >= THRESHOLD)]
        direct = float(hits[0]) if len(hits) else None
    time, probability = displayed(directory('review')/'display.npz')
    hits = time[(time >= SCENE_START)&(time <= SCENE_END)&(probability >= THRESHOLD)]
    legacy = float(hits[0]) if len(hits) else None
    return dict(first_display_sec=direct, legacy_metric_sec=legacy,
                passed=legacy is not None and legacy <= DEADLINE)


def main() -> None:
    """37件の母数・誤確定ゼロ・場面・第14試合を検査する。"""
    auditor.events = lambda source: events(directory(source)/'events.jsonl')
    auditor.read_records = lambda path: read_records(EXEV/'logs/pending_expiry/full/records'/path.name)
    existing = fixed_labels()
    rows = [row for source in SOURCES for row in auditor.deaths(source, existing)]
    baseline = json.loads((EXEV/'logs/pending_expiry/e36b_on/DEATH_AUDIT_ALL.json').read_text())['rows']
    keys = lambda items: {(r['source'],r['game'],r['side']) for r in items}
    counts = dict(total=len(rows), false=sum(r['false'] for r in rows),
                  unresolved=sum(r['unresolved'] for r in rows), same_cases=keys(rows)==keys(baseline))
    game14 = report_e36b.game14_false_rows(events(directory('q_7gc4TgFig')/'events.jsonl'))
    target = scene()
    passed = (counts['total']==EXPECTED_CASES and counts['false']==0 and counts['unresolved']==0
              and not game14 and target['passed'])
    result = dict(records=len(SOURCES), deaths=counts, scene=target,
                  game14_false_times=game14, passed=passed, rows=rows)
    save('REGRESSION.json', result)
    print(json.dumps({k:v for k,v in result.items() if k!='rows'}, ensure_ascii=False))


if __name__ == '__main__':
    main()
