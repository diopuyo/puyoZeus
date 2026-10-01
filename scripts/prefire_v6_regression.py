"""既存5記録の独立ラベルで誤確定件数を採点する。時刻一致は合否条件に加えない。"""
from __future__ import annotations

import json

import numpy as np

from scripts import report_e35 as auditor
from scripts import report_e36b
from scripts import report_e10b_exchange_20260927 as labels
from scripts import aggregate_e3_exchange_eval_20260926 as metrics
from scripts import prefire_gate_report_20260930 as legacy
from scripts.prefire_v6_replay import OUT, EVALSET
from scripts.run_prefire_replay_20260930 import EXEV, BASELINE_DIRS
from src.exchange_event_record import read_records


def fixed_labels() -> list[dict]:
    """セット2検証で使った37件と同じWIN・独立判定・補完ラベルを適用する。"""
    games = labels.read(EXEV/'logs/review_zenchi_part3/official_games.json')
    rows = []
    for source in BASELINE_DIRS:
        if source in ('review', 'zenchi'):
            windows = games
        else:
            frames, _ = metrics.outcomes(source)
            windows = [dict(start=w['start']/metrics.FPS, end=w['end']/metrics.FPS,
                            winner=w['winner']) for w in frames]
        rows.extend(labels.death_rows(OUT/'replay'/source, source, windows))
    panel = labels.read(EXEV/'logs/e10b/panel_outcomes.json')
    for row in rows:
        if row['winner'] is None:
            match = next((r for r in panel if r['source']==row['source']
                          and r['game_idx']==row['game_idx']), None)
            if match is not None and match['winner'] is not None:
                row.update(winner=match['winner'], false_positive=match['winner']==row['side'])
    return rows


def score() -> dict:
    """実際の確定を数える。増えた正常な確定や時刻の差だけでは不合格にしない。"""
    from scripts.prefire_v6_score import events, regression_predictions
    old_events, old_reader = auditor.events, auditor.read_records
    auditor.events = lambda source: events(OUT/'replay'/source)
    auditor.read_records = lambda path: read_records(EXEV/'logs/pending_expiry/full/records'/path.name)
    try:
        existing = fixed_labels()
        rows = [row for source in BASELINE_DIRS for row in auditor.deaths(source, existing)]
    finally:
        auditor.events, auditor.read_records = old_events, old_reader
    baseline = json.loads((EVALSET/'set2/REGRESSION.json').read_text())['deaths']
    predictions = regression_predictions()
    existing_false = {(row['source'], row['game'], row['side']) for row in rows if row['false']}
    added = len(predictions-existing_false)
    false = len(existing_false | predictions)
    unresolved = sum(row['unresolved'] for row in rows)
    if unresolved:
        raise ValueError(f'誤確定のラベル未解決: {unresolved}')
    target = legacy.scene_first(OUT/'replay/review/display.npz')
    game14 = report_e36b.game14_false_rows(events(OUT/'replay/q_7gc4TgFig'))
    with np.load(OUT/'replay/q_7gc4TgFig/prefire_v6_features.npz') as data:
        columns = list(data['columns'])
        with np.load(OUT/'replay/q_7gc4TgFig/display.npz') as shown:
            visible = np.isin(data['values'][:,0], shown['t_sec'])
        selected = visible & (data['values'][:,1] == 14) & (data['values'][:,columns.index('proof_2p')] > 0)
        predicted_game14 = data['values'][selected,0].tolist()
    return dict(reference_deaths=baseline, cases=len(rows), false_deaths=false, rows=rows,
        added_prediction_false_deaths=added, scene_first_sec=target, game14=game14+predicted_game14,
        passed=false <= baseline['false'] and target is not None and target <= 2766.
               and not game14 and not predicted_game14)
