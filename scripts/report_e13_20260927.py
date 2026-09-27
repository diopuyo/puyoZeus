"""E12bの固定入力・固定分母でE13の事前登録基準を照合する。"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scripts import report_e10c_exchange_20260927 as death_metrics
from scripts import report_e10b_exchange_20260927 as helpers
from scripts.exchange_value_spikes import value_spikes
from scripts.report_e9_exchange_20260927 import metric_rows
from scripts.run_e3_exchange_eval_20260926 import SOURCES, save_json

OUT = Path('logs/e13')
BASELINE = Path('logs/e12_final')
SPIKE_RATIO = .1
LOSS_TOLERANCE = .005
AGREEMENT_TOLERANCE = .005
MAX_FALSE_DEATH = .05
FLIP_RATIO = 1.1


def spikes(root: Path) -> dict:
    """平滑化前のdisplay_p1列を使い、4入力の時間を加算する。"""
    paths = {'zenchi': root / 'zenchi/display.npz'}
    paths.update({s: root / 'renders' / s / 'on/display.npz' for s in SOURCES})
    rows = {}
    for source, path in paths.items():
        with np.load(path) as display:
            rows[source] = value_spikes(display['t_sec'], display['display_p1'], display['game_idx'])
    duration = sum(r['duration_seconds'] for r in rows.values())
    count = sum(r['count'] for r in rows.values())
    return dict(count=count, duration_seconds=duration, per_minute=count * 60 / duration, sources=rows)


def report() -> dict:
    """全指標と5つの固定ゲートを同じJSONへ保存する。"""
    roots = dict(E12b=BASELINE, E13=OUT / 'final')
    pooled = {v: helpers.read(p / 'v2/pooled.json') for v, p in roots.items()}
    games = helpers.read(helpers.ZENCHI / 'official_games.json')
    agreements = {v: helpers.agreement(p / 'zenchi/display.npz', games) for v, p in roots.items()}
    jumps = {v: spikes(p) for v, p in roots.items()}
    death_metrics.OUT = roots['E13']
    deaths = death_metrics.deaths()
    labelled = [d for d in deaths if d['winner'] is not None]
    false = sum(d['false_positive'] for d in labelled)
    summary = dict(total=len(deaths), false=false, unlabelled=len(deaths)-len(labelled),
                   fraction=false / max(1, len(labelled)))
    before, after = pooled['E12b'], pooled['E13']
    gates = dict(spikes=jumps['E13']['per_minute'] <= jumps['E12b']['per_minute'] * SPIKE_RATIO,
        q_log_loss=after['M3_q']['on']['groups']['all']['log_loss'] <=
            before['M3_q']['on']['groups']['all']['log_loss'] + LOSS_TOLERANCE,
        zenchi=agreements['E13']['agreement'] >= agreements['E12b']['agreement'] - AGREEMENT_TOLERANCE,
        deaths=summary['unlabelled'] == 0 and summary['fraction'] <= MAX_FALSE_DEATH,
        flips=after['M4']['on']['flips'] <= before['M4']['on']['flips'] * FLIP_RATIO)
    metrics = [dict(metric=a, E12b=b, E13=c) for (a, b), (_, c) in
               zip(metric_rows(before), metric_rows(after))]
    return dict(gates=gates, spikes=jumps, zenchi=agreements, metrics=metrics,
                death_summary=dict(E12b=helpers.read(BASELINE / 'report_e12b.json')['death_summary']['E12b'],
                                   E13=summary), deaths=deaths)


if __name__ == '__main__':
    result = report()
    save_json(OUT / 'report.json', result)
    print(json.dumps({k: v for k, v in result.items() if k not in ('deaths', 'spikes')},
                     ensure_ascii=False, indent=2))
