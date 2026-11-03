"""段2診断と同じ試合等重みで経路別寄与と対bootstrap区間を算出する。"""
from __future__ import annotations

import json
import numpy as np
import pandas as pd
from scripts.gfe_phase_stage2b_train_20261002 import OUT
from scripts.gfe_phase_stage2b_score_20261002 import root
from scripts.gfe_phase_stage2_replay_20261002 import SET1, SET2, EVALSET
from scripts._diag_gfe_stage2_analyze import loss, grouped

GAMES, BOOTSTRAPS, SEED = 115, 10000, 0


def frames(mode: str) -> pd.DataFrame:
    """公式試合窓・表示行・本番経路を前回診断と同じ形式へ展開する。"""
    parts = []
    for set_name, sources, labels in [('set1', SET1, EVALSET/'labels.json'),
                                      ('set2', SET2, EVALSET/'set2/labels.json')]:
        displays = []
        for source in sources:
            with np.load(root('baseline', source)/'display.npz') as a:
                with np.load(root(mode, source)/'display.npz') as b:
                    np.testing.assert_array_equal(a['t_sec'], b['t_sec'])
                    displays.append(pd.DataFrame(dict(t=a['t_sec'], p0=a['display_p1'],
                        p3=b['display_p1'], route=a['source'], route3=b['source'], source=source)))
        display = pd.concat(displays).sort_values('t')
        assert display.t.is_unique
        for game in json.loads(labels.read_text())['games']:
            d = display[(display.t >= game['start']) & (display.t < game['end'])].copy()
            d['set'], d['game'], d['y'] = set_name, game['game'], int(game['winner'] == '1P')
            d['weight'] = 1/(GAMES*len(d))
            parts.append(d)
    result = pd.concat(parts, ignore_index=True)
    result['l0'], result['l3'] = loss(result.p0, result.y), loss(result.p3, result.y)
    result['delta'], result['dp'] = result.l3-result.l0, result.p3-result.p0
    return result


def analyze(metrics: dict) -> dict:
    """全寄与の足し戻しと115試合の同時再標本化を検査する。"""
    result = {}
    for mode in ('V3s', 'R1s'):
        d = frames(mode)
        expected = metrics['decisions'][mode]['overall']['difference']
        assert abs(np.dot(d.delta, d.weight)-expected) < 1e-12
        routes = grouped(d, ['route']).rename(columns={'V3': 'candidate'})
        transitions = grouped(d, ['route', 'route3']).rename(columns={'V3': 'candidate'})
        transitions.to_csv(OUT/f'{mode}_route_transitions.csv', index=False)
        grouped(d, ['set', 'game']).rename(columns={'V3': 'candidate'}).to_csv(OUT/f'{mode}_games.csv', index=False)
        d['contribution'] = d.delta*d.weight
        matrix = d.pivot_table(index=['set', 'game'], columns='route', values='contribution',
                               aggfunc='sum', fill_value=0)*GAMES
        assert len(matrix) == GAMES
        index = np.random.default_rng(SEED).integers(0, GAMES, (BOOTSTRAPS, GAMES))
        intervals = []
        for column in matrix:
            v = matrix[column].to_numpy()
            ci = np.quantile(v[index].mean(axis=1), [.025, .975])
            intervals.append(dict(route=column, lower=ci[0], upper=ci[1]))
        routes = routes.merge(pd.DataFrame(intervals), on='route', validate='one_to_one')
        routes.to_csv(OUT/f'{mode}_routes.csv', index=False)
        result[mode] = dict(frames=len(d), games=GAMES, route_changes=int((d.route != d.route3).sum()),
                            difference=expected, rows=routes.to_dict('records'))
    return result
