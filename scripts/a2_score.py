"""事前登録の採点器・固定ラベルを変えず、A2の全比較を生成する。"""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np

from scripts.a2_replay import OUT, INPUT
from scripts import falsedeath_score_20261001 as prior
from scripts import falsedeath_v2_score_20261002 as v2
from scripts import falsedeath_v2d_display as display
from scripts.falsedeath_replay_20261001 import records
from scripts._diag_set2_falsedeath_audit import read
from scripts.eval_set2_regression_20261001 import fixed_labels
from scripts import eval_set2_regression_20261001 as regression_inputs

MODES = ('prod', 'a2')
EXPECTED_HISTORY_GAMES = 176
REVIEW = ((1, 'set1', 214., 213., 216.), (2, 'set1', 947.2, 946., 950.),
          (3, 'set1', 1729., 1728., 1733.), (4, 'set1', 2762., 2748., 2763.),
          (5, 'set2', 4244., 4240., 4247.), (6, 'set2', 4272., 4261., 4273.),
          (7, 'set2', 6060., 6049., 6061.))


def histories() -> dict:
    """176記録内試合と、H1が止めた計算確定の呼出し・試合側数を区別する。"""
    result = {}
    for source in records():
        result[source] = {}
        for mode in MODES:
            base = OUT/'replay'/mode/source
            audit = read(base/'history_integrity_audit.json')
            blocked = [r for r in audit['rows'] if r['blocked']]
            with np.load(base/'display.npz') as data:
                first, last = data['t_sec'][[0, -1]]
            visible = [r for r in blocked if first <= r['t_sec'] <= last]
            result[source][mode] = dict(games=len(audit['games']),
                incomplete=sum(not r['history_complete'] for r in audit['games']),
                blocked_calls=len(blocked), display_blocked_calls=len(visible),
                blocked_game_sides=len({(r['game'], s) for r in blocked for s in r['blocked_dead_sides']}),
                elapsed=read(base/'status.json')['elapsed_seconds'],
                incomplete_games=[r['game'] for r in audit['games'] if not r['history_complete']])
        assert result[source]['prod']['games'] == result[source]['a2']['games'], source
    assert sum(r['prod']['games'] for r in result.values()) == EXPECTED_HISTORY_GAMES
    return result


def frames(terminal: dict, labels: list[dict]) -> dict:
    """全18記録の表示誤確定数。重複するzenchiの再加算はしない。"""
    result = {}
    for source in records():
        result[source] = {}
        for mode in MODES:
            if source in prior.SOURCES:
                row = display.scan(source, mode, labels, terminal[source])
            else:
                group = 'set1' if source.startswith('c') else 'set2'
                games = read(prior.SETS[group][1])['games']
                row = v2.visible_scan([OUT/'replay'/mode/source], games)
            result[source][mode] = row
    return result


def review_values() -> list[dict]:
    """レビュー区間の表示値・評価器・計算確定時刻を同じ秒で比較する。"""
    result = []
    for number, group, target, start, end in REVIEW:
        row = dict(case=number, target=target, start=start, end=end)
        for mode in MODES:
            parts = prior.SETS[group][0]
            data = review_display(mode, parts)
            index = int(np.argmin(abs(data['t_sec']-target)))
            mask = (data['t_sec'] >= start) & (data['t_sec'] <= end)
            death = mask & (data['source'] == 'unavoidable_death')
            row[mode] = dict(t=float(data['t_sec'][index]), p1=float(data['display_p1'][index]),
                source=str(data['source'][index]), minimum=float(data['display_p1'][mask].min()),
                maximum=float(data['display_p1'][mask].max()),
                first_computed_death=float(data['t_sec'][death][0]) if np.any(death) else None)
        result.append(row)
    return result


def review_display(mode: str, parts: list[str]) -> dict:
    """採点器の時刻順を維持し、レビューに必要な評価器名も読む。"""
    columns = ('t_sec', 'display_p1', 'source')
    rows = []
    for part in parts:
        with np.load(OUT/'replay'/mode/part/'display.npz') as data:
            rows.append({k: data[k] for k in columns})
    values = {k: np.concatenate([r[k] for r in rows]) for k in columns}
    order = np.argsort(values['t_sec'], kind='stable')
    return {k: v[order] for k, v in values.items()}


def main() -> None:
    """既存採点の入出力ルートだけ差し替え、門の閾値は固定する。"""
    prior.OUT = display.OUT = OUT
    regression_inputs.OUT = prior.ROOT/'logs/eval_set/set2'
    prior.input_record = lambda mode, source: INPUT/'collect/records'/f'{source}.jsonl.gz'
    prior.scan, prior.events = v2.visible_scan, v2.visible_events
    v2.MODE = 'a2'
    terminal = read(INPUT/'TERMINALS.json')['all']
    sets = {name: prior.score_set(name, 'a2', terminal) for name in prior.SETS}
    existing = prior.existing('a2', terminal)
    merged = v2.combined(sets)
    regression = {mode: prior.regression(mode) for mode in MODES}
    audits = frames(terminal, fixed_labels())
    no_false_events = existing['false'] == existing['unresolved'] == 0 and all(
        row['deaths']['after']['false'] == 0 for row in sets.values())
    gates = dict(**v2.metric_gates(merged), false_death=no_false_events and all(
        r['a2']['false_frames'] == 0 and r['a2'].get('unknown_frames', 0) == 0 for r in audits.values()),
        regression=regression['a2']['passed'])
    result = dict(sets=sets, existing5=existing, combined=merged, regression=regression,
                  display_audit=audits, history=histories(), review=review_values(),
                  gates=gates, passed=all(gates.values()))
    (OUT/'RESULT.json').write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding='utf-8')
    print(json.dumps(dict(gates=gates, combined=merged), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
