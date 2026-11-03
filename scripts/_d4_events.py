"""上位区間と3分場面のイベント・盤面・スナップショット差を保存する。"""
from __future__ import annotations
import json
from pathlib import Path
from scripts._d4_loss import OUT, Q, directory, save
from scripts.r1_measure_helpers import lines
from scripts._d3_inventory import decode

SCENE_TIMES = (2749., 2750.85, 2751.4, 2752., 2754., 2756., 2759.05, 2762., 2771.25)


def details(source: str, times: list[float]) -> list[dict]:
    """各指定時刻以前の最終評価と、その時点の盤面を並べる。"""
    result = [dict(t=t) for t in times]
    for mode in ('off', 'on'):
        events = list(lines(directory(mode, source)/'events.jsonl'))
        prefire = json.loads((directory(mode, source)/'prefire_audit.json').read_text())['rows']
        root = Path('logs/e31' if mode == 'off' else 'logs/r1')
        updates = [decode(r)['args'] for r in lines(root/'records'/f'{source}.jsonl.gz') if r['kind']=='update']
        for output in result:
            t = output['t']
            args = min(updates, key=lambda r: abs(r[3]-t))
            relevant = [e for e in events if e['game_idx']==args[4] and e['trigger_sec']<=t]
            event = relevant[-1] if relevant else None
            values = [] if not event else [v for v in event['values'] if v['t_sec']<=t]
            latest = {}
            for v in values:
                latest[v['source']] = v
            output[mode] = dict(input=args, event=event, latest=latest,
                prefire=[r for r in prefire if event and r['game']==args[4] and
                    r['chain_id'] in [c['chain_id'] for c in event['chains']]])
    return result


def main() -> None:
    """区間の最大悪化点と、レビュー時系列を固定する。"""
    top = json.loads((OUT/'q_top10_seconds.json').read_text())
    q = details(Q, [r['peak']['t'] for r in top])
    scene = details('review', list(SCENE_TIMES))
    save('q_top10_inputs.json', q)
    save('scene_inputs.json', scene)
    for title, rows in (('q', q), ('scene', scene)):
        for row in rows:
            print(title, row['t'], {m: row[m]['latest'].get('S3_landing') for m in ('off','on')}, flush=True)


if __name__ == '__main__':
    main()
