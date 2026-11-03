"""fit10動画の直前公開値と、実測応手リンクを分離して照合する。"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from scripts.exchange_review_io import jsonl, save
from scripts.prefire_v4_data import OUT as OLD, VIDEOS, source
from scripts.prefire_v4_fit import features, LARGE_SEND, write_csv
from src.prefire_worker import maximum_attack
from src.scoring import compute_effective_rate

OUT = OLD.parent/'v4_1'
EVIDENCE = Path('docs/verification/prefire_v41')


def statistics(predicted: list[float], actual: list[float]) -> dict:
    p, y = np.asarray(predicted), np.asarray(actual)
    if not len(p):
        return dict(n=0, bias=None, mae=None, correlation=None)
    correlation = float(np.corrcoef(p, y)[0, 1]) if np.std(p)*np.std(y) else None
    return dict(n=len(p), predicted_mean=float(p.mean()), actual_mean=float(y.mean()),
                bias=float((p-y).mean()), mae=float(np.abs(p-y).mean()), correlation=correlation)


def paired() -> tuple[list[dict], list[dict]]:
    import puyo_core
    events = json.loads((OLD/'scoring_events.json').read_text())
    result, missing = [], []
    for video in VIDEOS:
        computed = list(jsonl(OLD/f'compute_{video}.jsonl'))
        by_input = {r['input_sec']: r for r in computed}
        fires = {r['event_id']: r for r in jsonl(source(video)/'exchange_projection_shadow.jsonl')
                 if r.get('kind') == 'event'}
        large = sorted([e for e in events if e['video'] == video and e['actual_attack_net'] >= LARGE_SEND],
                       key=lambda e: e['trigger'])
        frames = [dict(video=video, game=e['game'], t=np.nextafter(e['trigger'], -np.inf)) for e in large]
        for event, frame in zip(large, features(frames, computed)):
            if frame['input_sec'] is None:
                missing.append(dict(event, reason='発火直前に有効な公開値なし'))
                continue
            record, fire = by_input[frame['input_sec']], fires[event['event']]
            side = int(fire['attacker'] == '2P')
            threat = record['sides'][side]
            request = record['request']
            attack = maximum_attack(tuple(request['boards'][side]), tuple(request['known'][side]), puyo_core)
            rate = compute_effective_rate(max(0., request['elapsed_sec']), from_first_move=True)
            row = dict(event, side=side, input_sec=record['input_sec'], ready_sec=record['ready_sec'],
                age_sec=event['trigger']-record['input_sec'], a=threat['attack'], b=threat['returned'],
                quick=threat['response']['quick'], k=threat['response']['k'], hands=threat['hands'],
                a_gross=attack['score']//rate,
                b_gross=threat['response'].get('gross_amount', threat['returned']),
                actual_b_net=max(0., event['actual_return']-request['pending'][1-side]),
                actual_a_net_at_input=max(0., event['actual_attack']-request['pending'][side]),
                request=request, sides=record['sides'])
            if record['ready_sec'] >= event['trigger']:
                raise ValueError('発火後に公開された値を直前値へ流用')
            result.append(row)
        print('直前値照合', video, len(result), len(missing), flush=True)
    return result, missing


def summarize(rows: list[dict]) -> dict:
    comparisons = dict(A=('a_gross', 'actual_attack'), B=('b_gross', 'actual_return'),
                       A_net=('a', 'actual_a_net_at_input'), B_net=('b', 'actual_b_net'))
    result = {key: statistics([r[p] for r in rows], [r[y] for r in rows])
              for key, (p, y) in comparisons.items()}
    result['net_balance'] = statistics([r['a']-r['b'] for r in rows],
                                        [r['actual_a_net_at_input']-r['actual_b_net'] for r in rows])
    result.update(quick_dominates=sum(r['quick'] >= r['b_gross'] for r in rows),
                  k=dict(mean=float(np.mean([r['k'] for r in rows])),
                         minimum=min(r['k'] for r in rows), maximum=max(r['k'] for r in rows)),
                  attack_needs_two_hands=sum(r['hands'] == 2 for r in rows),
                  age_percentiles=np.percentile([r['age_sec'] for r in rows], [50, 95, 100]).tolist(),
                  response_zero=sum(r['actual_return'] == 0 for r in rows))
    return result


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows, missing = paired()
    save(OUT/'diagnostic_pairs.json', rows)
    save(EVIDENCE/'diagnostic_missing.json', missing)
    summary = dict(videos=list(VIDEOS), excluded_video=80, paired=len(rows), missing=len(missing),
                   all=summarize(rows), responding=summarize([r for r in rows if r['actual_return'] > 0]))
    save(OUT/'diagnostic_initial.json', summary)
    write_csv(EVIDENCE/'diagnostic_pairs.csv', [{k: v for k, v in r.items() if k not in ('request', 'sides')}
                                               for r in rows])
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
