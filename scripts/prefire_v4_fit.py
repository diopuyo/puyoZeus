"""試合grouped CV。採点用の勝敗・実際の発火はworkerへ渡さない。"""
from __future__ import annotations
from bisect import bisect_right
import csv
import json
from pathlib import Path
import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.special import expit, logit
from sklearn.model_selection import GroupKFold
from scripts.prefire_v4_data import OUT, EVIDENCE, VIDEOS, source, run_path, SAMPLE_SEC, publication_fps
from scripts.exchange_review_io import jsonl, pickles, save
from scripts.exchange_review_v2_fit import samples
from scripts.exchange_review_labels import corrected_winners
from src.prefire_display import EPS, COUNT_UNIT

FOLDS, PRE_SECONDS, LARGE_SEND, MOVE_POINTS = 5, 5., 30., 2.
TAUS = (.5, 1., 2.)


def load_video(video: int) -> tuple[list[dict], list[dict]]:
    computed = list(jsonl(OUT/f'compute_{video}.jsonl'))
    wins, frames, seen = corrected_winners(run_path(video)), [], set()
    fps = publication_fps(source(video))
    for row in pickles(source(video)/'publications.pickle'):
        game = row.get('public_game', row['game'])
        stamp = row['frame']/fps
        key = (game, int(stamp/SAMPLE_SEC))
        if key in seen or game not in wins or row.get('probability') is None:
            continue
        seen.add(key)
        frames.append(dict(video=video, game=game, t=stamp, b3=row['probability'], win=int(wins[game] == '1P'),
            terminal=row.get('source') in ('confirmed_death', 'unavoidable_death')))
    return sorted(frames, key=lambda r: r['t']), computed


def features(frames: list[dict], computed: list[dict]) -> list[dict]:
    cursor, current, previous_game, generation = 0, None, None, None
    messages = [(r['input_sec'], 0, r) for r in computed]
    messages += [(r['ready_sec'], 1, r) for r in computed if 'sides' in r and not r['over_budget']]
    ready = sorted(messages, key=lambda r: r[:2])
    result = []
    for row in frames:
        if row['game'] != previous_game:
            current = None
            previous_game = row['game']
        while cursor < len(ready) and ready[cursor][0] <= row['t']:
            _, kind, event = ready[cursor]
            if kind == 0:
                generation = event['input_sec']
                if 'sides' not in event:
                    current = None
            elif event['input_sec'] == generation:
                current = event
            cursor += 1
        valid = current is not None and current['game'] == row['game']
        sides = current['sides'] if valid else [dict(attack=0., returned=0.)]*2
        result.append(dict(row, a=(sides[0]['attack']-sides[1]['attack'])/COUNT_UNIT,
            b=(sides[0]['returned']-sides[1]['returned'])/COUNT_UNIT,
            ready_sec=current['ready_sec'] if valid else None,
            input_sec=current['input_sec'] if valid else None))
    return result


def loss(y: np.ndarray, p: np.ndarray, weights: np.ndarray | None = None) -> float:
    p = np.clip(p, EPS, 1.-EPS)
    return float(np.average(-y*np.log(p)-(1-y)*np.log1p(-p), weights=weights))


def fit(rows: list[dict], corrected: bool) -> dict:
    rows = [r for r in rows if not r.get('terminal', False)]
    base, y = np.array([r['b3'] for r in rows]), np.array([r['win'] for r in rows])
    x = np.array([[r['a'], -r['b']] for r in rows])
    offset = logit(np.clip(base, EPS, 1.-EPS))
    keys = [(r['video'], r['game']) for r in rows]
    from collections import Counter
    sizes = Counter(keys)
    weights = np.array([1./sizes[k] for k in keys])
    def objective(beta: np.ndarray) -> float:
        slopes = beta if corrected else np.repeat(beta, 2)
        return loss(y, expit(offset+x@slopes), weights)
    fitted = minimize(objective, np.zeros(2 if corrected else 1), bounds=[(0., None)]*(2 if corrected else 1))
    if not fitted.success:
        raise RuntimeError(fitted.message)
    beta = fitted.x if corrected else np.repeat(fitted.x, 2)
    delta = expit(x@beta)-.5
    result = minimize_scalar(lambda c: loss(y, base+c*delta, weights), bounds=(0., 1.), method='bounded')
    return dict(attack_beta=float(beta[0]), return_beta=float(beta[1]), c=float(result.x),
                corrected=corrected, loss=objective(fitted.x),
                fire_scale=float(beta[0]/max(beta)) if max(beta) else 1.,
                return_scale=float(beta[1]/max(beta)) if max(beta) else 1.)


def predictions(rows: list[dict], model: dict) -> np.ndarray:
    base = np.array([r['b3'] for r in rows])
    z = np.array([model['attack_beta']*r['a']-model['return_beta']*r['b'] for r in rows])
    values = np.clip(base+model['c']*(expit(z)-.5), 0., 1.)
    return np.where([r.get('terminal', False) for r in rows], base, values)


def event_metrics(rows: list[dict], events: list[dict], key: str) -> list[dict]:
    result = []
    for event in events:
        if event.get('actual_attack_net', event['actual_attack']) < LARGE_SEND:
            continue
        part = [r for r in rows if r['video'] == event['video'] and r['game'] == event['game']
                and event['trigger']-PRE_SECONDS <= r['t'] < event['trigger']]
        if not part or event['balance'] == 0:
            continue
        sign = event['winner_sign']
        anchor = part[0]['b3']
        first = lambda field: next((r['t'] for r in part if sign*(r[field]-anchor)*100 >= MOVE_POINTS), event['trigger'])
        wrong = [r['t'] for r in part if sign*(r[key]-r['b3']) < -EPS] if event['balance'] > 0 else []
        result.append(dict(video=event['video'], game=event['game'], event=event['event'],
            version=key, points=float(np.mean([sign*(r[key]-r['b3'])*100 for r in part])),
            earlier_sec=first('b3')-first(key),
            wrong_lower_sec=len(wrong)*SAMPLE_SEC, wrong_times=json.dumps(wrong),
            receiver_won=event['balance'] > 0))
    return result


def exchange_winner(row: dict, labels: dict, fires: dict, outcomes: dict) -> tuple[int, float]:
    """観測済み応手リンクを終端までたどる。予測器には渡さない採点専用処理。"""
    first = fires[row['event']]
    current, total, visited = row, float(first.get('pending_p2', 0)-first.get('pending_p1', 0)), set()
    while current['event'] not in visited:
        event = current['event']
        visited.add(event)
        fire = fires[event]
        sign = 1 if fire['attacker'] == '1P' else -1
        total += sign*current['actual_attack']
        response = outcomes.get(event, {}).get('response')
        if not response:
            break
        following = [v for eid, v in labels.items() if eid not in visited and v['game'] == row['game']
            and fires[eid]['attacker'] == fire['receiver'] and fires[eid]['trigger_sec'] > fire['trigger_sec']
            and abs(fires[eid]['trigger_sec']-response['trigger_sec']) <= 1.]
        if not following:
            total -= sign*current['actual_return']
            break
        current = min(following, key=lambda v: abs(fires[v['event']]['trigger_sec']-response['trigger_sec']))
    return (1 if total > 0 else -1 if total < 0 else 0), total


def observed_events() -> list[dict]:
    result = []
    for video in VIDEOS:
        run = run_path(video)
        shadow = list(jsonl(run/'out/spool/exchange_projection_shadow.jsonl'))
        fires = {r['event_id']: r for r in shadow if r.get('kind') == 'event'}
        outcomes = {r['event_id']: r for r in shadow if r.get('kind') == 'outcome'}
        labels = {r['event']: r for r in samples(run, video)}
        for row in labels.values():
            fire = fires[row['event']]
            winner, total = exchange_winner(row, labels, fires, outcomes)
            receiver_sign = 1 if fire['receiver'] == '1P' else -1
            result.append(dict(row, trigger=fire['trigger_sec'], winner_sign=winner,
                balance=total*receiver_sign, final_net=total,
                actual_attack_net=max(0., row['actual_attack']-fire.get(
                    'pending_p1' if fire['attacker'] == '1P' else 'pending_p2', 0))))
    return result


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    rows = [r for video in VIDEOS for r in features(*load_video(video))]
    groups = np.array([f"{r['video']}/{r['game']}" for r in rows])
    fold_models = []
    for fold, (train, test) in enumerate(GroupKFold(FOLDS).split(rows, groups=groups)):
        training, held = [rows[i] for i in train], [rows[i] for i in test]
        for corrected in (False, True):
            key = 'corrected' if corrected else 'raw'
            model = fit(training, corrected)
            fold_models.append(dict(fold=fold, version=key, model=model,
                train_groups=sorted(set(groups[train])), test_groups=sorted(set(groups[test]))))
            for index, p in zip(test, predictions(held, model)):
                rows[index][key], rows[index]['fold'] = float(p), fold
    events, metrics = observed_events(), []
    save(OUT/'scoring_events.json', events)
    for key in ('raw', 'corrected'):
        metrics += event_metrics(rows, events, key)
    y, base = np.array([r['win'] for r in rows]), np.array([r['b3'] for r in rows])
    summary = dict(videos=list(VIDEOS), games=len(set(groups)), rows=len(rows), baseline_log_loss=loss(y, base), versions={})
    for key in ('raw', 'corrected'):
        part = [r for r in metrics if r['version'] == key]
        summary['versions'][key] = dict(log_loss=loss(y, np.array([r[key] for r in rows])),
            events=len(part), earlier_sec=float(np.mean([r['earlier_sec'] for r in part])),
            points=float(np.mean([r['points'] for r in part])),
            wrong_lower_sec=len({(r['video'], r['game'], t) for r in part
                                 for t in json.loads(r['wrong_times'])})*SAMPLE_SEC)
    models = {key: fit(rows, corrected) for key, corrected in (('raw', False), ('corrected', True))}
    save(EVIDENCE/'prefire_cv.json', summary)
    save(EVIDENCE/'folds.json', fold_models)
    save(EVIDENCE/'model.json', dict(models=models, videos=list(VIDEOS), excluded_video=80,
        locked=True, pre_seconds=PRE_SECONDS, large_send=LARGE_SEND, move_points=MOVE_POINTS))
    write_csv(OUT/'oof.csv', rows)
    write_csv(EVIDENCE/'advance.csv', metrics)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
