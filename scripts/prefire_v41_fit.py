"""勝敗と実測撃ち合い収支を同じgrouped CVで比較する。動画80は読まない。"""
from __future__ import annotations
from collections import Counter
import csv
import json
import numpy as np
from scipy.optimize import nnls
from sklearn.model_selection import GroupKFold
from scripts.exchange_review_io import jsonl, save, sha256
from scripts.prefire_v4_data import VIDEOS, SAMPLE_SEC
from scripts.prefire_v4_fit import fit, loss, event_metrics, write_csv, PRE_SECONDS, FOLDS
from scripts.prefire_v41_diagnose import OUT, OLD, EVIDENCE, statistics
from src.prefire_v41_model import fresh_rows, predict, implied_balance, V2_SLOPE, V2_C
from src.b3_timing import SEC_PER_HAND_DEFAULT
from src.prefire_display import COUNT_UNIT

# 無減衰、1手、5手の寿命を内側CVで比較する。動画80で変更しない。
DECAYS = (0., 1./SEC_PER_HAND_DEFAULT, 1./(5*SEC_PER_HAND_DEFAULT))
OBJECTIVES = ('game', 'exchange')
INNER_FOLDS = 3


def load_rows() -> list[dict]:
    with (OLD/'oof.csv').open(encoding='utf-8-sig') as stream:
        values = list(csv.DictReader(stream))
    result = []
    for row in values:
        parsed = {k: float(row[k]) for k in ('t', 'b3', 'a', 'b')}
        parsed.update({k: int(row[k]) for k in ('video', 'game', 'win', 'fold')})
        parsed.update({k: float(row[k]) if row[k] else None for k in ('ready_sec', 'input_sec')})
        parsed['terminal'] = row['terminal'] == 'True'
        result.append(parsed)
    if set(r['video'] for r in result) != set(VIDEOS):
        raise ValueError('fit動画の不一致')
    return result


def label_rows(rows: list[dict]) -> list[dict]:
    """実際に撃った側のA-Bを教師にする。予測時には両方向を独立に計算する。"""
    events = json.loads((OLD/'scoring_events.json').read_text())
    pairs = json.loads((OUT/'diagnostic_pairs.json').read_text())
    sides = {(r['video'], r['event']): r['side'] for r in pairs}
    usable = [e for e in events if (e['video'], e['event']) in sides]
    computed = {(v, r['input_sec']): r for v in VIDEOS for r in jsonl(OLD/f'compute_{v}.jsonl')}
    for row in rows:
        future = [e for e in usable if e['video'] == row['video'] and e['game'] == row['game']
                  and 0 < e['trigger']-row['t'] <= PRE_SECONDS]
        event = min(future, key=lambda e: e['trigger'], default=None)
        if event is not None and row['input_sec'] is not None and not row['terminal']:
            side = sides[event['video'], event['event']]
            threat = computed[row['video'], row['input_sec']]['sides'][side]
            row.update(exchange_target=event['actual_attack']-event['actual_return'],
                       label_event=event['event'], local_a=threat['attack']/COUNT_UNIT,
                       local_b=threat['returned']/COUNT_UNIT)
    return rows


def group(row: dict) -> str:
    return f"{row['video']}/{row['game']}"


def fit_exchange(rows: list[dict], corrected: bool) -> dict:
    part = [r for r in rows if 'exchange_target' in r]
    sizes = Counter((r['video'], r['label_event']) for r in part)
    weight = np.sqrt([1./sizes[r['video'], r['label_event']] for r in part])
    x = np.asarray([[r['local_a'], -r['local_b']] for r in part])
    if not corrected:
        x = x.sum(axis=1, keepdims=True)
    y = np.asarray([r['exchange_target'] for r in part])
    beta, _ = nnls(x*weight[:, None], y*weight)
    beta = beta if corrected else np.repeat(beta, 2)
    return dict(attack_beta=float(beta[0]*V2_SLOPE), return_beta=float(beta[1]*V2_SLOPE),
                c=V2_C, corrected=corrected, objective='exchange', samples=len(part))


def train(rows: list[dict], objective: str, corrected: bool, decay: float) -> dict:
    fresh = fresh_rows(rows, decay)
    for original, transformed in zip(rows, fresh):
        if 'local_a' in original:
            local = fresh_rows([dict(original, a=original['local_a'], b=original['local_b'])], decay)[0]
            transformed.update(local_a=local['a'], local_b=local['b'])
    model = fit(fresh, corrected) if objective == 'game' else fit_exchange(fresh, corrected)
    return dict(model, decay=decay, objective=objective)


def choose(rows: list[dict], objective: str, corrected: bool) -> dict:
    groups = [group(r) for r in rows]
    splits = list(GroupKFold(INNER_FOLDS).split(rows, groups=groups))
    trials = []
    for decay in DECAYS:
        actual, predicted = [], []
        for training, held in splits:
            model = train([rows[i] for i in training], objective, corrected, decay)
            part = [rows[i] for i in held]
            actual.extend(r['win'] for r in part)
            predicted.extend(predict(part, model))
        trials.append(dict(decay=decay, game_log_loss=loss(np.asarray(actual), np.asarray(predicted))))
    best = min(trials, key=lambda r: (r['game_log_loss'], r['decay']))
    return dict(train(rows, objective, corrected, best['decay']), inner_trials=trials)


def cross_validate(rows: list[dict]) -> list[dict]:
    models = []
    for fold in range(FOLDS):
        training = [r for r in rows if r['fold'] != fold]
        held = [r for r in rows if r['fold'] == fold]
        assert not {group(r) for r in training} & {group(r) for r in held}
        for objective in OBJECTIVES:
            for corrected in (False, True):
                key = objective+('_corrected' if corrected else '_raw')
                model = choose(training, objective, corrected)
                local = [dict(r, a=r.get('local_a', 0.), b=r.get('local_b', 0.)) for r in held]
                for row, p, x in zip(held, predict(held, model), implied_balance(local, model)):
                    row[key], row[key+'_x'] = float(p), float(x)
                models.append(dict(fold=fold, key=key, model=model,
                    train_groups=sorted({group(r) for r in training}), test_groups=sorted({group(r) for r in held})))
                print('CV', fold, key, model['decay'], flush=True)
    return models


def metric(rows: list[dict], events: list[dict], key: str) -> tuple[dict, list[dict]]:
    metrics = event_metrics(rows, events, key)
    local = [r for r in rows if 'exchange_target' in r]
    result = dict(log_loss=loss(np.asarray([r['win'] for r in rows]), np.asarray([r[key] for r in rows])),
        events=len(metrics), earlier_sec=float(np.mean([r['earlier_sec'] for r in metrics])),
        points=float(np.mean([r['points'] for r in metrics])),
        wrong_lower_sec=len({(r['video'], r['game'], t) for r in metrics for t in json.loads(r['wrong_times'])})*SAMPLE_SEC)
    if key+'_x' in rows[0]:
        result['exchange'] = statistics([r[key+'_x'] for r in local], [r['exchange_target'] for r in local])
        result['exchange']['rmse'] = float(np.sqrt(np.mean([(r[key+'_x']-r['exchange_target'])**2 for r in local])))
    return result, metrics


def main() -> None:
    rows = label_rows(load_rows())
    folds = cross_validate(rows)
    events = json.loads((OLD/'scoring_events.json').read_text())
    versions, advances, models = {}, [], {}
    for objective in OBJECTIVES:
        for corrected in (False, True):
            key = objective+('_corrected' if corrected else '_raw')
            versions[key], part = metric(rows, events, key)
            advances.extend(part)
            models[key] = choose(rows, objective, corrected)
    selected = min(versions, key=lambda k: versions[k]['log_loss'])
    baseline = loss(np.asarray([r['win'] for r in rows]), np.asarray([r['b3'] for r in rows]))
    summary = dict(videos=list(VIDEOS), rows=len(rows), groups=len({group(r) for r in rows}),
        local_rows=sum('exchange_target' in r for r in rows), baseline_log_loss=baseline,
        versions=versions, selected=selected, improved=versions[selected]['log_loss'] < baseline)
    save(EVIDENCE/'cv.json', summary)
    save(EVIDENCE/'folds.json', folds)
    save(EVIDENCE/'model.json', dict(locked=True, videos=list(VIDEOS), excluded_video=80,
        selected=selected, models=models, cv_sha256=sha256(EVIDENCE/'cv.json'), v2_slope=V2_SLOPE, v2_c=V2_C))
    keys = set().union(*(r.keys() for r in rows))
    write_csv(OUT/'oof.csv', [{k: r.get(k) for k in sorted(keys)} for r in rows])
    write_csv(EVIDENCE/'advance.csv', advances)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
