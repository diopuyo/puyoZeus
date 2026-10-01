"""セット1の57試合だけで反映係数を最尤推定し、採点前に凍結する。"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit

from src.prefire_best_play_v6 import FEATURE_NAMES
from src.prefire_v6_value import PROBABILITY_EPSILON
from scripts.prefire_v6_replay import OUT, EVALSET

RIDGE = 1e-4
MAX_ITERATIONS = 1000
TRAIN_SOURCES = (*(f'c{i}' for i in range(1, 7)), 'zenchi')


def displayed_samples(root: Path) -> tuple[np.ndarray, np.ndarray]:
    """表示がないウォームアップ行を除き、隣の分割記録と二重計上しない。"""
    with np.load(root/'prefire_v6_features.npz') as data:
        rows = data['values']
    with np.load(root/'display.npz') as data:
        stamps = data['t_sec']
    return rows[np.isin(rows[:, 0], stamps)], stamps


def samples() -> tuple:
    """表示全フレーム数で重みを決め、欠測・計算待ちも母数へ含む。"""
    games = json.loads((EVALSET/'labels.json').read_text())['games']
    if len(games) != 57:
        raise ValueError('セット1は57試合で固定')
    rows, stamps, files = [], [], []
    for source in TRAIN_SOURCES:
        root = OUT/'train'/source
        if not (root/'DONE.json').exists():
            raise FileNotFoundError(root/'DONE.json')
        path = root/'prefire_v6_features.npz'
        eligible, displayed = displayed_samples(root)
        rows.append(eligible)
        stamps.extend(displayed)
        files.extend([path, root/'display.npz'])
    table, display = np.concatenate(rows), np.asarray(stamps)
    chosen, labels, weights = [], [], []
    for game in games:
        mask = (table[:, 0] >= game['start']) & (table[:, 0] < game['end'])
        count = np.count_nonzero((display >= game['start']) & (display < game['end']))
        if count == 0:
            raise ValueError('学習試合の表示が欠落')
        chosen.append(table[mask])
        labels.extend([float(game['winner'] == '1P')]*int(mask.sum()))
        weights.extend([1/(len(games)*count)]*int(mask.sum()))
    return np.concatenate(chosen), np.asarray(labels), np.asarray(weights), files


def fit(table: np.ndarray, labels: np.ndarray, weights: np.ndarray) -> dict:
    """a(z)=sigmoid(b+wz)。目的は試合等重みLL、L2係数は採点前固定。"""
    x = np.column_stack((np.ones(len(table)), table[:, 5:5+len(FEATURE_NAMES)]))
    current = logit(np.clip(table[:, 2], PROBABILITY_EPSILON, 1-PROBABILITY_EPSILON))
    delta = logit(np.clip(table[:, 3], PROBABILITY_EPSILON, 1-PROBABILITY_EPSILON))-current
    def objective(beta: np.ndarray) -> tuple:
        alpha = expit(x@beta)
        prediction = current+alpha*delta
        loss = np.dot(weights, np.logaddexp(0, prediction)-labels*prediction)+RIDGE*np.dot(beta, beta)
        gradient = x.T@(weights*(expit(prediction)-labels)*delta*alpha*(1-alpha))+2*RIDGE*beta
        return float(loss), gradient
    optimum = minimize(objective, np.zeros(x.shape[1]), jac=True, method='L-BFGS-B',
                       options={'maxiter': MAX_ITERATIONS, 'gtol': 1e-10})
    if not optimum.success:
        raise RuntimeError(optimum.message)
    return dict(coefficients=optimum.x.tolist(), rows=len(table), nonzero_delta=int(np.count_nonzero(delta)),
                identifiable=bool(np.count_nonzero(delta)), objective=float(optimum.fun),
                ridge=RIDGE, optimizer=str(optimum.message), training_set='zenchi_set1_57',
                features=list(FEATURE_NAMES), iterations=int(optimum.nit))


def main() -> None:
    """原票SHAを添えて保存し、以後はセット2から学習しない。"""
    table, labels, weights, files = samples()
    result = fit(table, labels, weights)
    files.extend([EVALSET/'labels.json', OUT/'latency.json'])
    result['sha256'] = {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    with (OUT/'strength.json').open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
    print(result, flush=True)


if __name__ == '__main__':
    main()
