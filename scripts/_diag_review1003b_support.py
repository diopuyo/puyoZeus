"""再生観測の学習範囲・係数寄与・バイト同一性を検証する。"""
from __future__ import annotations

import argparse
import json
from typing import Any
import numpy as np
from scripts._diag_review1003b_extract import ROOT, OUT, CASES, resources, save, records
from scripts._diag_review1003b_inspect import TIMES
from src.exchange_event_record import decode
from src.exchange_event_evaluator import FileExchangeModels
from src.exchange_event_features import D_COLUMNS, SCORE_COLUMNS, S3_COLUMNS
from src.exchange_event_count_features import NEW_COLUMNS
from scripts._diag_review1003b_models import tree_path

SCORE_START, SCORE_STOP = len(S3_COLUMNS)-len(SCORE_COLUMNS), len(S3_COLUMNS)
RANGE_TOLERANCE = 1e-6
SCORE_NEIGHBOR_TOLERANCE = .01
SENT_COLUMNS = 3


def contributions(model: Any, x: np.ndarray) -> tuple[float, np.ndarray]:
    """木経路寄与を加法検証付きで返す。"""
    estimator = model[-1] if hasattr(model, 'steps') else model
    z = model[:-1].transform(x[None, :])[0] if hasattr(model, 'steps') else x
    intercept = float(estimator._baseline_prediction[0, 0])
    parts_sum = np.zeros(len(z))
    for predictors in estimator._predictors:
        base, parts = tree_path(predictors[0].nodes, z)
        intercept += base
        parts_sum += parts
    assert np.isclose(intercept+parts_sum.sum(), estimator.decision_function(z[None, :])[0])
    return intercept, parts_sum


def analyze(case: int) -> None:
    """全代表時刻の特徴範囲とS3単一候補の寄与を記録する。"""
    _, set_no, game, _, _ = next(c for c in CASES if c[0] == case)
    replay = OUT / f'replay_case{case}'
    expected = ROOT / 'work' / f'set{set_no}_game{game:02}'
    equal = {name: (replay / name).read_bytes() == (expected / name).read_bytes()
             for name in ('events.jsonl', 'display.npz')}
    equal['diagnostics_json_semantic'] = (json.loads((replay / 'events.diagnostics.json').read_text())
        == json.loads((expected / 'events.diagnostics.json').read_text()))
    assert all(equal.values()), equal
    rows = [decode(r) for r in records(replay / 'observations.jsonl')]
    matrix = np.load('logs/e15/S3_prime.npy', mmap_mode='r')
    ranges = json.loads((OUT / 's3_training_support.json').read_text())['feature_ranges']
    model = FileExchangeModels.load('models/exchange_event_v3').estimators['S3']
    names = S3_COLUMNS+NEW_COLUMNS
    results = []
    for stamp in TIMES[case]:
        row = min(rows, key=lambda r: abs(r['t_sec']-stamp))
        if 'features' not in row:
            continue
        x = row['features']
        intercept, contribution = contributions(model, x)
        outside = [name for j, name in enumerate(names) if x[j] < ranges[name]['min']-RANGE_TOLERANCE
                   or x[j] > ranges[name]['max']+RANGE_TOLERANCE]
        grouped = {'D': float(contribution[:len(D_COLUMNS)].sum()),
            'arrival': float(contribution[len(D_COLUMNS):SCORE_START].sum()),
            'score': float(contribution[SCORE_START:SCORE_STOP].sum()), 'count': float(contribution[SCORE_STOP:].sum())}
        score = x[SCORE_START:SCORE_STOP]
        near = np.all(np.abs(matrix[:, SCORE_START:SCORE_START+SENT_COLUMNS]-score[:SENT_COLUMNS])
                      < SCORE_NEIGHBOR_TOLERANCE, axis=1)
        results.append(dict(t_sec=row['t_sec'], source=row['source'], displayed_p1=row['p1'],
            single_candidate_p1=float(model.predict_proba(x[None, :])[0, 1]), outside=outside,
            totals=row['totals'].tolist(), score_elapsed=row['score_elapsed'],
            intercept=intercept, groups=grouped, similar_score_rows=int(near.sum()),
            features=[dict(name=name, value=float(x[j]), logit_contribution=float(contribution[j]))
                      for j, name in enumerate(names)]))
    save(f'case{case}_support.json', dict(equivalence=equal, estimator=str(model),
        training_rows=len(matrix), results=results, resources=resources()))
    print(case, equal, [(r['t_sec'], r['outside'], r['groups']) for r in results])


def main() -> None:
    """完了済み再生の指定事例を解析する。"""
    parser = argparse.ArgumentParser()
    parser.add_argument('case', type=int, choices=(4, 6, 7))
    resources()
    analyze(parser.parse_args().case)


if __name__ == '__main__':
    main()
