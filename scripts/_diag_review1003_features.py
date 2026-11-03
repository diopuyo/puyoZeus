"""再生時のモデル入力と反実仮想を保存する診断専用処理。"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from scripts._diag_review1003_extract import OUT, ROOT, resources
from src.exchange_event_record import decode, encode
from src.exchange_event_evaluator import (FileExchangeModels, StaticInput, FiringInput,
    ExchangeEndInput, CountObservation, evaluate_exchange_event)
from src.exchange_event_features import S3_COLUMNS, D_COLUMNS
from src.exchange_event_count_features import NEW_COLUMNS, orient

TARGETS = {13: [946.7167, 947.2167, 948.7167], 26: [1728.9833, 1730.9833, 1732.9833],
           2: [214.0, 220.0]}


def inspect(game: int, models: Any) -> list[dict]:
    """推論を再計算し、固定D・更新count・得点を区別して保存する。"""
    path = OUT / f'replay_game{game:02}' / 'observations.jsonl'
    if not path.exists():
        return []
    rows = [decode(json.loads(s)) for s in path.read_text().splitlines()]
    if not rows:
        return []
    model = models.estimators['S3']
    print('MODEL', type(model), str(model)[:300])
    results = []
    for time in TARGETS[game]:
        row = min(rows, key=lambda r: abs(r['t_sec']-time))
        side = row['firing']['static']['source_side']
        x = np.r_[row['features'], orient(row['count_sides'], side)]
        probability = models.predict_source_probability('S3', x)
        feature_map = dict(zip(S3_COLUMNS + NEW_COLUMNS, x.tolist()))
        result = dict(game=game, t_sec=row['t_sec'], p1=probability,
            features=feature_map, chains=row['chains'],
            firing=encode(row['firing']))
        result['sensitivity'] = sensitivity(row, models)
        print('FEATURE', game, row['t_sec'], probability, row['score_totals'],
              'elapsed', row['score_elapsed'], 'counts', np.expm1(row['count_sides'][:, :-1]).tolist())
        print('D_frozen', {k: feature_map[k] for k in D_COLUMNS if 'chain' in k or 'puyo' in k})
        if hasattr(model, 'coef_'):
            contributions = model.coef_[0] * x
            result['logit_contributions'] = dict(zip(S3_COLUMNS + NEW_COLUMNS, contributions.tolist()))
            result['intercept'] = float(model.intercept_[0])
        results.append(result)
    expected = ROOT / 'work' / f'set1_game{game:02}'
    actual = OUT / f'replay_game{game:02}'
    print('IDENTICAL', game, {name: (expected/name).read_bytes() == (actual/name).read_bytes()
                            for name in ('events.jsonl', 'display.npz')})
    return results


def sensitivity(row: dict, models: Any) -> dict:
    """得点だけを置換する入力感度。正解勝率や修正後評価とは扱わない。"""
    raw = row['firing']
    firing = FiringInput(StaticInput(**raw['static']), raw['prefire_sides'], raw['firing'],
                         CountObservation(**raw['count_observation']))
    variants = {'original': row['score_totals']}
    if 943 <= row['t_sec'] <= 950:
        variants['1P_3160_future_observed'] = np.array([3160, 2540])
    if 1725 <= row['t_sec'] <= 1735:
        variants['1P_0_diagnostic_only'] = np.array([0, row['score_totals'][1]])
        variants['1P_13800_remove_bad_finalization'] = np.array([13800, row['score_totals'][1]])
    return {name: evaluate_exchange_event(ExchangeEndInput(firing, np.zeros(2), scores,
            row['score_elapsed']), models) for name, scores in variants.items()}


def main() -> None:
    """同じモデルで記録値を検証する。"""
    resources()
    models = FileExchangeModels.load(Path('models/exchange_event_v3'), lightweight=True)
    values = [v for game in TARGETS for v in inspect(game, models)]
    (OUT / 'model_inputs.json').write_text(json.dumps(values, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
