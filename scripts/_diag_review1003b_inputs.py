"""保存入力の時系列化とG_feの同一入力比較を行う診断専用CLI。"""
from __future__ import annotations

import csv
from dataclasses import asdict
import importlib.util
import json
from typing import Any
import numpy as np

from scripts._diag_review1003b_extract import ROOT, OUT, CASES, resources, save
from src.exchange_event_record import read_records, encode
from src.exchange_event_evaluator import FileExchangeModels, StaticInput, build_features
from src.exchange_event_features import G_COLUMNS


def collect(set_no: int, game: int) -> list[dict[str, Any]]:
    """各フレームの盤面・NEXT・式得点・静止評価呼出を関連付ける。"""
    rows, cache, calls = [], {}, []
    for item in read_records(ROOT / 'work' / f'set{set_no}_game{game:02}' / 'inputs.jsonl.gz'):
        if item['kind'] == 'static':
            cache[item['key']] = item['input']
        elif item['kind'] == 'static_call':
            calls.append(dict(cache[item['key']], **{k: item[k] for k in
                ('elapsed_sec', 'm0_probability_1p', 'source_side')}))
        elif item['kind'] == 'update':
            rows.append(dict(t_sec=item['args'][3], args=item['args'], static_calls=calls))
            calls = []
    return rows


def static_analysis(rows: list[dict[str, Any]]) -> None:
    """G_fe全静止行を同じ入力の新旧推定器で比較する。"""
    old = FileExchangeModels.load('models/exchange_event_v3')
    new = old.with_static_gfe('models/exchange_event_v1_r1_hgb')
    frame_rows = list(csv.DictReader((OUT / 'case5_frames.csv').open()))
    by_time = {float(r['t_sec']): r for r in frame_rows}
    entries = []
    for row in rows:
        frame = by_time.get(row['t_sec'])
        if not frame or frame['new_evaluator'] != 'G_fe' or not row['static_calls']:
            continue
        for call in row['static_calls']:
            _, x, _ = build_features(StaticInput(**call), new.elapsed_thresholds)
            pnew = new.predict_source_probability('G_fe', x)
            if abs(pnew-float(frame['new_p1'])) > 1e-10:
                continue
            entries.append(dict(t_sec=row['t_sec'], features=x.tolist(),
                new_p1=pnew, old_same_input=old.predict_source_probability('G_fe', x),
                old_record_p1=float(frame['old_p1']), static_input=encode(call)))
    save('gfe_same_input.json', entries)
    explain(entries, old.estimators['G_fe'], new.estimators['G_fe'])


def explain(entries: list[dict[str, Any]], old: Any, new: Any) -> None:
    """単変量中和とLRの正確なlogit寄与を保存する。"""
    x = np.array([r['features'] for r in entries])
    print('matched G_fe calls', len(entries), 'old', old, 'shap', bool(importlib.util.find_spec('shap')))
    result = []
    for time in (4240., 4242., 4244., 4246.):
        idx = min(range(len(entries)), key=lambda i: abs(entries[i]['t_sec']-time))
        row, source = x[idx], entries[idx]
        changed = np.tile(row, (len(row), 1))
        np.fill_diagonal(changed, 0.)
        deltas = source['new_p1']-new.predict_proba(changed)[:, 1]
        transformed = old[:-1].transform(row[None, :]) if hasattr(old, 'steps') else row[None, :]
        lr = old[-1] if hasattr(old, 'steps') else old
        contributions = transformed[0]*lr.coef_[0]
        result.append(dict(t_sec=source['t_sec'], new_p1=source['new_p1'],
            old_same_input=source['old_same_input'], old_record_p1=source['old_record_p1'],
            lr_intercept=float(lr.intercept_[0]), features=[dict(name=name, value=float(row[j]),
            hgb_neutralization_delta=float(deltas[j]), lr_logit_contribution=float(contributions[j]))
            for j, name in enumerate(G_COLUMNS)]))
    save('gfe_attribution.json', result)
    save('gfe_features_schema.json', list(G_COLUMNS))


def main() -> None:
    """許可領域へ全入力を保存しモデルを比較する。"""
    resources()
    for set_no, game in sorted({(c[1], c[2]) for c in CASES}):
        rows = collect(set_no, game)
        save(f'set{set_no}_game{game}_inputs.json', encode(rows))
        if (set_no, game) == (2, 11):
            static_analysis(rows)
    save('inputs_resources.json', resources())


if __name__ == '__main__':
    main()
