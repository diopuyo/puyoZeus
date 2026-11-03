"""保存学習行とlive入力口を照合。採点・モデル更新は行わない。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.b3_realtime import RealtimeRows
from src.b3_realtime_replay import replay_inputs
from src.b3_rows import _inside_match, to_frame
from src.b3_source import resolve_attempt, sha256_file
from src.b3_timing import TimingTable
from src.b3_train_model import load_model

EXCLUDE = {'won_p1', 'held_sec', 'weight'}


def compare(actual: pd.DataFrame, expected: pd.DataFrame, model: object,
            raw_predictions: dict[str, float] | None = None) -> dict:
    a, b = actual.set_index('row_key'), expected.set_index('row_key')
    shared = a.index.intersection(b.index)
    differences = {}
    for name in (set(a.columns) & set(b.columns)) - EXCLUDE:
        x, y = a.loc[shared, name], b.loc[shared, name]
        equal = (x == y) | (x.isna() & y.isna())
        if not equal.all():
            delta = None if x.dtype == object else float(np.nanmax(np.abs(x.to_numpy(float)-y.to_numpy(float))))
            differences[name] = dict(count=int((~equal).sum()), max_abs=delta)
    result = dict(rows=len(a), reference_rows=len(b), common_rows=len(shared),
        missing=len(b.index.difference(a.index)), extra=len(a.index.difference(b.index)),
        column_differences=differences, new_reference_columns=sorted(set(b)-set(a)),
        missing_reference_columns=sorted(set(a)-set(b)),
        first_rows=[dict(game=int(g), frame=int(part.iloc[0].frame_idx))
                    for g, part in actual.groupby('game_idx')])
    if model is None:
        return dict(result, prediction_skipped=True)
    return dict(result, **compare_predictions(a.loc[shared], b.loc[shared], model, raw_predictions))


def compare_predictions(a: pd.DataFrame, b: pd.DataFrame, model: object,
                        raw_predictions: dict[str, float] | None) -> dict:
    """モデルの列が揃う場合だけ、一括・単行の予測を照合する。"""
    p, q = model.predict(a), model.predict(b)
    live = np.asarray([raw_predictions[key] for key in a.index]) if raw_predictions is not None else np.asarray(
        [model.predict_row(row) for row in a.to_dict('records')])
    return dict(
        prediction_max_abs=float(np.max(np.abs(p-q))) if len(p) else None,
        prediction_bit_equal=bool(np.array_equal(p.view(np.uint64), q.view(np.uint64))),
        single_row_max_abs=float(np.max(np.abs(live-q))) if len(q) else None,
        single_row_bit_equal=bool(np.array_equal(live.view(np.uint64), q.view(np.uint64))))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('root', 'rows', 'model', 'timing-table', 'out'):
        parser.add_argument('--'+key, type=Path, required=True)
    parser.add_argument('--video', default='b3_zenchi_s1')
    parser.add_argument('--max-frame', type=int, default=5400)
    parser.add_argument('--no-timing', action='store_true', help='計時せず特徴と予測だけを照合する')
    parser.add_argument('--features-only', action='store_true', help='旧モデルでは特徴だけを照合する')
    args = parser.parse_args()
    record = resolve_attempt(args.video, args.root).record
    manifest = json.loads((args.rows / (args.video+'.manifest.json')).read_text(encoding='utf-8'))
    engine = RealtimeRows(TimingTable.from_json(args.timing_table.read_text()), args.video,
                          measure_timing=not args.no_timing)
    model = None if args.features_only else load_model(args.model)
    rows, costs, predictions = [], [], {}
    for frame, events, queues in replay_inputs(record, args.max_frame):
        span = manifest['winner_mapping'].get(str(frame['game_idx']), {})
        row = engine.update(frame, events, queues, _inside_match(frame['t_sec'], span))
        if row is not None:
            rows.append(row)
            if not args.no_timing:
                costs.append(engine.feature_ms)
            if model is not None:
                predictions[row['row_key']] = model.predict_row(row)
    reference = pd.read_parquet(args.rows / (args.video+'.parquet'))
    reference = reference[(reference.orientation == 0) & (reference.frame_idx <= args.max_frame)]
    result = compare(to_frame(rows), reference, model, predictions)
    result.update(model_sha256=sha256_file(args.model), timing_measured=not args.no_timing)
    if costs:
        result['feature_ms'] = dict(zip(('p95', 'p99', 'max'),
                                       map(float, np.percentile(costs, (95, 99, 100)))))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
