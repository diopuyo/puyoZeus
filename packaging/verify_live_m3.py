"""M3研究生成器・再生・配信の凍結入力計算を、同じ保存観測で照合する。"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from time import perf_counter

FLAGS = ('B3_LEDGER_INFLIGHT_DEFAULT', 'B3_LEDGER_OBSERVED_RULE_DEFAULT',
         'B3_LEDGER_MODE_DEFAULT', 'B3_LEDGER_LANDING_DEFAULT', 'B3_ICON_LINE_ALIAS_DEFAULT')
OUTPUT_BUFFER_BYTES = 1024*1024


def serial(value: object) -> object:
    return value.item() if hasattr(value, 'item') else str(value)


def equal(a: object, b: object) -> bool:
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        return True
    return a == b


def run(args: argparse.Namespace) -> dict:
    sys.path.insert(0, str(args.root))
    if args.offline:
        from src import production_config
        for flag in FLAGS:
            setattr(production_config, flag, True)
    from src.b3_train_model import load_model, feature_columns
    from src.b3_column_spec import COLUMN_NAMES
    from src.b3_timing import TimingTable
    from src.b3_source import load_lean, obs_game_ranges, winners_by_overlap
    from src.b3_rows import generate_rows, _inside_match
    model, table = load_model(args.model), TimingTable.from_json(args.table.read_text(encoding='utf-8'))
    features = feature_columns(COLUMN_NAMES)
    assert len(features) == 295, len(features)
    _, mapping = winners_by_overlap(load_lean(args.record), obs_game_ranges(args.record))
    if args.offline:
        rows, info = generate_rows(args.record, 'b3_zenchi_s1', table, max_frame=args.max_frame,
                                  require_winner=False, match_mapping=mapping)
        import pandas as pd
        predictions = model.predict(pd.DataFrame(rows))
        with args.out.open('x', encoding='utf-8') as stream:
            for row, p in zip(rows, predictions):
                stream.write(json.dumps(dict(row=row, prediction=float(p)), default=serial)+'\n')
        return dict(rows=len(rows), info=info, features=features)
    return compare_live(args, model, table, mapping, features, _inside_match)


def compare_live(args: argparse.Namespace, model: object, table: object,
                 mapping: dict, features: list, inside: object) -> dict:
    from src.b3_realtime import RealtimeRows
    from src.b3_realtime_replay import replay_inputs
    from src.b3_async import snapshot, calculate
    expected = {int(v['row']['frame_idx']): v for v in
                (json.loads(s) for s in args.reference.read_text(encoding='utf-8').splitlines())}
    state, computation = RealtimeRows(table, 'b3_zenchi_s1'), RealtimeRows(table, 'b3_zenchi_s1')
    receipt = dict(rows=0, features=len(features), max_probability_difference=0., mismatches=[])
    with args.out.open('x', encoding='utf-8', buffering=OUTPUT_BUFFER_BYTES) as stream:
        for frame, events, queues in replay_inputs(args.record, args.max_frame):
            emit = inside(frame['t_sec'], mapping.get(frame['game_idx'], {}))
            if not state.advance(frame, events, queues, emit):
                continue
            actual = calculate(dict(snapshot=snapshot(state), requested_at=perf_counter()), computation, model)
            if actual['error']:
                raise RuntimeError(actual['error'])
            target = expected.pop(frame['frame'], None)
            assert target is not None, ('unexpected frame', frame['frame'])
            row = actual['row']
            different = [k for k in row if k not in ('won_p1', 'weight', 'held_sec') and not equal(row[k], target['row'][k])]
            delta = abs(actual['raw_probability']-target['prediction'])
            receipt['max_probability_difference'] = max(receipt['max_probability_difference'], delta)
            if different or delta > 1e-9:
                receipt['mismatches'].append(dict(frame=frame['frame'], columns=different,
                    differences={k: [target['row'][k], row[k]] for k in different}, probability_delta=delta))
            stream.write(json.dumps(dict(row=row, prediction=actual['raw_probability']), default=serial)+'\n')
            receipt['rows'] += 1
            if receipt['rows'] % 100 == 0:
                print(receipt['rows'], len(receipt['mismatches']), flush=True)
    receipt['missing_reference_rows'] = sorted(expected)
    receipt['passed'] = not receipt['mismatches'] and not expected and receipt['rows'] > 0
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in ('root', 'record', 'model', 'table', 'out'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--reference', type=Path)
    parser.add_argument('--offline', action='store_true')
    parser.add_argument('--max-frame', type=int, default=7950)
    args = parser.parse_args()
    receipt = run(args)
    args.out.with_suffix('.receipt.json').write_text(json.dumps(receipt, default=serial, indent=2), encoding='utf-8')
    print({k: v for k, v in receipt.items() if k not in ('info', 'features', 'mismatches')})
    if receipt.get('passed') is False:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
