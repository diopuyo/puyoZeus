"""配布と同じWindows Python/nativeで専用評価processの原票を検収する。"""
from __future__ import annotations
import argparse
import gzip
from io import BytesIO
import json
import math
from pathlib import Path
import sys
from time import perf_counter

PROBABILITY_TOLERANCE = 1e-9
PROBABILITY_FIELDS = {'v_post', 'v_post_variants', 'pre_delta_full'}
SUMMARY = dict(max_probability_difference=0., rounded_fields=0, discrete_differences=0)


def differences(a: object, b: object, path: tuple = ()) -> list[dict]:
    """確率とその補正量だけ許容差を使い、端点とその他の出力は厳密に比べる。"""
    if isinstance(a, dict) and isinstance(b, dict) and a.keys() == b.keys():
        return [d for key in a for d in differences(a[key], b[key], (*path, key))]
    if isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        return [d for i, (x, y) in enumerate(zip(a, b)) for d in differences(x, y, (*path, i))]
    if a == b and type(a) is type(b):
        return []
    if path and path[0] in PROBABILITY_FIELDS and type(a) in (int, float) and type(b) in (int, float):
        delta = abs(a-b)
        SUMMARY['max_probability_difference'] = max(SUMMARY['max_probability_difference'], delta)
        SUMMARY['rounded_fields'] += 1
        endpoint = path[0] != 'pre_delta_full' and (a in (0, 1) or b in (0, 1))
        if math.isfinite(delta) and delta <= PROBABILITY_TOLERANCE and not endpoint:
            return []
    SUMMARY['discrete_differences'] += 1
    return [dict(path=list(path), expected=a, actual=b)]


def stable(value: object, path: tuple = ()) -> object:
    if isinstance(value, dict):
        return {k: stable(v, (*path, k)) for k, v in value.items() if not k.endswith('_ms')
                and k not in ('ready_sec', 'publish_sec', 'busy_until') and not (path == ('stage',) and k == 'ms')}
    if isinstance(value, (list, tuple)):
        return [stable(v, (*path, i)) for i, v in enumerate(value)]
    return value


def check(before: dict | None, after: dict, out: Path, kind: str, index: int) -> None:
    a, b = stable(before), stable(after)
    found = differences(a, b)
    if found:
        (out/'first_mismatch.json').write_text(json.dumps(dict(kind=kind, index=index,
            differences=found, expected=a, actual=b), indent=2), encoding='utf-8')
        raise AssertionError(f'{kind} {index} differs')


def run(args: argparse.Namespace) -> None:
    sys.path.insert(0, str(args.app))
    if args.native:
        sys.path.insert(0, str(args.native))
    from src.phase_j.live_v49f_async import AsyncV49F
    args.out.mkdir(exist_ok=False)
    video, game = args.video, args.game
    assets = (args.assets/'model.joblib', args.assets/'timing_table.json',
        args.policy, args.pre_model)
    worker = AsyncV49F(assets, args.out)
    expected = iter(json.loads(s) for s in (args.packets/f'predictions_{video}_{game}.jsonl').read_text().splitlines())
    old = iter(json.loads(s) for s in (args.packets/f'base_{video}_{game}.jsonl').read_text().splitlines())
    count, base_count, began, input_ids = 0, 0, perf_counter(), []
    try:
        with gzip.open(BytesIO((args.packets/f'{video}_{game}.jsonl.gz').read_bytes()),
                       'rt', encoding='utf-8') as stream:
            for line in stream:
                packet = json.loads(line)
                input_ids.append(packet['id'])
                worker.submit(packet)
                if not worker.connection.poll(120.):
                    raise TimeoutError('Windows評価process応答なし')
                result = worker.connection.recv()
                if result.get('fatal'):
                    raise RuntimeError(result)
                if result['base'] is not None:
                    check(next(old, None), dict(result['base'], video=video), args.out, 'base', base_count)
                    base_count += 1
                if result['value'] is not None:
                    check(next(expected, None), dict(result['value'], video=video), args.out, 'v49f', count)
                    count += 1
                    if count % 100 == 0:
                        print(video, game, count, flush=True)
        assert next(expected, None) is None and next(old, None) is None
    finally:
        worker.close()
    audit_ids = [json.loads(line)['id'] for line in (args.out/'v49f_updates.jsonl').read_text().splitlines()]
    assert audit_ids == input_ids, '正常終了後の監査件数または順序が入力と一致しない'
    receipt = dict(video=video, game=game, rows=count, base_rows=base_count,
                   input_records=len(input_ids), audit_records=len(audit_ids),
                   tolerance=PROBABILITY_TOLERANCE, **SUMMARY,
                   seconds=perf_counter()-began, python=sys.executable)
    (args.out/'receipt.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for key in ('app', 'assets', 'policy', 'pre-model', 'packets', 'out'):
        parser.add_argument('--'+key, type=Path, required=True)
    parser.add_argument('--native', type=Path)
    parser.add_argument('--video', type=int, required=True)
    parser.add_argument('--game', type=int, required=True)
    run(parser.parse_args())
