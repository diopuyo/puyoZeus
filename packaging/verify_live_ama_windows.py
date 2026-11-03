"""保存v7.1入力を配信と共用するWindows ama評価口へ通す。原票は読取専用。"""
from __future__ import annotations

import sys as _public_release_sys
from pathlib import Path as _PublicReleasePath
_public_release_root = str(_PublicReleasePath(__file__).resolve().parents[0])
if _public_release_root not in _public_release_sys.path:
    _public_release_sys.path.insert(0, _public_release_root)
from src.public_dev_paths import resolve_text as _public_dev_text

import argparse
import hashlib
import json
from pathlib import Path
import sys

OUTPUT_BUFFER_BYTES = 1024*1024
PROGRESS_RECORDS = 100


def differences(a: object, b: object, path: tuple = ()) -> list[dict]:
    if isinstance(a, dict) and isinstance(b, dict) and a.keys() == b.keys():
        return [d for k in a for d in differences(a[k], b[k], (*path, k))]
    if isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        return [d for i, (x, y) in enumerate(zip(a, b)) for d in differences(x, y, (*path, i))]
    if a == b:
        return []
    if path[-1:] == ('probability',) and abs(a-b) <= 1e-9:
        return []
    return [dict(path=path, expected=a, actual=b)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    from src.phase_j.live_ama_engine import AmaEngine
    model = Path(_public_dev_text('<work>/wt_ama_plain/docs/verification/ama_plain/model.json'))
    args.out.mkdir(exist_ok=False)
    engine = AmaEngine(args.binary, model)
    summary = dict(binary_sha256=hashlib.sha256(args.binary.read_bytes()).hexdigest(),
                   model_sha256=hashlib.sha256(model.read_bytes()).hexdigest(), games={})
    try:
        for stem in ('80_g7', '1042_g1'):
            inputs = json.loads(Path(_public_dev_text(f'<work>/wt_ama_eval/logs/ama_eval_v71/inputs_{stem}.json')).read_text(encoding='utf-8'))
            expected = [json.loads(line) for line in Path(_public_dev_text(f'<work>/wt_ama_plain/logs/ama_plain_v71/evaluations_{stem}.jsonl')).read_text(encoding='utf-8').splitlines()]
            assert len(inputs) == len(expected)
            receipt = dict(rows=0, valid=0, max_probability_difference=0., discrete_differences=0)
            summary['games'][stem] = receipt
            with (args.out/f'{stem}.jsonl').open('x', encoding='utf-8', buffering=OUTPUT_BUFFER_BYTES) as stream:
                for item, target in zip(inputs, expected):
                    actual = engine.evaluate(item)
                    stream.write(json.dumps(actual, ensure_ascii=False)+'\n')
                    compared = {k: v for k, v in target.items() if k != 'query_keys'}
                    found = differences(compared, {k: actual[k] for k in compared})
                    if found:
                        (args.out/'first_mismatch.json').write_text(json.dumps(dict(stem=stem, index=item['index'], differences=found), ensure_ascii=False, indent=2), encoding='utf-8')
                        raise AssertionError(f'{stem} index={item["index"]}: {len(found)} differences')
                    receipt['rows'] += 1
                    if 'value' in target:
                        receipt['valid'] += 1
                        receipt['max_probability_difference'] = max(receipt['max_probability_difference'], abs(target['value']['probability']-actual['value']['probability']))
                    if receipt['rows'] % PROGRESS_RECORDS == 0:
                        print(stem, receipt['rows'], flush=True)
            print(stem, receipt, flush=True)
        summary['passed'] = True
    finally:
        engine.close()
        (args.out/'receipt.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
