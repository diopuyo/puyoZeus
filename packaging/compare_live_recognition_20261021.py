"""保存された全行を完全一致で比較する。時計の除外は保存側で固定済み。"""
from __future__ import annotations

import argparse
import gzip
from hashlib import sha256
from itertools import zip_longest
import json
from pathlib import Path

TIME_FIELDS = frozenset({'image_compute_ms'})


def without_measured_time(value: object) -> object:
    """発火時刻・入力時刻は比較する。実行時間の診断値だけを除く。"""
    if isinstance(value, dict):
        return {k: without_measured_time(v) for k, v in value.items() if k not in TIME_FIELDS}
    if isinstance(value, list):
        return [without_measured_time(v) for v in value]
    return value


def comparison_bytes(line: bytes | None) -> bytes | None:
    if line is None:
        return None
    return json.dumps(without_measured_time(json.loads(line)),
                      sort_keys=True, ensure_ascii=False).encode('utf-8')


def compare(root: Path) -> dict:
    before, after = (root/('replay_'+name) for name in ('baseline', 'candidate'))
    inputs = [(p/'inputs.json').read_bytes() for p in (before, after)]
    assert inputs[0] == inputs[1], '保存入力列が異なる'
    receipts = [json.loads((p/'receipt.json').read_text()) for p in (before, after)]
    mismatches, count = [], 0
    image_source_rows = [0, 0]
    digests = [sha256(), sha256()]
    with gzip.open(before/'outputs.jsonl.gz', 'rb') as a, gzip.open(after/'outputs.jsonl.gz', 'rb') as b:
        for count, pair in enumerate(zip_longest(a, b), 1):
            for index, line in enumerate(pair):
                image_source_rows[index] += int(line is not None and b'"image_compute_ms"' in line)
            pair = tuple(comparison_bytes(line) for line in pair)
            for digest, line in zip(digests, pair):
                if line is not None:
                    digest.update(line)
            if pair[0] != pair[1]:
                mismatches.append(count)
    result = dict(rows=count, differing_rows=len(mismatches), first_differences=mismatches[:20],
        input_sha256=sha256(inputs[0]).hexdigest(), output_sha256=[d.hexdigest() for d in digests],
        receipts=receipts, excluded_measured_time_fields=sorted(TIME_FIELDS),
        image_source_rows=image_source_rows,
        criterion='実行時間以外の認識全フィールド・PipelineView・B3 M3/v4.9f/ama入力のJSON全行完全一致')
    (root/'recognition_parity.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    assert not mismatches and count == receipts[0]['rows'] == receipts[1]['rows']
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    print(compare(parser.parse_args().root))
