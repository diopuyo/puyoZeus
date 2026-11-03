"""本番確定と完全に同じ時刻の評価入力を旧・新記録から抜き出す。"""
from __future__ import annotations

import gzip
import json
from pathlib import Path

from scripts._diag_dv2d_loss_inventory import OUT, read, save, resource
from scripts.falsedeath_replay_20261001 import input_record

EPS = 1e-6


def extract(path: Path, targets: list[dict]) -> dict:
    """前後フレームによる補完はせず、同時刻の行番号と生の入力を保存する。"""
    result = {}
    with gzip.open(path, 'rt', encoding='utf-8') as stream:
        for number, line in enumerate(stream, 1):
            row = json.loads(line)
            if row['kind'] != 'update':
                continue
            stamp = row['args']['tuple'][3]
            for target in targets:
                if abs(stamp-target['first_sec']) <= EPS:
                    key = f"{target['group']}_{target['game']}"
                    result[key] = dict(record=str(path), line=number, **row)
    assert len(result) == len(targets), (path, len(result), len(targets))
    return result


def main() -> None:
    """確定時刻の入力だけなので、モデルや認識器は実行しない。"""
    save('RESOURCE_exact.json', resource())
    targets = read(OUT/'EVENTS.json')
    rows = {}
    for source in dict.fromkeys(r['record'] for r in targets):
        selected = [r for r in targets if r['record'] == source]
        rows[source] = {mode:extract(input_record(mode, source), selected) for mode in ('prod', 'v2d')}
        print(source, len(selected), flush=True)
    save('EXACT_INPUTS.json', rows)


if __name__ == '__main__':
    main()
