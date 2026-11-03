"""保存入力を順に読み、盤面・NEXT・保存予告量の同一性を数える。"""
from __future__ import annotations

import sys as _public_release_sys
from pathlib import Path as _PublicReleasePath
_public_release_root = str(_PublicReleasePath(__file__).resolve().parents[0])
if _public_release_root not in _public_release_sys.path:
    _public_release_sys.path.insert(0, _public_release_root)
from src.public_dev_paths import resolve_text as _public_dev_text


import gzip
import hashlib
import json
from pathlib import Path

SOURCE = Path('C:/PuyoLive_20261012e/PuyoLive/output/inputs.jsonl.gz')
OUT = Path(_public_dev_text('<work>/speed3/input_duplicates.json'))
SIDES = ('p1', 'p2')
WARNINGS = ('net_balance_capped', 'forecast_p1')


def main() -> None:
    seen = {key: set() for key in (*SIDES, 'pair')}
    counts = dict.fromkeys(seen, 0)
    updates, complete = 0, False
    with gzip.open(SOURCE, 'rt', encoding='utf-8') as stream:
        while True:
            try:
                line = stream.readline()
            except EOFError:
                break
            if not line:
                break
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                break
            complete |= row['kind'] == 'complete'
            if row['kind'] != 'update':
                continue
            args = row['args']['tuple']
            result, snapshot = args[0]['namespace'], args[1]['namespace']
            warning = {key: snapshot[key] for key in WARNINGS}
            values = {side: ({key: result[side]['namespace'][key] for key in
                              ('confirmed_board', 'next_pair', 'dnext_pair')}, warning) for side in SIDES}
            values['pair'] = tuple(values[side] for side in SIDES)
            for key, value in values.items():
                digest = hashlib.sha256(json.dumps(value, sort_keys=True).encode()).digest()
                counts[key] += digest in seen[key]
                seen[key].add(digest)
            updates += 1
    value = dict(updates=updates, complete=complete, duplicate=counts,
                 unique={k: len(v) for k, v in seen.items()}, warning_fields=WARNINGS,
                 note='保存snapshotの純予告量・forecastを使用。画像の予告アイコン自体の一致率ではない。')
    OUT.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(value, ensure_ascii=False))


if __name__ == '__main__':
    main()
