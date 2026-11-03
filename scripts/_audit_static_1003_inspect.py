"""静止勝率監査の入力形式を読み取り専用で確認する。"""
from pathlib import Path
import gzip
import json
import os
import numpy as np

BASE = Path('/mnt/d/puyo_analyzer/wt_evalset/logs/eval_set/falsedeath_fix')


def main() -> None:
    """小さい標本だけ表示し、入力資産には書き込まない。"""
    os.nice(max(0, 10 - os.nice(0)))
    for variant in ('v2h2b', 'prod'):
        root = BASE / 'replay' / variant / 's1'
        print(variant, (root / 'COMMAND.json').read_text())
        with np.load(root / 'display.npz') as data:
            print({k: (data[k].shape, str(data[k].reshape(-1)[:3])) for k in data.files})
    seen: set[str] = set()
    with gzip.open(BASE / 'collect/records/s1.jsonl.gz', 'rt') as stream:
        for line in stream:
            row = json.loads(line)
            if row['kind'] not in seen:
                print(row['kind'], line[:12000])
                seen.add(row['kind'])
            if len(seen) >= 6:
                break


if __name__ == '__main__':
    main()
