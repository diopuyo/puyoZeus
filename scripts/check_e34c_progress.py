"""収集中gzipの書込み済み行を検査する。完了判定には使用しない。"""
from __future__ import annotations

import gzip
import json

from scripts.collect_e34c import InputCheck, OUT
from scripts.run_e3_exchange_eval_20260926 import save_json


def main() -> None:
    """圧縮ストリーム末尾の未書込み部分だけを許容し、差分を保存する。"""
    for path in sorted((OUT/'records').glob('*.jsonl.tmp')):
        source = path.name.removesuffix('.jsonl.tmp')
        check = InputCheck(source, checkpoint=False)
        try:
            with gzip.open(path, 'rt', encoding='utf-8') as stream:
                for line in stream:
                    row = json.loads(line)
                    if row['kind'] == 'update':
                        check.observe(row)
        except EOFError:
            pass
        value = check.summary()
        save_json(OUT/'checks'/f'{source}.progress.json', value)
        print({key: val for key, val in value.items() if key != 'first'}, flush=True)


if __name__ == '__main__':
    main()
