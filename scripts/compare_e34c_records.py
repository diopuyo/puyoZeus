"""完成原票の再生入力と、再生には使わない評価履歴を区別して照合する。"""
from __future__ import annotations

from collections import Counter
import argparse
import gzip
import hashlib
from itertools import zip_longest
import json
from pathlib import Path
from typing import Iterator

OUT = Path('logs/e34c')
KINDS = ('update', 'display', 'static', 'static_call', 'complete')
INPUT_KINDS = ('update', 'display', 'complete')


def rows(path: Path, kind: str) -> Iterator[dict]:
    """保持印だけを外し、他の列と欠測表現をそのまま保持する。"""
    with gzip.open(path, 'rt', encoding='utf-8') as stream:
        for line in stream:
            row = json.loads(line)
            if row['kind'] != kind:
                continue
            if kind == 'update':
                for side in ('p1', 'p2'):
                    row['args']['tuple'][0]['namespace'][side]['namespace'].pop('prefire_origin_hold', None)
            yield row


def compare(source: str, partial: bool = False) -> dict:
    """ヘッダーの実験メタデータ以外は全列と母数を照合する。"""
    result = {}
    for kind in KINDS if not partial else KINDS[1:-1]:
        old = rows(Path('logs/e31/records')/f'{source}.jsonl.gz', kind)
        suffix = 'tmp' if partial else 'gz'
        new = rows(OUT/'records'/f'{source}.jsonl.{suffix}', kind)
        count, first = Counter(), None
        try:
            pairs = zip(old, new) if partial else zip_longest(old, new)
            for index, (a, b) in enumerate(pairs, 1):
                count.update(old=int(a is not None), new=int(b is not None), matching=int(a == b))
                if a != b and first is None:
                    first = dict(index=index, old=a, new=b)
        except EOFError:
            if not partial:
                raise
        result[kind] = dict(count, first=first,
                            passed=count['old'] == count['new'] == count['matching'])
    return result


def feature_cache(source: str) -> dict:
    """再生器が参照するD特徴だけをキーで照合する。M0と経過秒は再計算される。"""
    paths = [Path('logs/e31/records'), OUT/'records']
    caches = [{row['key']: json.dumps(row['input']['d_features'], sort_keys=True)
               for row in rows(path/f'{source}.jsonl.gz', 'static')} for path in paths]
    old, new = caches
    common = old.keys() & new.keys()
    changed = [key for key in common if old[key] != new[key]]
    return dict(common=len(common), matching=len(common)-len(changed), changed=changed,
                old_only=len(old.keys()-new.keys()), new_only=len(new.keys()-old.keys()),
                passed=not changed)


def completed_source(source: str) -> dict:
    """完成した原票を検収し、入力と検査コードのハッシュが同じ場合だけ再利用する。"""
    paths = [Path('logs/e31/records')/f'{source}.jsonl.gz',
             OUT/'records'/f'{source}.jsonl.gz', Path(__file__)]
    hashes = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    dest = OUT/'checks'/f'{source}.payload.json'
    if dest.exists():
        previous = json.loads(dest.read_text())
        if previous['sha256'] == hashes:
            return previous
    value = dict(records=compare(source), feature_cache=feature_cache(source), sha256=hashes)
    dest.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    return value


def main() -> None:
    """全記録の完成後に完全一致の判定を保存する。"""
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--partial-source')
    modes.add_argument('--source')
    args = parser.parse_args()
    if args.source:
        value = completed_source(args.source)
        print(dict(records={k: {n: v for n, v in row.items() if n != 'first'}
                            for k, row in value['records'].items()}, feature_cache=value['feature_cache']))
        return
    if args.partial_source:
        result = compare(args.partial_source, partial=True)
        (OUT/'checks'/f'{args.partial_source}.aux_progress.json').write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
        print({kind: {k: v for k, v in value.items() if k != 'first'} for kind, value in result.items()})
        return
    checks = json.loads((OUT/'INPUT_CHECK.json').read_text())
    completed = {source: completed_source(source) for source in checks['sources']}
    result = {source: value['records'] for source, value in completed.items()}
    checks['record_kinds'] = result
    checks['feature_cache'] = {source: value['feature_cache'] for source, value in completed.items()}
    checks['passed'] = all(v['passed'] for v in checks['sources'].values()) and all(
        kinds[kind]['passed'] for kinds in result.values() for kind in INPUT_KINDS)
    checks['passed'] = checks['passed'] and all(v['passed'] for v in checks['feature_cache'].values())
    (OUT/'INPUT_CHECK.json').write_text(json.dumps(checks, ensure_ascii=False, indent=2), encoding='utf-8')
    print({source: {kind: {k: v for k, v in value.items() if k != 'first'}
                    for kind, value in kinds.items()} for source, kinds in result.items()})


if __name__ == '__main__':
    main()
