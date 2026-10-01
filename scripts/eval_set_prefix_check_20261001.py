"""測定器の検査: 第41試合からの短区間で取り直した記録 (p3check) が、既存の第3パート記録の先頭と
バイト一致するかを行単位で確かめる (2026-10-01)。

比較: 各記録の gzip を展開した JSON 行の文字列。短区間側の 'complete' 行 (フレーム数が違う) だけを除き、
短区間の全行が既存記録の同じ位置の行と一致すること。画像窓 (windows.json.gz) は短区間側の全キーが一致すること。
使い方: python -m scripts.eval_set_prefix_check_20261001
"""
from __future__ import annotations

from collections import Counter
import gzip
import json
from pathlib import Path

EXISTING = Path('/mnt/c/Users/ryouj/.codex/worktrees/exev/puyo_analyzer/logs/pending_expiry/full/records/zenchi')
CHECK = Path('logs/eval_set/collect/records/p3check')
OUT = Path('logs/eval_set/PREFIX_CHECK.json')


def compare_rows(existing: Path, check: Path) -> dict:
    """短区間の各行が既存の同位置の行と一致するか (complete 行を除く)。"""
    kinds, mismatch, first = Counter(), 0, None
    with gzip.open(existing, 'rt') as left, gzip.open(check, 'rt') as right:
        for index, (old, new) in enumerate(zip(left, right)):
            kind = json.loads(new)['kind']
            if kind == 'complete':
                kinds['complete_skipped'] += 1
                continue
            kinds[kind] += 1
            if old != new:
                mismatch += 1
                first = first or dict(index=index, kind=kind, old=old[:300], new=new[:300])
        tail = sum(1 for _ in right)
    return dict(rows_compared=sum(v for k, v in kinds.items() if k != 'complete_skipped'),
                kinds=dict(kinds), mismatched_rows=mismatch, first_mismatch=first, check_rows_unpaired=tail)


def compare_windows(existing: Path, check: Path) -> dict:
    """短区間で取った画像窓が既存の同じキーの窓と一致するか。"""
    with gzip.open(existing, 'rt') as a, gzip.open(check, 'rt') as b:
        old, new = json.load(a), json.load(b)
    missing = [k for k in new if k not in old]
    differ = [k for k in new if k in old and old[k] != new[k]]
    return dict(check_windows=len(new), existing_windows=len(old), missing=missing, differ=differ)


def main() -> None:
    """結果を PREFIX_CHECK.json に保存して表示する。"""
    rows = compare_rows(EXISTING.with_suffix('.jsonl.gz'), CHECK.with_suffix('.jsonl.gz'))
    windows = compare_windows(EXISTING.with_suffix('.windows.json.gz'), CHECK.with_suffix('.windows.json.gz'))
    passed = (rows['mismatched_rows'] == 0 and rows['rows_compared'] > 0 and rows['check_rows_unpaired'] == 0
              and not windows['missing'] and not windows['differ'] and windows['check_windows'] > 0)
    result = dict(passed=passed, rows=rows, windows=windows)
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=1))
    print(json.dumps(dict(passed=passed, rows_compared=rows['rows_compared'], mismatched=rows['mismatched_rows'],
                          windows=windows['check_windows']), ensure_ascii=False))


if __name__ == '__main__':
    main()
