"""探索診断とFIX4基準の判定不変照合を保存する。"""
from __future__ import annotations

import sys as _public_release_sys
from pathlib import Path as _PublicReleasePath
_public_release_root = str(_PublicReleasePath(__file__).resolve().parents[0])
if _public_release_root not in _public_release_sys.path:
    _public_release_sys.path.insert(0, _public_release_root)
from src.public_dev_paths import resolve_text as _public_dev_text


import json
from pathlib import Path
from typing import Any

import numpy as np

OUT = Path(_public_dev_text('<work>/speed3'))
REFERENCE = Path(_public_dev_text('<work>/fix4/reference_replay_pair'))
PERCENTILES = (50, 95, 99, 100)
COST_FIELDS = ('elapsed_sec', 'enumeration_sec', 'filter_sec')
AUDITS = ('post_counter_bound_audit.json', 'hidden_death_audit.json',
          'midchain_audit.json', 'prefire_audit.json', 'completion_score_audit.json')


def stats(values: list[float]) -> dict:
    return dict(n=len(values), total=sum(values),
                p=np.percentile(values, PERCENTILES).tolist() if values else [])


def without_cost(value: Any) -> Any:
    """FIX4と同じ費用列のみ除く。証明と動画時刻は省略しない。"""
    if isinstance(value, list):
        return [without_cost(v) for v in value]
    if isinstance(value, dict):
        return {k: without_cost(v) for k, v in value.items() if k not in COST_FIELDS}
    return value


def compare(actual: Path) -> dict:
    result = {}
    for name in ('events.jsonl', 'history_integrity_audit.json'):
        assert (REFERENCE/name).read_bytes() == (actual/name).read_bytes(), name
        result[name] = 'byte_identical'
    for name in AUDITS:
        a, b = [json.loads((p/name).read_text(encoding='utf-8')) for p in (REFERENCE, actual)]
        assert without_cost(a) == without_cost(b), name
        result[name] = 'identical_except_cost'
    with np.load(REFERENCE/'display.npz') as a, np.load(actual/'display.npz') as b:
        assert a.files == b.files
        for key in a.files:
            assert a[key].dtype == b[key].dtype and a[key].tobytes() == b[key].tobytes(), key
        result['display_columns'] = {key: list(a[key].shape) for key in a.files}
    return result


def summarize(data: dict) -> dict:
    calls, updates = data['calls'], data['updates']
    result = dict(input_n=data['input_n'], input_duplicates=data['input_duplicates'])
    for group in ('all', 'chain', 'ojama', 'other'):
        selected = [r for r in updates if group == 'all' or r['group'] == group]
        result[group] = {key: stats([r[key] for r in selected]) for key in
                         ('update_ms', 'reference_ms', 'search_ms')}
    result['searches'] = {}
    for name in sorted({c['name'] for c in calls}):
        selected = [r for r in calls if r['name'] == name]
        result['searches'][name] = dict(n=len(selected), ms=stats([r['ms'] for r in selected]),
            repeats=sum(r['repeated'] for r in selected), hits=sum(r['hit'] for r in selected),
            reusable_ms=sum(r['ms'] for r in selected if r['hit']))
    return result


def main() -> None:
    data = json.loads((OUT/'replay_diagnostic.json').read_text(encoding='utf-8'))
    result = summarize(data)
    result['invariance'] = compare(OUT/'reference_replay_1')
    (OUT/'replay_summary.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
