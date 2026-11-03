"""保存された全ツモ履歴で候補特定を比較する。単体時間は正式な配信速度と区別する。"""
from __future__ import annotations

import sys as _public_release_sys
from pathlib import Path as _PublicReleasePath
_public_release_root = str(_PublicReleasePath(__file__).resolve().parents[0])
if _public_release_root not in _public_release_sys.path:
    _public_release_sys.path.insert(0, _public_release_root)
from src.public_dev_paths import resolve_text as _public_dev_text

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
from time import perf_counter, process_time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'packaging')]
from src.ama_tsumo_seed_v5 import identify, load_table
from src.phase_j.live_ama_seed import identify_exact
from summarize_live_obs_time_gate import stats
from smoke_live_obs_20261020 import memory_guard


def records() -> dict:
    result = {}
    for stem in ('80_g7', '1042_g1'):
        path = Path(_public_dev_text(f'<work>/wt_ama_eval/logs/ama_eval_v71/inputs_{stem}.json'))
        for row in json.loads(path.read_text(encoding='utf-8')):
            if 'bundle' not in row:
                continue
            history, colors = row['bundle']['observed_history'], row['request']['colors']
            key = repr((history, colors))
            if key in result:
                assert result[key]['expected'] == row['bundle']['identification']
            result[key] = dict(history=history, colors=colors, expected=row['bundle']['identification'])
    return result


def compare(table: object, index: int, record: dict) -> dict:
    costs, outputs = {}, {}
    pairs = [('before', identify), ('after', identify_exact)]
    for name, function in pairs[::1 if index % 2 else -1]:
        wall, cpu = perf_counter(), process_time()
        outputs[name] = asdict(function(table, record['history'], record['colors']))
        costs[name] = dict(wall_ms=(perf_counter()-wall)*1000, cpu_ms=(process_time()-cpu)*1000)
    assert outputs['before'] == outputs['after'] == record['expected'], (index, outputs, record)
    return dict(index=index, identification=outputs['after'], **costs)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(exist_ok=False)
    memory_guard()
    table = load_table(ROOT/'models/ama_plain/seed_table_cache.npz')
    _ = table.tape_codes  # 両案に共通する一度だけの表展開を先に行う。
    rows = []
    for index, record in enumerate(records().values()):
        memory_guard()
        rows.append(compare(table, index, record))
        if index % 25 == 0:
            print('seed', index, flush=True)
    result = dict(unique_histories=len(rows), mismatches=0, all_fields_exact=True,
        timings={name: {key: stats([r[name][key] for r in rows]) for key in ('wall_ms', 'cpu_ms')}
                 for name in ('before', 'after')},
        note='CPU12〜15・BelowNormalの単体参考値。全表展開後、交互の順で同一入力を比較。')
    (args.out/'rows.json').write_text(json.dumps(rows, indent=2), encoding='utf-8')
    (args.out/'receipt.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(result, flush=True)


if __name__ == '__main__':
    main()
