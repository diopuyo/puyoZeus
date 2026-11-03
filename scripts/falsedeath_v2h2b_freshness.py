"""D-v2h2とD-v2h2bの鮮度切れを、同じ18入力の実監査から数える。"""
from __future__ import annotations

import json
from collections import Counter

from scripts.falsedeath_replay_20261001 import OUT, records
from scripts.falsedeath_v2d_report import table

DEST = OUT/'v2h2b'
MODES = ('v2h2', 'v2h2b')


def counts(mode: str, source: str) -> dict:
    """監査呼出しを母数とし、最終段観測と未完走・欠測を区別する。"""
    path = OUT/'replay'/mode/source/'post_counter_bound_audit.json'
    audit = json.loads(path.read_text())['rows']
    result: Counter = Counter()
    for row in audit:
        check = row.get('completion_prefix_check', {})
        if 'fresh' not in check:
            continue
        result['checks'] += 1
        observed = check['observed_latest_stage']
        predicted = check['predicted_chain_count']
        complete = observed is not None and (predicted or 0) > 0 and observed >= predicted
        stale = check['stage_lag'] is not None and not check['fresh']
        result['completed'] += complete
        result['stale'] += stale
        result['completed_stale'] += complete and stale
        result['incomplete_stale'] += not complete and stale
        result['unknown'] += check['stage_lag'] is None
        result['accepted'] += check['accepted']
    return {key: result[key] for key in ('checks', 'completed', 'stale', 'completed_stale',
                                       'incomplete_stale', 'unknown', 'accepted')}


def main() -> None:
    """全件表と差分を保存し、再生経路変化による母数差も明示する。"""
    data = {source: {mode: counts(mode, source) for mode in MODES} for source in records()}
    data['total'] = {mode: {key: sum(row[mode][key] for row in data.values())
                           for key in next(iter(data.values()))[mode]} for mode in MODES}
    for row in data.values():
        row['delta'] = {key: row['v2h2b'][key]-value for key, value in row['v2h2'].items()}
    text = table('鮮度切れの変化（監査呼出し数）',
        ['記録', '構成', '照合数', '最終段観測済', '鮮度切れ', '最終段観測済の鮮度切れ', '途中の鮮度切れ', '時刻・段欠測', '照合許可'],
        [[source, mode, *row[mode].values()] for source, row in data.items()
         for mode in (*MODES, 'delta')])
    text += ('最終段観測済は観測最大段 ≥ 予測段数 > 0。鮮度切れは段差が存在しfresh=Falseの呼出し。'
             '途中段の連続性・増分一致・conflicts検査は別途維持する。'
             '18記録単純合計は重複区間を含む。再生経路が変わるため各構成の呼出し母数も併記する。\n\n')
    with (DEST/'FRESHNESS.json').open('x') as stream:
        json.dump(data, stream, ensure_ascii=False, indent=1)
    with (DEST/'REPORT.md').open('a') as stream:
        stream.write(text)
    print(json.dumps(data['total']))


if __name__ == '__main__':
    main()
