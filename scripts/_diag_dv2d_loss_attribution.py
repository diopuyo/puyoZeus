"""完了した追加解除の帰属を読み、重複条件の追加調査対象を抽出する。"""
from __future__ import annotations

import json
import argparse

from scripts._diag_dv2d_loss_inventory import OUT, read, resource, save
from scripts._diag_dv2d_loss_analyze import flat, recovered


def main() -> None:
    """未実施をゼロ効果とせず、完了済みの証拠だけで残例を列挙する。"""
    print('RESOURCE', resource(), flush=True)
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='ATTRIBUTION_INTERIM.json')
    args = parser.parse_args()
    rows = read(OUT/'PARTIAL_ANALYSIS.json')['cases']
    output = []
    for row in rows:
        if row['group'] not in ('set1', 'set2'):
            continue
        detail = dict(group=row['group'], game=row['game'], record=row['record'],
                      old_recovery=row['independently_restored_by'])
        for mode in ('no_d3', 'no_d4'):
            directory = OUT/f"{mode}_{row['record']}_r2"
            if not (directory/'status.json').exists():
                continue
            events = flat(directory/'events.jsonl')
            detail[mode] = recovered(row, events)
            detail[mode+'_blocked_h1'] = recovered(row, events, history=True)
        output.append(detail)
        print(json.dumps(detail, ensure_ascii=False), flush=True)
    save(args.output, output)


if __name__ == '__main__':
    main()
