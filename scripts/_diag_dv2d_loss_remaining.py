"""単独解除で残った事例の、複数条件による抑止を調べる。"""
from __future__ import annotations

import json

from scripts._diag_dv2d_loss_inventory import OUT, read, save, resource
from scripts._diag_dv2d_loss_analyze import flat, recovered


def main() -> None:
    """一つ抜き再生中にH1が止めた証明を独立して列挙する。"""
    save('RESOURCE_remaining.json', resource())
    rows = [r for r in read(OUT/'PARTIAL_ANALYSIS.json')['cases']
            if r['group'] in ('set1','set2') and not r['independently_restored_by']]
    results = []
    for row in rows:
        combinations = {}
        for mode in ('no_d1','no_e35'):
            values = flat(OUT/f"{mode}_{row['record']}_r2/events.jsonl")
            combinations[mode+'+no_h1'] = recovered(row, values, history=True)
        result = dict(group=row['group'], game=row['game'], record=row['record'],
            local_game=row['local_game'], first=row['first_reasons'], combinations=combinations,
            sequence=[r['signature'] for r in row['reason_series']])
        results.append(result)
        print(json.dumps({k:v for k,v in result.items() if k!='sequence'}, ensure_ascii=False), flush=True)
    save('REMAINING.json', results)


if __name__ == '__main__':
    main()
