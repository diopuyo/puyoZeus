"""抽出原票の代表時刻と安全条件を表示する。"""
from __future__ import annotations

import csv
import json
from scripts._diag_review1003b_extract import OUT, resources

TIMES = {4: [2748, 2751.5, 2755, 2760, 2762, 2762.3],
         5: [4240, 4242, 4244, 4246],
         6: [4261, 4262.5, 4265, 4270, 4272, 4273],
         7: [6049, 6050, 6052, 6055, 6060, 6061]}
VALUES = ['t_sec', 'new_evaluator', 'old_evaluator', 'new_p1', 'old_p1',
          'p1_G_fe', 'p1_S1', 'p1_S3_provisional', 'p1_S3', 'p1_landing_gfe', 'p1_combined']
SIDES = ['state', 'confirmed_sec', 'chain_count', 'formula_total', 'formula_send',
         'forecast', 'incoming', 'd_current_max_chain', 'NF_ojama_k1', 'NF_ojama_k2',
         'NF_ojama_k3', 'NF_ojama_k4', 'NF_ojama_k5', 'counter_margin']


def main() -> None:
    """代表行に対応する直近の着弾評価を保存する。"""
    resources()
    result = []
    for case, times in TIMES.items():
        rows = list(csv.DictReader((OUT / f'case{case}_frames.csv').open()))
        events = json.loads((OUT / f'case{case}_events.json').read_text())
        old = json.loads((OUT / f'case{case}_old_events.json').read_text())
        for time in times:
            row = min(rows, key=lambda r: abs(float(r['t_sec']) - time))
            values = [v for e in events for v in e['values']
                      if v['t_sec'] <= float(row['t_sec']) and 'incoming' in v]
            olds = [v for e in old for v in e['values']
                    if v['t_sec'] <= float(row['t_sec']) and 'incoming' in v]
            entry = dict(case=case, values={k: row.get(k) for k in VALUES},
                sides=[{k: row.get(f'{s}_{k}') for k in SIDES} for s in ('1P', '2P')],
                landing=max(values, key=lambda v: v['t_sec']) if values else None,
                old_landing=max(olds, key=lambda v: v['t_sec']) if olds else None)
            result.append(entry)
    (OUT / 'representatives.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))
    for entry in result:
        print(json.dumps(entry, ensure_ascii=False))


if __name__ == '__main__':
    main()
