"""保存入力の構造と代表時刻を表示する診断補助。"""
from __future__ import annotations

import csv
import json
from scripts._diag_review1003_extract import ROOT, OUT, records, resources

TIMES = {2: [210, 212, 214, 216, 218, 220, 222, 224],
         13: [943, 945.7, 946.7, 947.2, 948.7, 949.7, 950],
         26: [1725, 1727, 1729, 1731, 1733, 1735]}
VALUE_COLUMNS = ['t_sec', 'new_evaluator', 'p1_G_fe', 'p1_S1', 'p1_S3_provisional',
    'p1_S3', 'p1_landing_gfe', 'p1_combined', 'p1_selected', 'p1_display']
SIDE_COLUMNS = ['state', 'confirmed_sec', 'chain_count', 'formula_total', 'formula_send',
    'confirmed_send', 'net_send', 'forecast', 'incoming', 'd_current_max_chain',
    'NF_ojama_k1', 'NF_ojama_k2', 'NF_ojama_k3', 'NF_ojama_k4', 'NF_ojama_k5',
    'counter_margin']


def inspect(game: int, times: list[float]) -> None:
    """代表行と直前イベントを出し、入力記録を診断領域へ複製する。"""
    rows = list(csv.DictReader((OUT / f'game{game:02}_frames.csv').open()))
    events = json.loads((OUT / f'game{game:02}_events.json').read_text())
    inputs = records(ROOT / 'work' / f'set1_game{game:02}' / 'inputs.jsonl.gz')
    print('GAME', game, 'input samples', str(inputs[1:3])[:2500])
    for event in events:
        print('CHAINS', [{k: v for k, v in c.items() if k != 'predicted_final_board'}
                         for c in event['chains']])
    for time in times:
        row = min(rows, key=lambda r: abs(float(r['t_sec']) - time))
        print('VALUE', [row[k] for k in VALUE_COLUMNS])
        for side in ('1P', '2P'):
            print(side, [row[f'{side}_{k}'] for k in SIDE_COLUMNS])
        values = [v for e in events for v in e['values']
                  if v['t_sec'] <= float(row['t_sec']) and 'incoming' in v]
        if values:
            value = max(values, key=lambda v: v['t_sec'])
            print('LAND', {k: value.get(k) for k in ('t_sec', 'incoming', 'death_incoming',
                'state_safety', 'required_cancel', 'near_future_send', 'optimistic_send',
                'completion_certain', 'midchain_prediction', 'history_integrity')})
    (OUT / f'game{game:02}_all_inputs.json').write_text(json.dumps(inputs), encoding='utf-8')


def main() -> None:
    """リソースを確認して代表値を表示する。"""
    resources()
    print('VALUE_COLUMNS', VALUE_COLUMNS)
    print('SIDE_COLUMNS', SIDE_COLUMNS)
    for game, times in TIMES.items():
        inspect(game, times)


if __name__ == '__main__':
    main()
