"""D4の因果経路・残差内訳・後続観測を数値にまとめる。"""
from __future__ import annotations
from collections import Counter, defaultdict
import json
from pathlib import Path
import numpy as np
from scripts._d4_loss import OUT, Q, save
from scripts._d3_inventory import SOURCES
from scripts.r1_measure_helpers import lines


def origin_evidence() -> dict:
    """主因となる二連鎖の採用時刻と盤面差を保存する。"""
    rows = json.loads((OUT/f'{Q}_origins.json').read_text())
    selected = [r for r in rows if abs(r['t']-210.5666666667)<.001 or abs(r['t']-218.8)<.001]
    for row in selected:
        a, b = np.array(row['off']['board']), np.array(row['on']['board'])
        row['board_changes'] = [dict(row=int(r), col=int(c), off=int(a[r,c]), on=int(b[r,c])) for r,c in np.argwhere(a!=b)]
    save('main_origins.json', selected)
    return dict(rows=selected)


def corrections() -> dict:
    """合図別の誤修正と、189おじゃまセルの独立後続一致を数える。"""
    counts = defaultdict(Counter)
    for source in SOURCES:
        for row in lines(Path('logs/r1/followup')/f'{source}.jsonl'):
            counts[row['signal']][row['status']] += 1
    save('signal_followup.json', dict(counts))
    return dict(counts)


def residuals() -> dict:
    """同時刻で確認できる採否と過去の合図を区別して集計する。"""
    rows = json.loads((OUT/'residual_cells.json').read_text())
    groups = defaultdict(Counter)
    for row in rows:
        exact = row['age'] is not None and abs(row['age'])<1e-6
        groups[row['category']]['同時刻' if exact else '過去合図または欠測'] += 1
    result = dict(denominator=len(rows), groups=dict(groups),
        age_quantiles=np.percentile([r['age'] for r in rows if r['age'] is not None], [0,50,95,100]).tolist())
    save('residual_alignment.json', result)
    return result


def peak() -> dict:
    """最大悪化行のcount特徴・隠し段分布・物理予測を縮約する。"""
    rows = json.loads((OUT/f'{Q}_trace_examples.json').read_text())
    row = next(r for r in rows if abs(r['t']-220.2)<.001)
    result = dict(t=row['t'])
    for mode in ('off', 'on'):
        value = row[mode]
        result[mode] = dict(counts=value['counts'], hidden=value['hidden'],
            chains=[{k:c[k] for k in ('side','chain_id','trigger_sec','predicted_final_score','predicted_chain_count','score_delta')} for c in value['chains']])
    save('main_peak.json', result)
    return result


def main() -> None:
    """保存された数値だけで最終報告用の検査可能な原票を生成する。"""
    values = dict(origins=origin_evidence(), followup=corrections(), residual=residuals(), peak=peak())
    for name, value in values.items():
        print(name, json.dumps(value, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
