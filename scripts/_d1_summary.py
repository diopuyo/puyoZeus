"""D1の根拠と未確定を分離して集計する。"""
from __future__ import annotations
from collections import Counter
import json
from scripts._d1_inventory import OUT, SOURCES
from scripts._d1_count import cells

NON_GAME = {('q_7gc4TgFig',128.9),('q_7gc4TgFig',661.9666666666667),('mia8KCjr52g',703.9)}
SIMULATION = {('q_7gc4TgFig',483.1666666666667),('q_7gc4TgFig',780.),('fcXG83vInDY',523.5666666666667),('fcXG83vInDY',862.5666666666667)}
MIDCHAIN = {('fcXG83vInDY',238.4)}
COLORS = {0:'空',1:'赤',2:'青',3:'緑',4:'黄',5:'紫',9:'おじゃま',10:'不明'}


def supplement_pairs(rows: list[dict], transitions: list[dict]) -> None:
    """連鎖開始の前後で元盤面への2セル追加を照合できる組だけ補う。"""
    for row in rows:
        if (row['source'],row['trigger']) not in SIMULATION:
            continue
        evidence = next(t for t in transitions if t['source']==row['source'] and t['trigger']==row['trigger'])
        trans = evidence['transitions'][-1]
        before, snapshot = trans['before'],evidence['snapshot']
        added = cells(before,snapshot)
        if len(added)!=2 or not all(before[r][c]==0 and 1<=snapshot[r][c]<=5 for r,c in added):
            continue
        for cell in row['cells']:
            if (cell['row'],cell['col']) in added:
                if not cell['final_pair']:
                    cell['pair_evidence'] = '連鎖直前の保存盤面との差が2セル追加のみ'
                cell['final_pair'] = True


def classify(row: dict, cell: dict) -> str:
    """書込み元未保存のセルを経路確定済として扱わない。"""
    key = (row['source'],row['trigger'])
    if cell['final_pair']:
        return '最終組（特定済み）'
    if key in NON_GAME:
        return '試合外画像（W47系）'
    if key in SIMULATION:
        return '連鎖後盤面の先取り（W48整合）'
    if key in MIDCHAIN:
        return '連鎖途中通知・時点不一致'
    if cell['agreement']*2 <= cell['frames']:
        return 'CNN/HSV非一致'
    if cell['origin']==0:
        return '経路未確定：欠落'
    if cell['snapshot']==0:
        return '経路未確定：過剰'
    return '経路未確定：色違い'


def main() -> None:
    """全発火と差分セルの排他的な件数表を保存する。"""
    rows = json.loads((OUT/'differences.json').read_text())
    transitions = json.loads((OUT/'transitions.json').read_text())
    supplement_pairs(rows,transitions)
    counts, records, per_source = Counter(), [], {}
    for source in SOURCES[:-1]:
        group = [r for r in rows if r['source']==source]
        tally = Counter()
        for row in group:
            for cell in row['cells']:
                label = classify(row,cell)
                tally[label] += 1
                records.append(dict(source=source,game=row['game'],side=row['side'],trigger=row['trigger'],category=label,**cell))
        counts.update(tally)
        per_source[source] = dict(fires=len(group),measurable=sum('unmeasurable' not in r for r in group),counts=dict(tally),
            zero=sum('unmeasurable' not in r and not r['cells'] for r in group))
    result = dict(fires=sum(x['fires'] for x in per_source.values()),per_source=per_source,counts=dict(counts),cells=records)
    assert result['fires']==221 and sum(counts.values())==335
    (OUT/'classified.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k!='cells'},ensure_ascii=False,indent=2))
    for row in transitions:
        if (row['source'],row['trigger']) not in SIMULATION:
            continue
        for trans in row['transitions'][-2:]:
            diffs = cells(trans['before'],row['snapshot'])
            print('pair_from_before',row['source'],row['trigger'],trans['t'],diffs)


if __name__ == '__main__':
    main()
