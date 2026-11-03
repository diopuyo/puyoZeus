"""主因セルの画像観測・公開盤面と、残差の公開時刻を照合する。"""
from __future__ import annotations
from pathlib import Path
import json
from scripts._d4_loss import OUT, Q, save
from scripts.r1_measure_helpers import lines, records
from scripts._d3_candidates import load
from scripts._d3_inventory import SOURCES

WINDOW = (217.9, 219.)
CELLS = ((2,1), (5,2))


def main_timeline() -> list[dict]:
    """主因の2P赤修正と、後続の紫発火点を別々に追う。"""
    result = []
    old, new = load(Q), records(Q)
    for row in lines(Path('logs/r1/followup')/f'{Q}.observations.jsonl.gz'):
        if not WINDOW[0] <= row['t'] <= WINDOW[1]:
            continue
        obs = row['sides'][1]
        index = min(len(old)-1, round(row['t']*30))
        values = dict(t=row['t'], frame=row['frame'], quality=obs['quality'], erasing=obs['erasing'])
        values.update(cnn=[obs['cnn'][r][c] for r,c in CELLS], hsv=[obs['hsv'][r][c] for r,c in CELLS])
        for mode, rows in (('off',old),('on',new)):
            side = rows[index]['sides'][1]
            values[mode] = dict(state=side['state']['state'], cells=[side['confirmed_board'][r][c] for r,c in CELLS])
        result.append(values)
    save('q_critical_cell_timeline.json', result)
    return result


def residual_publication() -> list[dict]:
    """合図採用済みでも固定評価行へまだ反映されないセルを区別する。"""
    residual = json.loads((OUT/'residual_cells.json').read_text())
    result = []
    for source in SOURCES:
        targets = [r for r in residual if r['source']==source and r['category']=='修正後に残存・再変化']
        rows = records(source) if targets else []
        for target in targets:
            r,c,side = target['row'],target['col'],target['side']
            following = rows[target['i']:target['i']+3]
            result.append(dict(target, publication=[dict(t=x['t'], value=x['sides'][side]['confirmed_board'][r][c]) for x in following]))
    save('residual_publication.json', result)
    return result


def main() -> None:
    """画像の独立一致と公開の順序を分けて記録する。"""
    print(json.dumps(main_timeline(), ensure_ascii=False), flush=True)
    print(json.dumps(residual_publication(), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
