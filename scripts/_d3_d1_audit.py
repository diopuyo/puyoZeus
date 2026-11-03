"""D1の96セルを、純追加に限定しない元STABLE書換えまで遡る。"""
from __future__ import annotations

from bisect import bisect_left
import json
import os
from scripts._d3_candidates import load
from scripts._d3_inventory import OUT, SOURCES
from scripts._d3_measure import observations, point, reference


def trace(cell: dict, rows: list[dict], events: list[dict], obs: dict, fps: float) -> dict:
    """画像支持付き書換え→同周期の合図との差→D1までの値保持を照合する。"""
    result = dict(**cell, direct_matches=[], reason='同一周期内の画像支持付き早期書換え未特定')
    if cell['last_change_sec'] is None:
        return result
    stamp, side = cell['last_change_sec'], cell['side']
    r, c, old, target = cell['row'], cell['col'], cell['origin'], cell['snapshot']
    if old == 9 or target == 9:
        return dict(result, reason='色組ぷよ以外のおじゃま差分')
    index = bisect_left([row['t'] for row in rows], stamp-1e-6)
    visual = point(obs, stamp, side, fps)
    if visual is None or visual[r, c] != old:
        return dict(result, reason='最終書換え時のCNN単独=HSV単独による保存色支持なし')
    if rows[index]['sides'][side]['state']['state'] != 'STABLE':
        return dict(result, reason='最終書換えがSTABLEでない')
    before = rows[max(0, index-1)]['sides'][side]['confirmed_board']
    current = rows[index]['sides'][side]['confirmed_board']
    added = before is not None and current is not None and any(
        before[rr][cc] != current[rr][cc] and 1 <= current[rr][cc] <= 5 and visual[rr, cc] == current[rr][cc]
        for rr in range(1, 13) for cc in range(6))
    if not added:
        return dict(result, reason='同時刻の画像支持付き色ぷよ書込み証拠なし')
    for event in events:
        if not (event['cycle_start'] < stamp < event['t'] <= cell['trigger']):
            continue
        board, frame = reference(event, obs, fps)
        if board is None or board[r, c] != target:
            continue
        span = [row for row in rows[index:] if row['t'] <= cell['trigger']]
        if not all(row['sides'][side]['confirmed_board'] is not None and
                   row['sides'][side]['confirmed_board'][r][c] == old for row in span):
            continue
        result['direct_matches'].append(dict(write_t=stamp, signal_t=event['t'],
            signal_type=event['type'], reference_frame=frame,
            same_cycle=True, uninterrupted_to_d1=True))
    return result


def main() -> None:
    """96セルの陽性・未特定をすべて保存する。"""
    os.nice(19)
    residual = json.loads((OUT.parent/'d1/classified.json').read_text())['cells']
    output = []
    for source in SOURCES[:3]:
        rows, (obs, fps) = load(source), observations(source)
        groups = json.loads((OUT/f'{source}_candidates.json').read_text())
        for cell in residual:
            if cell['source'] == source and cell['category'].startswith('経路未確定'):
                output.append(trace(cell, rows, groups[cell['side']]['events'], obs, fps))
    assert len(output) == 96
    (OUT/'d1_write_audit.json').write_text(json.dumps(output, ensure_ascii=False, indent=2))
    print('D1 direct', sum(bool(c['direct_matches']) for c in output), '/96', flush=True)


if __name__ == '__main__':
    main()
