"""実画像のNEXT移動を周期境界とし、3合図の最初を対応付ける。"""
from __future__ import annotations

from bisect import bisect_right
import json
import os
import numpy as np
from scripts._d3_candidates import load
from scripts._d3_inventory import OUT, SOURCES
from scripts._d3_measure import observations, point

FPS = 30
LATE_WRITE_FRAMES = 8
START = 2580.6
OJAMA_FOLLOW_SEC = .5
OJAMA_STATIC_CONFIRM_SEC = .15
OJAMA_FOLLOW_COUNT = 2


def falling_ojama(before: np.ndarray, onset: np.ndarray, stamp: float,
                  side: int, obs: dict, fps: float) -> bool:
    """一過性の発光を除き、同列への下降か、接地後の残存で裏付ける。"""
    entries = np.argwhere((onset[1:3] == 9) & (before[1:3] >= 0) & (before[1:3] != 9)) + [1, 0]
    for row, col in entries:
        count, descended, last = 0, False, stamp
        for frame in range(round(stamp*fps)+1, round((stamp+OJAMA_FOLLOW_SEC)*fps)+1):
            current = obs.get((frame, side))
            if current is None:
                continue
            new = np.flatnonzero((current[:, col] == 9) & (before[:, col] >= 0) & (before[:, col] != 9))
            new = new[new >= row]
            if len(new):
                count += 1
                descended |= bool(np.any(new > row))
                last = frame/fps
        if count >= OJAMA_FOLLOW_COUNT and (descended or last-stamp >= OJAMA_STATIC_CONFIRM_SEC):
            return True
    return False


def image_ojama(group: dict, rows: list[dict], obs: dict, fps: float) -> list[dict]:
    """状態名だけでは数えず、可視上端への独立一致おじゃま流入を要求する。"""
    result = []
    for signal in group['signals']:
        if signal['type'] != 'ojama' or signal['repeated']:
            continue
        side, i = signal['side'], signal['i']
        previous = None
        for index in range(max(0, i-8), min(len(rows), i+2)):
            grid = point(obs, rows[index]['t'], side, fps)
            if grid is None:
                continue
            if previous is not None and falling_ojama(previous, grid, rows[index]['t'], side, obs, fps):
                result.append(dict(signal, t=rows[index]['t'], i=index,
                    image_evidence='CNN単独=HSV単独のおじゃま上端流入'))
                break
            previous = grid
    return result


def physical_signals(scan: dict, rows: list[dict], side: int) -> tuple[list, list]:
    """原フレームの実移動と式出現を、保存記録の直前時点へ対応付ける。"""
    stamps = [r['t'] for r in rows]
    nexts, formulas = [], []
    for kind, key, destination in [('NEXT', 'next_motions', nexts), ('formula', 'intervals', formulas)]:
        for entry in scan[key]:
            if entry['side'] != side or not stamps[0] <= entry['t'] <= stamps[-1]:
                continue
            index = bisect_right(stamps, entry['t'])-1
            state = rows[index]['sides'][side]['state']['state']
            if state == 'MENU':
                continue
            destination.append(dict(entry, i=index, type=kind, state=state,
                game=rows[index]['game'], repeated=False, during_chain=False))
    return sorted(nexts, key=lambda e: e['t']), sorted(formulas, key=lambda e: e['t'])


def cycles(group: dict, rows: list[dict], scan: dict, obs: dict, fps: float) -> list[dict]:
    """NEXTから次のNEXTまでの同一組について、最初の合図を一度だけ採る。"""
    side = group['signals'][0]['side']
    nexts, formulas = physical_signals(scan, rows, side)
    ojama = image_ojama(group, rows, obs, fps)
    outputs, lower = [], 0
    lower_t = rows[0]['t']
    for number, closing in enumerate(nexts):
        if rows[lower]['game'] != closing['game']:
            lower = next(i for i in range(lower, closing['i']+1) if rows[i]['game'] == closing['game'])
            lower_t = rows[lower]['t']
        signs = [e for e in formulas+ojama if lower_t < e['t'] <= closing['t'] and e['game'] == closing['game']]
        first = min([closing, *signs], key=lambda e: e['t'])
        upper = min(len(rows)-1, first['i']+LATE_WRITE_FRAMES)
        event = dict(first, cycle_start=lower_t, cycle_end=closing['t'],
            cycle_number=number, cycle_left_censored=number == 0,
            lower=lower, upper=upper, signals=sorted([closing, *signs], key=lambda e: e['t']),
            write_indices=[n for n, w in enumerate(group['writes']) if lower <= w['i'] <= upper and w['t'] > lower_t])
        outputs.append(event)
        lower, lower_t = closing['i']+1, closing['t']
    # 区間末尾でNEXTがまだ動かない発火・おじゃまも、1周期として保持する。
    tail = [e for e in formulas+ojama if e['t'] > lower_t]
    if tail:
        first = min(tail, key=lambda e: e['t'])
        outputs.append(dict(first, lower=lower, upper=first['i']+LATE_WRITE_FRAMES,
            cycle_start=lower_t, cycle_end=None, cycle_number=len(nexts), cycle_left_censored=False,
            signals=tail, write_indices=[n for n, w in enumerate(group['writes'])
                if lower <= w['i'] <= first['i']+LATE_WRITE_FRAMES and w['t'] > lower_t]))
    return outputs


def main() -> None:
    """初期候補を保存した上で、画像合図版へ置き換える。"""
    os.nice(19)
    for source in SOURCES:
        path = OUT/f'{source}_candidates.json'
        original = OUT/f'{source}_record_signal_candidates.json'
        if not original.exists():
            original.write_bytes(path.read_bytes())
        groups = json.loads(original.read_text())
        name = 'zenchi' if source == 'review' else source
        scan = json.loads((OUT/f'{name}_image_signals.json').read_text())
        rows, (obs, fps) = load(source), observations(source)
        for group in groups:
            events = cycles(group, rows, scan, obs, fps)
            group['events'] = [e for e in events if e['t'] >= (START if source in ('zenchi','review') else 0)]
        path.write_text(json.dumps(groups, separators=(',', ':')))
        print(source, [len(g['events']) for g in groups], flush=True)


if __name__ == '__main__':
    main()
