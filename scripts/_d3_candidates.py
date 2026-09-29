"""D3の合図候補とSTABLE書込みを、元記録だけから列挙する。"""
from __future__ import annotations

import gzip
import json
import os
from typing import Any
import numpy as np
from scripts._d3_inventory import OUT, SOURCES

FPS = 30
DEBOUNCE = 8
LOOKBACK = 3 * FPS
START = 2580.6
END = 3427.2
COLORS = (1, 2, 3, 4, 5)


def load(source: str) -> list[dict]:
    """元記録を読取り専用で展開する。"""
    with gzip.open(OUT/f'{source}_records.jsonl.gz', 'rt') as stream:
        return [json.loads(line) for line in stream]


def array(side: dict) -> np.ndarray | None:
    """欠測盤面をゼロ盤面へ変換しない。"""
    value = side['confirmed_board']
    return None if value is None else np.asarray(value, dtype=np.int8)


def changes(rows: list[dict], side: int) -> list[dict]:
    """全STABLE更新を保存し、組ぷよ追加の候補を区別する。"""
    result, previous = [], None
    for index, row in enumerate(rows):
        current = array(row['sides'][side])
        if current is not None and previous is not None and not np.array_equal(current, previous):
            if row['sides'][side]['state']['state'] == 'STABLE':
                diff = np.argwhere(current[1:] != previous[1:]) + [1, 0]
                additions = [(int(r), int(c)) for r, c in diff
                             if previous[r, c] == 0 and current[r, c] in COLORS]
                result.append(dict(i=index, t=row['t'], game=row['game'], side=side,
                    before=previous.tolist(), board=current.tolist(), diff=diff.tolist(),
                    additions=additions, pure_addition=len(diff) == len(additions)))
        previous = current
    return result


def signals(rows: list[dict], side: int) -> list[dict]:
    """3種の保存合図を列挙し、連鎖途中・連続検出を印付けする。"""
    result, last = [], {'NEXT': -LOOKBACK, 'formula': -LOOKBACK, 'ojama': -LOOKBACK}
    previous, in_formula = 'MENU', False
    for index, row in enumerate(rows):
        value = row['sides'][side]
        state = value['state']['state']
        tags = []
        if value['next_slide_motion']:
            tags.append('NEXT')
        if row['formula'][side] and not in_formula:
            tags.append('formula')
        if state == 'OJAMA_FALL' and previous != state:
            tags.append('ojama')
        for tag in tags:
            repeated = index-last[tag] <= DEBOUNCE
            during_chain = state in ('CHAIN', 'GRAVITY_SETTLE') if tag == 'NEXT' else (
                previous in ('CHAIN', 'GRAVITY_SETTLE') if tag == 'formula' else False)
            result.append(dict(i=index, t=row['t'], side=side, game=row['game'],
                type=tag, repeated=repeated, during_chain=during_chain, state=state))
            last[tag] = index
        previous, in_formula = state, bool(row['formula'][side])
    return result


def candidates(rows: list[dict], side: int, sig: list[dict], writes: list[dict]) -> list[dict]:
    """合図窓に属する書込みを列挙し、一意対応できない窓を残す。"""
    groups: list[dict] = []
    for signal in sig:
        if signal['repeated'] or signal['during_chain'] or signal['state'] == 'MENU':
            continue
        if groups and signal['i']-groups[-1]['i'] <= DEBOUNCE:
            groups[-1]['signals'].append(signal)
            continue
        groups.append(dict(**signal, signals=[signal]))
    for number, event in enumerate(groups):
        lower = max(0, groups[number-1]['i']+1 if number else 0, event['i']-LOOKBACK)
        upper = min(len(rows)-1, groups[number+1]['i']-1 if number+1 < len(groups) else len(rows)-1)
        event.update(lower=lower, upper=min(upper, event['i']+DEBOUNCE),
            write_indices=[n for n, w in enumerate(writes) if lower <= w['i'] <= min(upper, event['i']+DEBOUNCE)])
    return groups


def main() -> None:
    """観測点を先に固定し、任意の画像に都合のよい窓を選ばない。"""
    os.nice(19)
    for source in SOURCES:
        rows = load(source)
        outputs = []
        for side in range(2):
            sig, writes = signals(rows, side), changes(rows, side)
            events = candidates(rows, side, sig, writes)
            events = [e for e in events if (0 if source in SOURCES[:3] else START) <= e['t'] < (900 if source in SOURCES[:3] else END)]
            outputs.append(dict(signals=sig, writes=writes, events=events))
        (OUT/f'{source}_candidates.json').write_text(json.dumps(outputs, separators=(',', ':')))
        print(source, [(len(s['signals']), len(s['events']), len(s['writes'])) for s in outputs], flush=True)


if __name__ == '__main__':
    main()
