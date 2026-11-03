"""有界化 (事前登録の B1/B2) が評価出力をどれだけ動かしたかを、基準との差として母数つきで数える。

events.jsonl の全評価値 (各 value 1 行) の p1 / source / dead_sides と、display.npz の表示確率・有利不利を比べる。
使い方: python -m scripts.compare_bounded_effect <基準の出力dir> <比較の出力dir>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

P1_TOLERANCE = 0.0          # 0 = 完全一致以外は差として数える
LARGE_P1_DIFF = 0.01        # 「大きい差」の目安 (勝率で 1 ポイント)
DECISION_KEYS = ('source', 'dead_sides')


def events(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def event_effect(base: Path, other: Path) -> dict:
    """評価値ごとの差。両者の値の個数が違えば構造差として報告する。"""
    a, b = events(base / 'events.jsonl'), events(other / 'events.jsonl')
    total = changed = large = decision = 0
    max_diff = 0.0
    if len(a) != len(b):
        return dict(structure_mismatch=True, base_events=len(a), other_events=len(b))
    for x, y in zip(a, b):
        if len(x['values']) != len(y['values']):
            return dict(structure_mismatch=True, at=x['exchange_id'])
        for vx, vy in zip(x['values'], y['values']):
            total += 1
            diff = abs(vx['p1'] - vy['p1'])
            changed += diff > P1_TOLERANCE
            large += diff > LARGE_P1_DIFF
            decision += any(vx.get(k) != vy.get(k) for k in DECISION_KEYS)
            max_diff = max(max_diff, diff)
    return dict(values=total, p1_changed=int(changed), p1_over_1pt=int(large), max_p1_diff=max_diff,
                decision_changed=int(decision), structure_mismatch=False)


def display_effect(base: Path, other: Path) -> dict:
    """表示列 (display_p1 / display_adv) の差。"""
    a, b = np.load(base / 'display.npz'), np.load(other / 'display.npz')
    out = dict(frames=int(len(a['t_sec'])))
    for key in ('display_p1', 'display_adv'):
        diff = np.abs(a[key] - b[key])
        out[key] = dict(changed=int((diff > 0).sum()), max=float(np.nanmax(diff)))
    return out


def main() -> None:
    base, other = Path(sys.argv[1]), Path(sys.argv[2])
    print(json.dumps(dict(events=event_effect(base, other), display=display_effect(base, other)),
                     ensure_ascii=False))


if __name__ == '__main__':
    main()
