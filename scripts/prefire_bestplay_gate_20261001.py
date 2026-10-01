"""Phase 3 (最善手) を事前登録の門 (exev DECISIONS.md 2026-10-01) で採点する。

比較基準は Phase 2 と同じ現本番記録 `exev/logs/pending_expiry/e36b_on/on` (同じ5記録・同じ行)。
発火前区間・撃ち合いの目標値は OFF の表示とイベントで固定する (ON の結果で母数を変えない)。
使い方: PYTHONPATH=. python -m scripts.prefire_bestplay_gate_20261001 --variant bestplay_l0
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scripts import aggregate_e3_exchange_eval_20260926 as e3
from scripts import prefire_gate_report_20260930 as phase2
from scripts.run_prefire_replay_20260930 import BASELINE_DIRS, OUT as REPLAY

Q_SOURCE, ZENCHI, REVIEW = phase2.Q_SOURCE, phase2.ZENCHI, phase2.REVIEW
Q_MAX = .507567 + 5e-7                 # 悪化なし (既存の門と同じ許容)
PREFIRE_LL_MAX = .7308 - .035          # 改善 (Phase 2 と同じ)
ANTICIPATION_MIN = 23 + 10             # 改善 (Phase 2 と同じ)
ZENCHI_HITS_MIN, SCENE_DEADLINE, FLIP_RATIO_MAX = 7671, 2766.0, 1.2
GAME14, GAME14_SIDE = 14, 0            # q 第14試合 (勝者1P) で 1P の負けを表示で断定しないこと
LETHAL_COLUMNS = ('lethal_1p', 'lethal_2p')
FPS = 30


def trace(root: Path) -> dict[str, np.ndarray]:
    """prefire_trace.npz を列名つきで読む。Phase 4 は計算した回の所要を 'computes' に別に持つ。"""
    with np.load(root / 'prefire_trace.npz') as data:
        table = dict(zip([str(c) for c in data['columns']], data['values'].T))
        if 'computes' in data.files:
            table['_computes_ms'] = data['computes'][:, 1] if len(data['computes']) else np.zeros(0)
        return table


def compute_times(table: dict) -> np.ndarray:
    """計算した回の所要 (ms)。Phase 3 は trace の compute_ms > 0 の行。"""
    if '_computes_ms' in table:
        return table['_computes_ms']
    return table['compute_ms'][table['compute_ms'] > 0]


def shown_lethal(table: dict, side: int) -> np.ndarray:
    """表示に出た確実な勝ちの行 (Phase 4 は保持の後の表示、Phase 3 は採った側)。"""
    if 'shown_lethal' in table:
        return table['shown_lethal'].astype(int) == side + 1
    chosen = table['chosen'].astype(int)
    return (table[LETHAL_COLUMNS[side]] > 0) & ((chosen == side) | (chosen == 2))


def false_lethal(source: str, table: dict) -> dict:
    """表示で採った確実な勝ち (E35 証明) が、ラベルの勝者と食い違った行数 (母数つき)。"""
    windows = e3.outcomes(source)[0]
    frames = np.rint(table['t_sec'] * FPS).astype(int)
    rows, wrong = 0, 0
    for side in range(len(LETHAL_COLUMNS)):
        used = shown_lethal(table, side)
        for window in windows:
            sel = used & (frames >= window['start']) & (frames < window['end'])
            rows += int(sel.sum())
            wrong += int(sel.sum()) if (window['winner'] == '1P') != (side == 0) else 0
    return dict(rows=rows, wrong=wrong)


def game14_display(table: dict) -> int:
    """q 第14試合で、最善手の層が 1P の確実な負け (2P の lethal) を表示した行数。"""
    return int(((table['game_idx'] == GAME14) & shown_lethal(table, 1)).sum())


def timing(on: Path) -> dict:
    """計算した回 (compute_ms > 0) の所要 (ms)。"""
    spent = np.concatenate([compute_times(trace(on / s)) for s in BASELINE_DIRS])
    pct = lambda q: float(np.percentile(spent, q)) if len(spent) else None
    return dict(computed=int(len(spent)), p50=pct(50), p95=pct(95), p99=pct(99), max=pct(100))


def score(on: Path) -> dict:
    """門の各項目を母数つきで出す。"""
    phase2.ON = on
    displays = {'off': {s: phase2.load(phase2.dirs(s)[0], s) for s in BASELINE_DIRS},
                'on': {s: phase2.load(on / s, s) for s in BASELINE_DIRS}}
    rows = {s: phase2.fixed_rows(s)[1] for s in BASELINE_DIRS}
    p1 = lambda which: {s: displays[which][s]['display_p1'] for s in BASELINE_DIRS}
    result = dict(off=phase2.variant_scores(dict(displays, on=displays['off']), rows, p1('off')),
                  on=phase2.variant_scores(displays, rows, p1('on')),
                  placebo=phase2.variant_scores(displays, rows, {s: phase2.placebo_p1(
                      displays['on'][s], on / s / 'prefire_trace.npz') for s in BASELINE_DIRS}))
    result['gate1'] = phase2.non_worse(displays)
    tables = {s: trace(on / s) for s in BASELINE_DIRS}
    result['false_lethal'] = {s: false_lethal(s, tables[s]) for s in (Q_SOURCE, 'fcXG83vInDY', 'mia8KCjr52g')}
    result['game14_display_rows'] = game14_display(tables[Q_SOURCE])
    result['timing_ms'] = timing(on)
    return result


def gates(result: dict, phase4: bool = False) -> dict:
    """事前登録の判定。改善は発火前3秒 LL と先読み件数の両方、対照はどちらかを落とすこと。"""
    g1, on, placebo = result['gate1'], result['on'], result['placebo']
    flips = g1['flips']
    out = dict(q=g1['q']['frames'] == 6526 and g1['q']['log_loss'] <= Q_MAX,
               zenchi=g1['zenchi']['frames'] == 8333 and g1['zenchi']['hits'] >= ZENCHI_HITS_MIN,
               false_certainty=all(g1['events_identical'].values())
               and all(v['wrong'] == 0 for v in result['false_lethal'].values()),
               scene=g1['scene_first_sec'] is not None and g1['scene_first_sec'] <= SCENE_DEADLINE,
               game14=all(g1['events_identical'].values()) and result['game14_display_rows'] == 0,
               flips=flips['on'] <= FLIP_RATIO_MAX * flips['off'],
               prefire3s=on['q']['prefire3s']['log_loss'] <= PREFIRE_LL_MAX,
               anticipation=on['anticipation']['hits'] >= ANTICIPATION_MIN)
    out['placebo_fails'] = not (placebo['q']['prefire3s']['log_loss'] <= PREFIRE_LL_MAX
                                and placebo['anticipation']['hits'] >= ANTICIPATION_MIN)
    if phase4:   # Phase 4 事前登録: 先読みは揺れで偶然増えるので、対照が先読み条件を通らないことも必須
        out['placebo_anticipation_fails'] = placebo['anticipation']['hits'] < ANTICIPATION_MIN
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant', required=True)
    parser.add_argument('--phase4', action='store_true', help='Phase 4 の門 (対照の先読み不合格も必須)')
    args = parser.parse_args()
    variant = args.variant
    result = score(REPLAY / variant)
    result['gates'] = gates(result, args.phase4)
    result['leak_suspect'] = result['on']['q']['all']['log_loss'] < phase2.ORACLE_Q_LL
    result['passed'] = all(result['gates'].values())
    out = Path('logs/prefire_prediction') / f'GATE_{variant}.json'
    out.write_text(json.dumps(result, ensure_ascii=False, indent=1, default=float), encoding='utf-8')
    print(json.dumps({k: result[k] for k in ('gates', 'passed', 'leak_suspect', 'timing_ms')}, default=float))


if __name__ == '__main__':
    main()
