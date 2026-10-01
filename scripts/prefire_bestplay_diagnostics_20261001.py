"""Phase 3 (最善手) の門の前の診断。門の採点値 (q LL・先読み件数) はここでは計算しない (事前登録の前に見ない)。

- 発火前3秒の区間で、確実な勝ち (E35 証明つき) の「今撃つ」が何回あったか、決定的な「今撃つ」(|値−現在値| ≥ .2) が何回あったか
- 向きの一致: 最善手の値の向き (p_shown − p_current) と撃ち合い後の表示の向き (目標値 − p_current) が揃うか。
  Phase 2 の診断と同じく、実際に撃った側の枝の向きも別に数える (分母は撃ち合い全体の変化 ≥5%pt)
- 1通知 (盤面・NEXT が変わって計算した回) の所要 P50/P95/最大、表示の書き換え量 |p_shown − p_current|
使い方: PYTHONPATH=. python -m scripts.prefire_bestplay_diagnostics_20261001 --variant bestplay_l0
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scripts import prefire_oracle_ceiling_20260930 as oracle
from scripts.run_prefire_replay_20260930 import BASELINE_DIRS, OUT as REPLAY

LEAD_SEC = 3.0
DECISIVE = .2          # 「決定的」とみなす値の差 (勝率 20%pt)。診断の区切りで、合否には使わない
MIN_TOTAL = oracle.MIN_TOTAL_MOVE
SIDE_INDEX = {'1P': 0, '2P': 1}


def trace_table(root: Path) -> dict[str, np.ndarray]:
    """prefire_trace.npz を列名つきの表にする (採点器と同じ読み方)。"""
    from scripts.prefire_bestplay_gate_20261001 import trace
    return trace(root)


def window(table: dict, row: dict) -> np.ndarray:
    """撃ち合い1件の発火前 LEAD_SEC 秒 (直前の撃ち合いの終了後) に入る trace 行。"""
    t, games = table['t_sec'], table['game_idx']
    start = max(row['trigger'] - LEAD_SEC, row['prev_close'])
    return np.flatnonzero((games == row['game']) & (t >= start) & (t < row['trigger']))


def side_fire_now(table: dict, idx: np.ndarray, side: int, what: str) -> bool:
    """区間内に、その側の手に持つ組で撃つ最善手 (hand=1) があり、条件 (lethal / decisive) を満たすか。"""
    label = ('1p', '2p')[side]
    now = table['hand_' + label][idx] == 1
    if what == 'lethal':
        return bool(np.any(now & (table['lethal_' + label][idx] > 0)))
    gain = np.abs(table['v_' + label][idx] - table['p_current'][idx])
    return bool(np.any(now & (gain >= DECISIVE)))


def exchange_items(source: str, table: dict) -> list[dict]:
    """撃ち合いごとの診断 (母数つき)。目標値・発火側は OFF 記録で固定する。"""
    display, events = oracle.load(source)
    firer = {e['exchange_id']: SIDE_INDEX[min(e['chains'], key=lambda c: c['trigger_sec'])['side']] for e in events}
    items = []
    for row in oracle.exchange_rows(display, events):
        idx = window(table, row)
        item = dict(source=source, exchange_id=row['exchange_id'], rows=int(len(idx)), firer=firer[row['exchange_id']])
        if len(idx):
            last = idx[-1]
            current = float(table['p_current'][last])
            item.update(total=row['target'] - current, shift=float(table['p_shown'][last]) - current,
                        firer_value=float(table[('v_1p', 'v_2p')[item['firer']]][last]) - current,
                        **{f'{what}_{who}': side_fire_now(table, idx, s, what)
                           for what in ('lethal', 'decisive') for who, s in (('firer', item['firer']),
                                                                              ('other', 1 - item['firer']))})
        items.append(item)
    return items


def agreement(items: list[dict], key: str) -> dict:
    """向きの一致 (分母 = 変化 ≥5%pt かつ値が動いた撃ち合い)。"""
    pool = [i for i in items if i['rows'] and abs(i['total']) >= MIN_TOTAL and np.isfinite(i[key]) and i[key] != 0]
    hits = sum(np.sign(i[key]) == np.sign(i['total']) for i in pool)
    values = np.array([[i[key], i['total']] for i in pool]) if pool else np.zeros((0, 2))
    corr = float(np.corrcoef(values.T)[0, 1]) if len(pool) > 2 else None
    return dict(hits=int(hits), denominator=len(pool), corr=corr)


def timing(tables: list[dict]) -> dict:
    """計算した回 (compute_ms > 0) の所要と、書き換え量の分布。"""
    from scripts.prefire_bestplay_gate_20261001 import compute_times
    spent = np.concatenate([compute_times(t) for t in tables])
    moved = np.concatenate([np.abs(t['p_shown'] - t['p_current']) for t in tables])
    pct = lambda v, q: float(np.percentile(v, q)) if len(v) else None
    return dict(computed=int(len(spent)), p50_ms=pct(spent, 50), p95_ms=pct(spent, 95), p99_ms=pct(spent, 99),
                max_ms=pct(spent, 100), shown_rows=int(len(moved)), moved_rows=int((moved > 0).sum()),
                moved_p50=pct(moved, 50), moved_p99=pct(moved, 99), moved_max=pct(moved, 100))


def counts(items: list[dict]) -> dict:
    """確実な勝ち・決定的な「今撃つ」があった撃ち合いの数 (撃った側 / 撃たなかった側)。"""
    seen = [i for i in items if i['rows']]
    out = dict(exchanges=len(items), with_window=len(seen))
    for key in ('lethal_firer', 'lethal_other', 'decisive_firer', 'decisive_other'):
        out[key] = sum(bool(i[key]) for i in seen)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant', default='bestplay_l0')
    variant = parser.parse_args().variant
    tables = {s: trace_table(REPLAY / variant / s) for s in BASELINE_DIRS}
    items = [i for s in BASELINE_DIRS for i in exchange_items(s, tables[s])]
    chosen = np.concatenate([t['chosen'] for t in tables.values()])
    result = dict(variant=variant, counts=counts(items), agreement_shift=agreement(items, 'shift'),
                  agreement_firer=agreement(items, 'firer_value'), timing=timing(list(tables.values())),
                  chosen={str(k): int((chosen == k).sum()) for k in (-1, 0, 1, 2)})
    out = Path('logs/prefire_prediction') / f'DIAG_{variant}.json'
    out.write_text(json.dumps(dict(result, items=items), ensure_ascii=False, indent=1, default=float))
    print(json.dumps(result, ensure_ascii=False, indent=1, default=float), flush=True)


if __name__ == '__main__':
    main()
