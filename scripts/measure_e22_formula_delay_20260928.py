"""既存3動画のbaseline通知から最初の有効な式読取までの遅延を測る。"""
from __future__ import annotations
import hashlib
import math
from pathlib import Path
import numpy as np
from src.exchange_event_record import read_records
from src.board_state_machine import BoardState
from scripts.run_e3_exchange_eval_20260926 import SOURCES, save_json

OUT = Path('logs/e22')
FPS, QUANTILE = 30, .99
ACTIVE_STATES = (BoardState.CHAIN, BoardState.GRAVITY_SETTLE)


def measure(source: str) -> list[dict]:
    """別試合・次の手・死亡後の式を当該baselineの確認へ混ぜない。"""
    rows, pending, seen, game = [], [None, None], set(), None
    formula_sec = [float('-inf'), float('-inf')]
    for item in read_records(Path('logs/e16/records')/f'{source}.jsonl.gz'):
        if item['kind'] != 'update':
            continue
        result, _, _, stamp, current, _, _, visible = item['args']
        if current != game:
            pending, game = [None, None], current
            formula_sec = [float('-inf'), float('-inf')]
        for idx, side in enumerate((result.p1, result.p2)):
            label, event = ('1P', '2P')[idx], side.chain_event
            if visible[idx]:
                formula_sec[idx] = stamp
            if pending[idx] is not None and (label in result.confirmed_dead_sides or
                    (side.state not in ACTIVE_STATES and event is None)):
                pending[idx]['censor_sec'] = stamp
                pending[idx] = None
            if event is not None and event.mechanism == 'baseline':
                key = current, idx, event.trigger_sec
                if key not in seen:
                    seen.add(key)
                    pending[idx] = dict(source=source, game=current, side=label,
                        trigger_sec=event.trigger_sec, observed_sec=stamp, formula_sec=None)
                    rows.append(pending[idx])
            if (pending[idx] is not None and formula_sec[idx] >= pending[idx]['trigger_sec']
                    and label not in result.confirmed_dead_sides):
                pending[idx].update(formula_sec=formula_sec[idx], delay_sec=max(0., formula_sec[idx]-pending[idx]['observed_sec']))
                pending[idx] = None
    return rows


def main() -> None:
    """可測例のP99を1表示フレーム単位で切り上げ、欠測例数も併記する。"""
    rows = [row for source in SOURCES for row in measure(source)]
    values = [r['delay_sec'] for r in rows if r['formula_sec'] is not None]
    assert values
    p99 = float(np.quantile(values, QUANTILE, method='higher'))
    frames = max(1, math.ceil(p99*FPS-1e-8))
    save_json(OUT/'FORMULA_DELAY.json', dict(sources=SOURCES, baseline_notifications=len(rows),
        measured=len(values), censored=len(rows)-len(values), p99_sec=p99,
        quantile_method='higher', max_hold_frames=frames, max_hold_sec=frames/FPS,
        definition='baseline初通知→同側の最初の有効式読取。試合境界・非連鎖復帰・死亡で打切り。欠測を0扱いしない。',
        sha256={s: hashlib.sha256((Path('logs/e16/records')/f'{s}.jsonl.gz').read_bytes()).hexdigest() for s in SOURCES}, rows=rows))
    print(dict(n=len(values), censored=len(rows)-len(values), p99=p99, frames=frames, hold=frames/FPS))


if __name__ == '__main__':
    main()
