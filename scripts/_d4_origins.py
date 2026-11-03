"""連鎖開始通知と起点盤面のOFF/ON差を独立シミュレーションする。"""
from __future__ import annotations
from collections import Counter
import json
from pathlib import Path
import numpy as np
from scripts._d4_loss import OUT, Q, save
from scripts.r1_measure_helpers import lines
from src.exchange_event_record import read_records
from src.chain import ChainSimulator
from src.scoring import calculate_chain_score
from src.production_config import GHOST_CHAIN_RULE_ENABLED
from src.board_state_machine import BoardState

TARGETS = (Q, 'review')


def effective(event: object, rows: list[tuple], game: int, side: int) -> dict | None:
    """通知に盤面がなければ本番と同じ直前STABLE盤面を選ぶ。"""
    if event is None:
        return None
    board = getattr(event, 'before_board', None)
    stamp = None
    if board is None:
        label = ('p1', 'p2')[side]
        saved = next((r for r in reversed(rows) if r[4]==game and r[3]<event.trigger_sec
            and getattr(r[0],label).state==BoardState.STABLE
            and getattr(r[0],label).confirmed_board is not None), None)
        if saved is not None:
            board, stamp = getattr(saved[0],label).confirmed_board, saved[3]
    value = summary(board)
    return dict(value, selected_sec=stamp) if value else None


def summary(board: object) -> dict | None:
    """発火起点の火力をモデルと無関係な物理計算で照合する。"""
    if board is None:
        return None
    simulator = ChainSimulator(exclude_hidden_row_from_pop=GHOST_CHAIN_RULE_ENABLED)
    result = simulator.simulate(board)
    return dict(board=board._grid.tolist(), chains=result.chain_count,
        score=calculate_chain_score(result).total_score)


def run(source: str) -> list[dict]:
    """連鎖IDは再生側で変わり得るため時刻・側・段・式で対応する。"""
    updates = {}
    for mode in ('off', 'on'):
        root = Path('logs/e31' if mode=='off' else 'logs/r1')
        updates[mode] = [r['args'] for r in read_records(root/'records'/f'{source}.jsonl.gz') if r['kind']=='update']
    result, seen = [], set()
    corrections = [r for r in lines(Path('logs/r1/capture')/source/'placement_signal_reconcile.jsonl') if r['corrections']]
    for old, new in zip(updates['off'], updates['on']):
        assert old[3:5] == new[3:5]
        for side, label in enumerate(('p1','p2')):
            a, b = getattr(old[0],label).chain_event, getattr(new[0],label).chain_event
            if a is None and b is None:
                continue
            key = (old[4], side, a.trigger_sec if a else None, b.trigger_sec if b else None)
            if key in seen:
                continue
            seen.add(key)
            sa = effective(a, updates['off'], old[4], side)
            sb = effective(b, updates['on'], new[4], side)
            recent = [r for r in corrections if r['side']==side and r['t_sec']<=new[3]]
            last = recent[-1] if recent else None
            result.append(dict(source=source, t=old[3], game=old[4], side=side,
                same_event=a is not None and b is not None and (a.trigger_sec,a.mechanism,a.chain_count,a.total_score)==(b.trigger_sec,b.mechanism,b.chain_count,b.total_score),
                off=sa, on=sb, previous_correction=last))
    save(f'{source}_origins.json', result)
    print(source, dict(total=len(result), event_different=sum(not r['same_event'] for r in result),
        origin_different=sum(r['off'] != r['on'] for r in result)), flush=True)
    return result


def main() -> None:
    """動画や認識器を開かずに保存された起点を検査する。"""
    for source in TARGETS:
        run(source)


if __name__ == '__main__':
    main()
