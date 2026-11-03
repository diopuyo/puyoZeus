"""会計を真値にせず、降下前後のSTABLE盤面の差を独立に読む。"""
from __future__ import annotations

from typing import Any
import numpy as np
import pandas as pd

from src.exchange_event_record import read_records
from scripts._audit_firepower_1003_inventory import OUT, ROOT, resources
from scripts._audit_firepower_1003_extract import games_for, label, landing_frames
from scripts._audit_firepower_1003_replay import NAMES

OJAMA = 9
UNKNOWN = 10
MAX_TSUMO_COLORS = 2
MAX_DROP = 30


def grid_counts(board: Any) -> tuple[int, list[int], bool]:
    """隠し段を含む色別個数を数え、UNKNOWNがあれば数量監査から外す。"""
    grid = board._grid
    return int((grid == OJAMA).sum()), [int((grid == c).sum()) for c in range(1, 6)], bool((grid == UNKNOWN).any())


def observe(side: Any, sec: float, memory: dict, output: list[dict], idx: int) -> None:
    """自側の連鎖を跨いだ前後差をおじゃま実測と誤認しない。"""
    state = side.state.name
    if state in ('CHAIN', 'GRAVITY_SETTLE'):
        memory['interrupted'] = True
    if state == 'OJAMA_FALL' and memory.get('previous') != state and 'pending' not in memory:
        memory['pending'] = dict(fall_start=sec, before=memory.get('stable'),
                                  interrupted=memory.get('interrupted', True))
    if state == 'STABLE' and side.confirmed_board is not None:
        counts = grid_counts(side.confirmed_board)
        if 'pending' in memory:
            pending = memory.pop('pending')
            before = pending['before']
            differences = np.array(counts[1])-np.array(before[1]) if before else np.array([-1])
            measurable = bool(before and not pending['interrupted'] and not before[2] and not counts[2]
                              and (differences >= 0).all() and differences.sum() <= MAX_TSUMO_COLORS)
            output.append(dict(side=f'{idx+1}P', fall_start=pending['fall_start'], landed=sec,
                before_ojama=before[0] if before else np.nan, after_ojama=counts[0],
                board_delta=counts[0]-before[0] if before else np.nan, measurable=measurable,
                chain_between=pending['interrupted'], unknown=counts[2] or (before[2] if before else True)))
        memory['stable'], memory['interrupted'] = counts, False
    memory['previous'] = state


def extract(name: str) -> pd.DataFrame:
    """全降下の観測を残し、解釈可能な前後差だけを別標本にする。"""
    memory, output, last_game = [{}, {}], [], None
    for row in read_records(ROOT / 'collect/records' / f'{name}.jsonl.gz'):
        if row['kind'] != 'update':
            continue
        result, _, _, sec, game = row['args'][:5]
        if game != last_game:
            memory, last_game = [{}, {}], game
        for idx, side in enumerate((result.p1, result.p2)):
            observe(side, sec, memory[idx], output, idx)
    data = pd.DataFrame(output)
    folder, games = OUT / 'replay' / name, games_for(name)
    with np.load(folder / 'display.npz') as display:
        low, high = display['t_sec'][[0, -1]]
    data = data[data.fall_start.between(low, high)].copy()
    data = data[[label(t, games)['match'] >= 0 for t in data.fall_start]].reset_index(drop=True)
    values = landing_frames(folder).rename(columns={'t_sec': 'prediction_sec'})
    data = pd.merge_asof(data.sort_values('fall_start'), values, left_on='fall_start',
                         right_on='prediction_sec', direction='backward', allow_exact_matches=False)
    data['predicted_pending'] = np.where(data.side == '1P', data.incoming1, data.incoming2)
    data['predicted_single_drop'] = data.predicted_pending.clip(upper=MAX_DROP)
    data['prediction_age'] = data.fall_start-data.prediction_sec
    data['segment'] = name
    return data


def main() -> None:
    """判定の独立性に制限があるため、盤面差を映像真値とは呼ばない。"""
    for name in NAMES:
        resources()
        folder = OUT / 'replay' / name
        path = folder / 'board_drops.csv'
        if (folder / 'status.json').exists() and not path.exists():
            result = extract(name)
            result.to_csv(path, index=False)
            print(name, len(result), int(result.measurable.sum()), flush=True)
    files = [OUT / 'replay' / n / 'board_drops.csv' for n in NAMES]
    all_rows = pd.concat([pd.read_csv(p) for p in files if p.exists()], ignore_index=True)
    all_rows.to_csv(OUT / 'board_drops.csv', index=False)


if __name__ == '__main__':
    main()
