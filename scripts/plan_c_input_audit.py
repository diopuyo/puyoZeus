"""案C: 元の148動画・G_fe行でPhase 4の割当入力が保存されているか監査する。"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.prefire_hazard_samples_20260930 import LEAN, queue_of
from scripts.train_exchange_event_models_20260926 import REFERENCE, run_path
from src.prefire_stable_queue import SideQueue

OUT = Path('logs/prefire_prediction/plan_c')
SIDES = ('1P', '2P')
MIN_AVAILABLE_KB = 4 * 1024 * 1024


def memory_available() -> int:
    """優先ジョブのため4GBの余裕を確認する。"""
    line = next(line for line in Path('/proc/meminfo').read_text().splitlines()
                if line.startswith('MemAvailable:'))
    available = int(line.split()[1])
    if available < MIN_AVAILABLE_KB:
        raise RuntimeError(f'空きメモリ4GB未満: {available} KiB')
    return available


def assigned(data: dict[str, np.ndarray]) -> np.ndarray:
    """保存された実観測だけをPhase 4へ順送りする。フレームの水増しは禁止。"""
    result = np.zeros((len(data['t_sec']), 6), dtype=np.int8)
    for side in SIDES:
        tracker, previous_game = SideQueue(), None
        indices = np.flatnonzero(data['side'] == side)
        for index in indices:
            game = int(data['game_idx'][index])
            if game != previous_game:
                tracker, previous_game = SideQueue(), game
            tracker.observe(float(data['t_sec'][index]), data['grids'][index].astype(np.int8).tobytes(),
                            np.asarray(queue_of(data, index)))
            result[index] = tracker.known()
    return result


def inspect_video(path: Path, training: pd.DataFrame) -> dict:
    """元の学習行へ同じ側・frame・試合の原票だけを対応させる。"""
    memory_available()
    with np.load(path) as saved:
        data = {key: saved[key] for key in saved.files}
    known = assigned(data)
    key_to_index = {(int(g), str(s), int(f)): i for i, (g, s, f) in
                    enumerate(zip(data['game_idx'], data['side'], data['frame_idx']))}
    indices = np.array([key_to_index.get((int(row.game_id.rsplit(':', 1)[1]), row.source_side,
                                         int(row.frame)), -1) for row in training.itertuples()], dtype=np.int64)
    valid = indices >= 0
    flags = np.zeros(len(training), dtype=bool)
    flags[valid] = (known[indices[valid]] > 0).all(axis=1)
    duplicate_boards, consecutive_frame_pairs = 0, 0
    for side in SIDES:
        ids = np.flatnonzero(data['side'] == side)
        same_game = np.diff(data['game_idx'][ids]) == 0
        same_board = (data['grids'][ids][1:] == data['grids'][ids][:-1]).all(axis=(1, 2))
        duplicate_boards += int((same_game & same_board).sum())
        consecutive_frame_pairs += int((same_game & (np.diff(data['frame_idx'][ids]) == 1)).sum())
    cut = len(known)//2
    prefix = assigned({key: value[:cut] for key, value in data.items()})
    np.testing.assert_array_equal(prefix, known[:cut])
    return dict(video='video_'+path.stem, raw_rows=len(known), training_rows=len(training),
                matched_rows=int(valid.sum()), complete_three_pairs=int(flags.sum()),
                complete_three_pairs_raw=int((known > 0).all(axis=1).sum()),
                consecutive_frame_pairs=consecutive_frame_pairs, same_board_pairs=duplicate_boards,
                prefix_rows=cut, prefix_mismatches=0,
                input_sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def main() -> None:
    """全148本を1プロセスで走査し、母数と欠測を保存する。"""
    OUT.mkdir(parents=True, exist_ok=True)
    rows_path = run_path(REFERENCE, 'base')/'rows.csv'
    rows = pd.read_csv(rows_path)
    parts = []
    for path in sorted(LEAN.glob('*.npz')):
        part = inspect_video(path, rows[rows.video_id == 'video_'+path.stem])
        parts.append(part)
        print(json.dumps(part), flush=True)
    totals = Counter()
    for part in parts:
        totals.update({key: value for key, value in part.items() if isinstance(value, int)})
    result = dict(raw_videos=len(parts), training_videos=int(rows.video_id.nunique()),
                  rows_sha256=hashlib.sha256(rows_path.read_bytes()).hexdigest(),
                  totals=dict(totals), videos=parts,
                  status='input_history_audit_not_quality_pass')
    (OUT/'INPUT_AUDIT.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(dict(totals)), flush=True)


if __name__ == '__main__':
    main()
