"""記録されたSTABLE起点と、そのままのシミュレーションを診断する。"""
from __future__ import annotations

from itertools import combinations_with_replacement
from pathlib import Path
from collections import Counter
from itertools import accumulate
import json

import numpy as np
import cv2

from src.exchange_event_record import read_records
from src.exchange_event_landing import ExchangeLandingProjection
from src.indicators_v2 import _enumerate_placements
from src.scoring import calculate_chain_score
from scripts.run_e3_exchange_eval_20260926 import save_json
from scripts.inspect_e24_saved_20260928 import save_rows

OUT = Path('logs/e25')
BEGIN, TRIGGER, END = 2738., 2751.383333333333, 2752.
COLORS = (1, 2, 3, 4, 5)
OBSERVATION_END = 2754.2
FRAME_TIMES = (2750.5, 2751.35, 2751.383333333333, 2752.6833333333334, 2754.116666666667)
# 原映像2751.350秒の目視転記。診断だけに使用し、実行時入力へは書き戻さない。
FRAME_INSPECTION_CELLS = ((1, 2, 4), (2, 2, 4), (1, 3, 1), (1, 4, 0), (1, 5, 3), (2, 5, 1))


def candidates(board: object, simulator: object) -> list[dict]:
    """未観測の設置を仮定した場合の曖昧さも記録し、都合の良い候補だけを採らない。"""
    rows, seen = [], set()
    for pair in combinations_with_replacement(COLORS, 2):
        for placement, placed in _enumerate_placements(board, pair, simulator):
            result = simulator.simulate(placed)
            if not result.chain_count:
                continue
            score = calculate_chain_score(result)
            key = (score.total_score, result.final_board._grid.tobytes())
            if key in seen:
                continue
            seen.add(key)
            rows.append(dict(pair=list(pair), placement=placement, count=result.chain_count,
                score=score.total_score, board=result.final_board._grid.tolist(),
                prefix=list(accumulate(s.score for s in score.steps)), start=placed._grid.tolist()))
    return rows


def main() -> None:
    """旧記録を読み、受け側の選択起点・UNKNOWN・0段を直接区別する。"""
    simulator = ExchangeLandingProjection().simulator
    rows, previous, selected, observed, event_origin = [], None, None, {}, None
    for item in read_records(Path('logs/e16/records/review.jsonl.gz')):
        if item['kind'] != 'update':
            continue
        result, _, _, stamp, *_ = item['args']
        side = result.p2
        event = side.chain_event
        if TRIGGER <= stamp <= OBSERVATION_END and event and event.mechanism == 'formula_read':
            observed.setdefault(event.chain_count, dict(t_sec=stamp, count=event.chain_count,
                score=event.total_score, before_board_recorded=getattr(event, 'before_board', None) is not None))
            if event_origin is None and abs(stamp-TRIGGER) < 1e-8:
                event_origin = getattr(event, 'before_board', None)
        if not BEGIN <= stamp <= END or side.state.name != 'STABLE' or side.confirmed_board is None:
            continue
        if stamp < TRIGGER:
            selected = (stamp, side.confirmed_board.copy())
        board = side.confirmed_board
        if board._grid.tobytes() == previous:
            continue
        previous = board._grid.tobytes()
        simulated = simulator.simulate(board)
        rows.append(dict(t_sec=stamp, unknown=int(np.count_nonzero(board._grid == 10)),
            count=simulated.chain_count, score=calculate_chain_score(simulated).total_score,
            board=board._grid.tolist(), final=simulated.final_board._grid.tolist()))
    save_rows(OUT/'COMPLETION_HISTORY.json', rows)
    stamp, fallback = selected
    board = event_origin if event_origin is not None else fallback
    replies = candidates(board, simulator)
    save_rows(OUT/'PLACEMENT_CANDIDATES.json', replies)
    result = dict(trigger=TRIGGER, selected_sec=stamp, board=board._grid.tolist(),
        unknown=int(np.count_nonzero(board._grid == 10)), direct_count=simulator.simulate(board).chain_count,
        candidates=len(replies), counts=dict(Counter(r['count'] for r in replies)),
        observations=list(observed.values()), reason='zero_chain_on_origin_board',
        origin_source='event.before_board' if event_origin is not None else 'fallback_stable',
        differs_from_last_stable=int(np.count_nonzero(board._grid != fallback._grid)))
    save_json(OUT/'COMPLETION_CAUSE.json', result)
    frame_probe(board, simulator)
    print({k: v for k, v in result.items() if k != 'board'}, flush=True)
    print([(r['t_sec'], r['count'], r['score'], r['unknown']) for r in rows], flush=True)
    frames()


def frame_probe(board: object, simulator: object) -> None:
    """原映像で見える未反映セルを転記し、実測13連鎖を再現できるかだけを検査する。"""
    corrected = board.copy()
    for row, col, color in FRAME_INSPECTION_CELLS:
        corrected._grid[row, col] = color
    result = simulator.simulate(corrected)
    score = calculate_chain_score(result)
    save_json(OUT/'ORIGIN_FRAME_PROBE.json', dict(diagnostic_only=True,
        cells=FRAME_INSPECTION_CELLS, count=result.chain_count, score=score.total_score,
        prefix=list(accumulate(s.score for s in score.steps)), origin=board._grid.tolist(),
        inspected=corrected._grid.tolist(), final=result.final_board._grid.tolist()))
    print('原映像転記の診断:', result.chain_count, score.total_score, flush=True)


def frames() -> None:
    """E22と同じ原動画から発火前後を抜き、要求時刻と実フレームを記録する。"""
    command = json.loads(Path('logs/review_zenchi_g41_43_e22/status.json').read_text())['command']
    source = command[command.index('--video')+1]
    capture, rows = cv2.VideoCapture(source), []
    for stamp in FRAME_TIMES:
        capture.set(cv2.CAP_PROP_POS_MSEC, stamp*1000)
        ok, frame = capture.read()
        assert ok
        dest = OUT/f'zenchi_{stamp:.3f}.jpg'
        assert cv2.imwrite(str(dest), frame)
        rows.append(dict(requested_sec=stamp, actual_sec=capture.get(cv2.CAP_PROP_POS_MSEC)/1000,
                         frame=int(capture.get(cv2.CAP_PROP_POS_FRAMES))-1, path=str(dest)))
    capture.release()
    save_json(OUT/'ORIGIN_FRAMES.json', dict(source=source, frames=rows))


if __name__ == '__main__':
    main()
