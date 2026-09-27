"""B5の盤面対応を実時刻へ適用し、捨てフレームの認識影響を測る。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from scripts.analyze_live_b5_alignment import events, pair_events
from scripts.measure_live_b6 import save

FPS = 30
START_SEC, END_SEC = 2580., 2700.
MAX_SHIFT_SEC = 2.
PLACEMENT_CELLS = 2
REFERENCE_LIMIT_FRAMES = 8
PERCENTILES = (50, 95, 99)
VISIBLE_CELLS = 72
HIDDEN_CELLS = 6
MILLISECONDS = 1000.


def distribution(values: list[float]) -> dict[str, Any]:
    return dict(count=len(values), minimum=min(values, default=None), maximum=max(values, default=None),
        **{f'P{p}': float(np.percentile(values, p)) if values else None for p in PERCENTILES})


def load_audit(path: Path, start: float = START_SEC, end: float = END_SEC) -> dict[str, np.ndarray]:
    with np.load(path) as saved:
        mask = (saved['t_sec'] >= start) & (saved['t_sec'] < end)
        return {key: saved[key][mask] for key in saved.files}


def board_events(data: dict, side: int) -> list[dict]:
    """欠けた行数を時刻と誤認せず、B5へ30Hzの絶対フレーム番号を渡す。"""
    result, _ = events(data, side)
    for event in result:
        event['indices'] = list(event['frames'])
        event['active'] = bool(data['active'][event['indices'][0]])
        event['frames'] = [int(round(data['t_sec'][i]*FPS)) for i in event['indices']]
    return result


def placement_indices(sequence: list[dict]) -> list[int]:
    """消去・おじゃま増加と混ぜず、可視盤面への色ぷよ2個の純増だけを設置とする。"""
    output = []
    for i in range(1, len(sequence)):
        before, after = sequence[i-1], sequence[i]
        if before['epoch'] != after['epoch'] or not after['active']:
            continue
        old, new = before['board'][1:], after['board'][1:]
        added = (old == 0) & (new >= 1) & (new <= 5)
        if added.sum() == PLACEMENT_CELLS and np.array_equal(old[~added], new[~added]):
            output.append(i)
    return output


def classify_missing(placements: list[int], pairs: list, a_count: int, b_count: int) -> dict:
    """前後アンカー間で候補盤面が一つもなければ見落とし、残りは判定不能とする。"""
    matched = {i for i, _, _ in pairs}
    omitted, ambiguous = [], []
    for i in placements:
        if i in matched:
            continue
        before = next(((a, b) for a, b, _ in reversed(pairs) if a < i), None)
        after = next(((a, b) for a, b, _ in pairs if a > i), None)
        (omitted if before and after and after[1]-before[1] == 1 else ambiguous).append(i)
    return dict(reference_placements=len(placements), matched=len(set(placements) & matched),
        omitted=len(omitted), unresolved=len(ambiguous), omitted_indices=omitted,
        unresolved_indices=ambiguous, reference_board_events=a_count, candidate_board_events=b_count)


def raw_arrival(data: dict, side: int, sequence: list[dict], index: int) -> float | None:
    """全フレーム側の可視瞬間盤面が2フレーム連続一致した初回を設置時刻の代理とする。"""
    last = sequence[index]['indices'][0]
    first = max(sequence[index-1]['indices'][0],
                int(np.searchsorted(data['t_sec'], data['t_sec'][last]-MAX_SHIFT_SEC)))
    board = sequence[index]['board'][1:]
    matches = np.all(data['raw'][first:last+1, side, 1:] == board, axis=(1, 2))
    for offset in range(len(matches)-1):
        contiguous = round((data['t_sec'][first+offset+1]-data['t_sec'][first+offset])*FPS) == 1
        if matches[offset] and matches[offset+1] and contiguous:
            return float(data['t_sec'][first+offset])
    return None


def side_comparison(left: dict, right: dict, side: int) -> dict:
    a, b = board_events(left, side), board_events(right, side)
    pairs = pair_events(a, b)
    placements = placement_indices(a)
    missing = classify_missing(placements, pairs, len(a), len(b))
    details = []
    for i, j, kind in pairs:
        ai, bi = a[i]['indices'][0], b[j]['indices'][0]
        difference = a[i]['board'] != b[j]['board']
        arrival = raw_arrival(left, side, a, i) if i in placements else None
        delay = float(right['t_sec'][bi]-left['t_sec'][ai])
        wall = float(right['recognized_at'][bi]-right['captured_at'][bi])
        details.append(dict(side=side+1, left_event=i, right_event=j, kind=kind,
            reference_sec=float(left['t_sec'][ai]), candidate_sec=float(right['t_sec'][bi]),
            visible_diff=int(difference[1:].sum()), hidden_diff=int(difference[0].sum()),
            confirmation_shift_frames=int(round(delay*FPS)), placement=i in placements, raw_arrival_sec=arrival,
            reference_placement_frames=None if arrival is None else int(round((left['t_sec'][ai]-arrival)*FPS)),
            candidate_placement_frames=None if arrival is None else int(round((right['t_sec'][bi]-arrival)*FPS)),
            candidate_placement_wall_frames=None if arrival is None else (right['t_sec'][bi]-arrival+wall)*FPS))
    return dict(side=side+1, placements=missing, pairs=details)


def board_comparison(left: dict, right: dict) -> dict:
    sides = [side_comparison(left, right, side) for side in range(2)]
    pairs = [r for side in sides for r in side['pairs']]
    placements = [r for r in pairs if r['placement']]
    observed = [r for r in placements if r['raw_arrival_sec'] is not None]
    proxy = [r for r in observed if r['candidate_placement_frames'] >= 0]
    return dict(reference_notifications=len(left['t_sec']), candidate_notifications=len(right['t_sec']),
        matched_board_events=len(pairs), visible_cells=len(pairs)*VISIBLE_CELLS, hidden_cells=len(pairs)*HIDDEN_CELLS,
        visible_differences=sum(r['visible_diff'] for r in pairs),
        hidden_differences=sum(r['hidden_diff'] for r in pairs),
        placements={key: sum(s['placements'][key] for s in sides)
                    for key in ('reference_placements', 'matched', 'omitted', 'unresolved')},
        stable_confirmation_shift_frames=distribution([r['confirmation_shift_frames'] for r in pairs]),
        placement_additional_delay_frames=distribution([r['confirmation_shift_frames'] for r in placements]),
        placement_extra_over_8_frames=sum(r['confirmation_shift_frames'] > REFERENCE_LIMIT_FRAMES for r in placements),
        raw_proxy_placement_reference_frames=distribution([r['reference_placement_frames'] for r in proxy]),
        raw_proxy_placement_candidate_frames=distribution([r['candidate_placement_frames'] for r in proxy]),
        raw_proxy_placement_candidate_wall_frames=distribution([r['candidate_placement_wall_frames'] for r in proxy]),
        raw_proxy_candidate_over_8_frames=int(sum(r['candidate_placement_frames'] > REFERENCE_LIMIT_FRAMES for r in proxy)),
        raw_proxy_unavailable=len(placements)-len(observed),
        raw_proxy_rejected_early=len(observed)-len(proxy), sides=sides)


def match_times(left: list[dict], right: list[dict], field: str) -> list[tuple[int, int]]:
    """同側・同試合・2秒以内で、対応数最大→時刻差最小の順序保存対応を取る。"""
    score = [[(0, 0.) for _ in range(len(right)+1)] for _ in range(len(left)+1)]
    moves = {}
    for i, a in enumerate(left, 1):
        for j, b in enumerate(right, 1):
            choices = [(score[i-1][j], (i-1, j)), (score[i][j-1], (i, j-1))]
            delta = abs(a[field]-b[field])
            if a['side'] == b['side'] and a['game'] == b['game'] and delta <= MAX_SHIFT_SEC:
                count, cost = score[i-1][j-1]
                choices.append(((count+1, cost-delta), (i-1, j-1)))
            score[i][j], moves[i, j] = max(choices, key=lambda item: item[0])
    pairs, i, j = [], len(left), len(right)
    while i and j:
        before_i, before_j = moves[i, j]
        if before_i == i-1 and before_j == j-1:
            pairs.append((i-1, j-1))
        i, j = before_i, before_j
    return list(reversed(pairs))


def read_physics(path: Path) -> tuple[list, list]:
    chains, landings = [], []
    for line in path.read_text(encoding='utf-8').splitlines():
        record = json.loads(line)
        chains.extend(dict(chain, game=record['game_idx']) for chain in record['chains'])
        landings.extend(dict(item, game=record['game_idx']) for item in record['landings']
                        if item['t_sec'] is not None)
    return chains, landings


def timing_result(left: list, right: list, pairs: list, field: str) -> dict:
    valid = [(i, j) for i, j in pairs if left[i].get(field) is not None and right[j].get(field) is not None]
    shifts = [right[j][field]-left[i][field] for i, j in valid]
    return dict(reference_events=sum(r.get(field) is not None for r in left),
        candidate_events=sum(r.get(field) is not None for r in right), matched=len(valid),
        signed_ms=distribution([s*MILLISECONDS for s in shifts]), absolute_ms=distribution([abs(s)*MILLISECONDS for s in shifts]),
        rows=[dict(side=left[i]['side'], reference_sec=left[i][field], candidate_sec=right[j][field])
              for i, j in valid])


def physics_comparison(reference: Path, candidate: Path) -> dict:
    a, al = read_physics(reference)
    b, bl = read_physics(candidate)
    chains = match_times(a, b, 'trigger_sec')
    return dict(chain_start=timing_result(a, b, chains, 'observed_sec'),
        chain_end=timing_result(a, b, chains, 'end_signal_sec'),
        landing=timing_result(al, bl, match_times(al, bl, 't_sec'), 't_sec'))


def summarize_recognition(path: Path, start: float = START_SEC, end: float = END_SEC) -> dict:
    data = load_audit(path/'recognition.npz', start, end)
    return dict(recognition_wall_ms=distribution(data['recognition_ms'].tolist()),
        recognition_cpu_ms=distribution(data['cpu_ms'].tolist()),
        recognition_cpu_wall_ratio=float(data['cpu_ms'].sum()/data['recognition_ms'].sum()),
        recognition_total_cpu_seconds=float(data['cpu_ms'].sum()/MILLISECONDS),
        queue_put_ms=distribution(data['queue_put_ms'].tolist()))


def summarize_case(path: Path) -> dict:
    metrics = json.loads((path/'metrics.json').read_text())
    command = metrics['command']
    start, end = (float(command[command.index(flag)+1]) for flag in ('--start-sec', '--end-sec'))
    return dict(dropped=metrics['capture_dropped_in_measured_window'], expected=metrics['expected_frames'],
        drop_fraction=metrics['capture_drop_fraction'], gated=metrics['gated_frames'],
        gpu=metrics['gpu'], evaluation_queue=metrics['evaluation_queue'],
        capture_to_sse=metrics['latency_ms']['capture_to_sse'], runtime=metrics['cpu_runtime'],
        **summarize_recognition(path, start, end),
        state_update_ms=metrics['state_update_ms'], calculation_ms=metrics['probability_calculation_ms'],
        loadavg_start=metrics['loadavg_start'], loadavg_end=metrics['loadavg_end'])


def summarize_isolated(root: Path) -> dict:
    paths = [root/f'recognition_only_{threads}' for threads in (0, 1)]
    if not all((path/'recognition.npz').exists() for path in paths):
        return {}
    left, right = [load_audit(path/'recognition.npz') for path in paths]
    return dict(conditions={path.name: dict(**summarize_recognition(path),
        **json.loads((path/'recognition.json').read_text())) for path in paths},
        identical={key: bool(np.array_equal(left[key], right[key]))
                   for key in ('t_sec', 'raw', 'boards', 'stable', 'states', 'active')})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=Path('logs/live_b8_diagnosis'))
    options = parser.parse_args()
    root = options.input
    report = dict(boards=board_comparison(load_audit(root/'full/recognition.npz'),
                                         load_audit(root/'realtime/recognition.npz')),
                  physics=physics_comparison(root/'full/events.jsonl', root/'realtime/events.jsonl'))
    save(root/'impact.json', report)
    cases = {path.name: summarize_case(path) for path in root.iterdir()
             if path.is_dir() and (path/'metrics.json').exists()}
    save(root/'cpu_comparison.json', cases)
    save(root/'recognition_only_comparison.json', summarize_isolated(root))
    print(json.dumps({k: v for k, v in report['boards'].items() if k != 'sides'}, ensure_ascii=False))


if __name__ == '__main__':
    main()
