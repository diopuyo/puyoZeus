"""保存済みv3.4初回入力・v3.10式通知から、送りと同じ候補の終了盤を取り出す。"""
from __future__ import annotations

import sys as _public_release_sys
from pathlib import Path as _PublicReleasePath
_public_release_root = str(_PublicReleasePath(__file__).resolve().parents[0])
if _public_release_root not in _public_release_sys.path:
    _public_release_sys.path.insert(0, _public_release_root)
from src.public_dev_paths import resolve_text as _public_dev_text

import json
from pathlib import Path
from time import perf_counter
from scripts.exchange_review_io import save
from scripts.replay_exchange_v310 import initialize, on_formula, on_score, sent_values
from src.prefire_worker import LOW_SPEC_RATIO
from src.exchange_correction_v310 import observation_gate
from src.exchange_count_prediction import COLS, simulate

MS = 1000.
BASE = Path(_public_dev_text('<work>/exch_review_render'))


def sources(video: int) -> list[tuple[dict, dict]]:
    if video == 52:
        rows = json.loads((BASE/'v4_3/post52/prediction_inputs.json').read_text())
        return [(r['input'], r['stream']) for r in rows]
    root = BASE/'v3_10'
    inputs = json.loads((root/('video80_inputs.json' if video == 80 else 'inputs.json')).read_text())
    return [(inp, json.loads((root/f'streams/{video}_{inp["event_id"]}.json').read_text()))
            for inp in inputs if inp['video'] == video]


def snapshot(state: dict, inp: dict, policy: dict, elapsed: float, initial: bool) -> dict:
    record = state['after'][-1]
    values = (state['initial']['variants']['board_candidates'] if initial else
              sent_values(state['pool'], inp, state['formulas'], state['reflected'], policy))
    best = max(values, key=lambda value: value['probability'], default=None)
    output = dict(game=inp['game'], side=inp['side'], trigger=inp['trigger_sec'],
        input_sec=record['input_sec'], evidence_times=record['evidence_times'],
        low_spec_ms=elapsed, event_id=inp['event_id'], source='v3.4/v3.10')
    if best:
        output.update(board=best['final'], scores=best['scores'], sent=best['sent'])
    return output


def replay(inp: dict, stream: dict, policy: dict, native: object) -> list[dict]:
    began = perf_counter()
    state = initialize(inp, policy, native)
    elapsed = ((perf_counter()-began)*MS+sum(s.get('image_compute_ms', 0.) for s in inp['sources']))*LOW_SPEC_RATIO
    result = [snapshot(state, inp, policy, elapsed, True)]
    result[-1]['low_spec_ms'] = ((perf_counter()-began)*MS+sum(s.get('image_compute_ms', 0.) for s in inp['sources']))*LOW_SPEC_RATIO
    messages = [(r['t_sec'], 0, r) for r in stream['formulas']]
    messages += [(r['t'], 1, r) for r in stream['scores']]
    messages += [(r['t'], 2, r) for r in stream['frames']]
    for stamp, kind, value in sorted(messages, key=lambda item: item[:2]):
        assert stamp >= inp['input_sec']
        began, length = perf_counter(), len(state['after'])
        if kind == 0:
            on_formula(state, value, inp, policy, native)
        elif kind == 1:
            on_score(state, value, inp, policy)
        else:
            result.append(terminal_observation(state, value, stream['initial_next'], inp, policy, native))
        if len(state['after']) > length:
            result.append(snapshot(state, inp, policy, (perf_counter()-began)*MS*LOW_SPEC_RATIO, False))
            result[-1]['low_spec_ms'] = (perf_counter()-began)*MS*LOW_SPEC_RATIO
    for row in result:
        assert all(t <= row['input_sec'] for t in row['evidence_times'])
    return result


def image_observation_needed(state: dict, frame: dict, initial_next: list | None,
                             inp: dict, policy: dict, native: object) -> bool:
    """既存ゲートの必要条件を確定盤で先に検査し、不要な再画像推論を省く。"""
    if (state['corrected'] or not frame['normal_score'] or frame['state'] != 'STABLE'
            or not state['formulas'] or frame['t']-state['formulas'][-1]['t'] < policy['quiet_sec']
            or initial_next is None or state['next_changes'] > 1):
        return False
    board, anchors, colors = frame.get('board'), state['anchors'], inp['colors']
    if board is None or not anchors or any(board[:COLS*2]) or any(c not in (0, 9, *colors) for c in board):
        return False
    if board.count(9) > max(v['origin'].count(9) for v in anchors):
        return False
    maximum = max(sum(c in colors for c in v['origin']) for v in anchors)
    if sum(c in colors for c in board) > maximum-sum(f['erased'] for f in state['formulas']):
        return False
    return not simulate(board, native)['erased']


def terminal_observation(state: dict, frame: dict, initial_next: list | None,
                         inp: dict, policy: dict, native: object) -> dict:
    """v3.10の既存ゲートだけで完走を確認。生のNEXT動きや単発STABLEは信用しない。"""
    began = perf_counter()
    previous_next = state['last_next'] if state['last_next'] is not None else initial_next
    state['next_changes'] += int(frame.get('next') != previous_next)
    state['last_next'] = frame.get('next')
    frame = dict(frame, next_changes=state['next_changes'])
    evidence = [s['t'] for s in frame['sources']]+[frame['t'], frame['score_t']]
    evidence += [f['t'] for f in state['formulas']+state['previous']]
    eligible = image_observation_needed(state, frame, initial_next, inp, policy, native)
    remaining = observation_gate(frame, initial_next, state['anchors'], state['formulas'],
        state['previous'], inp['colors'], policy, native)[0] if eligible else None
    value = dict(game=inp['game'], side=inp['side'], trigger=inp['trigger_sec'],
        input_sec=frame['t'], evidence_times=evidence, event_id=inp['event_id'], source='v3.10完走ゲート')
    if remaining is not None and not state['corrected']:
        best = snapshot(state, inp, policy, 0., False)
        value.update({k: best[k] for k in ('board', 'scores', 'sent') if k in best})
        value['completed'] = True
        state['corrected'] = True
    state['previous'].append({k: frame.get(k) for k in ('t', 'state', 'board', 'next')})
    image_ms = sum(s.get('image_compute_ms', 0.) for s in frame['sources']) if eligible else 0.
    value['low_spec_ms'] = ((perf_counter()-began)*MS+image_ms)*LOW_SPEC_RATIO
    return value


def collect(video: int, destination: Path, native: object, policy: dict) -> list[dict]:
    result = []
    for inp, stream in sources(video):
        result.extend(replay(inp, stream, policy, native))
    result.sort(key=lambda row: row['input_sec'])
    save(destination/f'origins_{video}.json', result)
    print('初回・逐次候補', video, len(result), flush=True)
    return result
