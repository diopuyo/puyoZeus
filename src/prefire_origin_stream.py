"""v4.9fの発火起点予測。元レビュー関数から学習・ファイル入出力依存を分離。"""
from __future__ import annotations
from copy import deepcopy
from time import perf_counter
from typing import Any
from src.exchange_count_prediction import distribution, LOW_SPEC_RATIO, COLS, simulate
from src.exchange_count_v34 import candidates, probabilities, condition
from src.exchange_count_v310 import update
from src.exchange_count_correction import RemainingCorrection
from src.exchange_correction_v310 import observation_gate
from src.exchange_count_ledger import gross, STEP_SEC, FORMULA_SCORE_DELAY_SEC
from src.prefire_confidence import candidate_summary

MS, MAX_LOW_SPEC_MS, FORMULA_GAP = 1000., 500., 2.



def with_send(values: list[dict], stamp: float, ledger: dict) -> list[dict]:
    for value in values:
        times = [stamp+FORMULA_SCORE_DELAY_SEC+STEP_SEC*i for i in range(len(value['scores']))]
        value['sent'] = gross(value['scores'], times, ledger['carry'], ledger['first_move'])[0]
    return values


def formula_step(row: dict, step: int) -> dict:
    return dict(t=row['t_sec'], step=step, erased=row['left']//10, bonus=row['right'])


def predict(inputs: dict, policy: dict, native: object) -> dict:
    began = perf_counter()
    stamp, sources = inputs['input_sec'], inputs['sources']
    out = candidates(sources, inputs['legacy'], inputs['colors'], policy, stamp,
                     native, inputs['history'], inputs['pair'])
    accepted, conflict = condition(out['candidates'], [inputs['first']], stamp)
    ledger = inputs['ledger']
    if ledger['observed_sec'] > stamp:
        raise ValueError('未来の得点台帳')
    values = distribution(with_send(deepcopy(accepted or out['candidates']), stamp, ledger), policy['temperature'])
    elapsed = (perf_counter()-began)*MS+sum(s.get('image_compute_ms', 0.) for s in sources)
    low_spec = elapsed*LOW_SPEC_RATIO
    if low_spec > MAX_LOW_SPEC_MS:
        raise RuntimeError(f'予測時間超過: {low_spec:.2f}ms')
    return dict(event_id=inputs['event_id'], input_sec=stamp, first=inputs['first'], ledger=ledger,
        variants=dict(board_candidates=values), conflicts=dict(board_candidates=conflict),
        elapsed_ms=elapsed, low_spec_ms=low_spec, candidates=out['candidates'],
        evidence_times=out['evidence_times']+[inputs['first']['t'], ledger['observed_sec']],
        status='分布', exhausted=False)


def point(values: list[dict]) -> dict:
    best = max(values, key=lambda v: v['probability'], default={})
    return dict(sent=best.get('sent'), remaining=best.get('remaining'))


def record(kind: str, stamp: float, elapsed: float, value: dict, evidence: list[float], **extra: Any) -> dict:
    low = elapsed*LOW_SPEC_RATIO
    if low > MAX_LOW_SPEC_MS:
        raise RuntimeError(f'低スペック予算超過: {low:.3f}ms')
    if any(t > stamp for t in evidence):
        raise ValueError('未来の入力が予測に混入')
    return dict(kind=kind, input_sec=stamp, publish_sec=stamp+low/MS,
                low_spec_ms=low, evidence_times=evidence, **deepcopy(value), **extra)


def sent_values(values: list[dict], inp: dict, formulas: list[dict], reflected: list[float], policy: dict) -> list[dict]:
    result = deepcopy(values)
    for value in result:
        times = [reflected[i] if i < len(reflected) else
            formulas[i]['t']+FORMULA_SCORE_DELAY_SEC if i < len(formulas) else
            formulas[-1]['t']+FORMULA_SCORE_DELAY_SEC+(i-len(formulas)+1)*STEP_SEC
            for i in range(len(value['scores']))]
        value['sent'] = gross(value['scores'], times, inp['ledger']['carry'], inp['ledger']['first_move'])[0]
    return distribution(result, policy['temperature'])


def initialize(inp: dict, policy: dict, native: object) -> dict:
    initial = predict(inp, policy, native)
    first = record('initial', inp['input_sec'], initial['elapsed_ms'],
                   point(initial['variants']['board_candidates']), initial['evidence_times'])
    pool, _ = condition(initial['candidates'], [inp['first']], inp['input_sec'])
    return dict(initial=initial, before=[first], after=[deepcopy(first)],
        formulas=[inp['first']], pool=pool, anchors=initial['candidates'],
        probs=None,
        current={'before': point(initial['variants']['board_candidates']),
                 'after': point(initial['variants']['board_candidates'])}, reflected=[],
        old_correction=RemainingCorrection(inp['input_sec']), corrected=False,
        previous=[], rejections={}, pending_score=None, score_run=0, last_score=None,
        repairs=[], next_changes=0, last_next=None)


def on_formula(state: dict, message: dict, inp: dict, policy: dict, native: object) -> None:
    stamp = message['t_sec']
    if stamp-state['formulas'][-1]['t'] > FORMULA_GAP:
        return
    state['formulas'].append(formula_step(message, len(state['formulas'])+1))
    evidence = state['initial']['evidence_times']+[f['t'] for f in state['formulas']]
    began = perf_counter()
    values, conflict = condition(state['anchors'], state['formulas'], stamp)
    state['current']['before'] = point(distribution(with_send(deepcopy(values), inp['input_sec'], inp['ledger']), policy['temperature']))
    if state['old_correction'].result is not None:
        state['current']['before']['remaining'] = state['old_correction'].result['remaining']
    state['before'].append(record('formula', stamp, (perf_counter()-began)*MS, state['current']['before'], evidence, conflict=conflict))
    began = perf_counter()
    if state['probs'] is None:
        state['probs'] = probabilities(inp['sources'], inp['colors'], inp['history'], policy)
    out = update(state['pool'], state['anchors'], state['formulas'], stamp, inp['colors'], state['probs'], policy, native)
    state['pool'] = out['candidates']
    value = point(sent_values(state['pool'], inp, state['formulas'], state['reflected'], policy))
    if state['corrected']:
        value['remaining'] = state['current']['after']['remaining']
    state['current']['after'] = value
    state['after'].append(record('formula', stamp, (perf_counter()-began)*MS, value, evidence,
        candidates=len(state['pool']), tested=out['tested'], exhausted=out['budget_exhausted']))
    if out['tested']:
        state['repairs'].append(dict(t=stamp, tested=out['tested'], candidates=len(out['candidates']),
            changes=[v.get('changes', []) for v in out['candidates']]))


def on_score(state: dict, row: dict, inp: dict, policy: dict) -> None:
    value, stamp = row['score'], row['t']
    state['score_run'] = state['score_run']+1 if value is not None and value == state['pending_score'] else int(value is not None)
    state['pending_score'] = value
    if state['score_run'] != 2 or value == state['last_score']:
        return
    state['last_score'] = value
    index = len(state['reflected'])
    if index >= len(state['formulas']) or stamp <= state['formulas'][index]['t']:
        return
    expected = inp['ledger']['score']+sum(f['erased']*10*f['bonus'] for f in state['formulas'][:index+1])
    if value-expected not in (0, 2100):
        return
    state['reflected'].append(stamp)
    evidence = state['initial']['evidence_times']+[f['t'] for f in state['formulas']]+state['reflected']
    began = perf_counter()
    prediction = point(sent_values(state['pool'], inp, state['formulas'], state['reflected'], policy))
    # 送り時刻の更新だけでは、観測補正した残存を上書きしない。
    state['current']['after']['sent'] = prediction['sent']
    state['after'].append(record('score_reflection', stamp, (perf_counter()-began)*MS, state['current']['after'], evidence))
    state['before'].append(record('score_reflection', stamp, 0., state['current']['before'], evidence))


def base_snapshot(state: dict, inp: dict, policy: dict, elapsed: float, initial: bool) -> dict:
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


def snapshot(state: dict, inp: dict, policy: dict, elapsed: float, initial: bool) -> dict:
    output = base_snapshot(state, inp, policy, elapsed, initial)
    values = (state['initial']['variants']['board_candidates'] if initial else
              sent_values(state['pool'], inp, state['formulas'], state['reflected'], policy))
    conflict = state['initial']['conflicts']['board_candidates'] if initial else not bool(values)
    output['confidence'] = candidate_summary(values, len(state['formulas']),conflict)
    return output
