"""正解を開かない配送順リプレイ。採点ラベルと最終段通知を推定器に渡さない。"""
from __future__ import annotations

from copy import deepcopy
from time import perf_counter
from typing import Any
from scripts.predict_exchange_counts_v34 import predict
from scripts.predict_exchange_counts_v33 import with_send, formula_step
from src.exchange_count_prediction import distribution, LOW_SPEC_RATIO
from src.exchange_count_v34 import probabilities, condition
from src.exchange_count_v310 import update
from src.exchange_count_correction import RemainingCorrection
from src.exchange_correction_v310 import observation_gate
from src.exchange_count_ledger import gross, STEP_SEC, FORMULA_SCORE_DELAY_SEC

MS, MAX_LOW_SPEC_MS, FORMULA_GAP = 1000, 500, 2.


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


def on_frame(state: dict, frame: dict, initial_next: list | None, inp: dict, policy: dict, native: object) -> None:
    stamp = frame['t']
    previous_next = state['last_next'] if state['last_next'] is not None else initial_next
    state['next_changes'] += int(frame.get('next') != previous_next)
    state['last_next'] = frame.get('next')
    frame = dict(frame, next_changes=state['next_changes'])
    evidence = [s['t'] for s in frame['sources']]+[stamp, frame['score_t']]
    if frame['sources']:
        image_ms = sum(s.get('image_compute_ms', 0.) for s in frame['sources'])
        began = perf_counter()
        correction = state['old_correction'].observe(stamp, frame['state'], frame['normal_score'],
            frame['sources'], inp['colors'], native, allow_visual_stable=True)
        if correction:
            state['current']['before']['remaining'] = correction['remaining']
            state['before'].append(record('observation', stamp, (perf_counter()-began)*MS+image_ms,
                                          state['current']['before'], evidence))
        began = perf_counter()
        remaining, reason = observation_gate(frame, initial_next, state['anchors'],
            state['formulas'], state['previous'], inp['colors'], policy, native)
        state['rejections'][reason] = state['rejections'].get(reason, 0)+1
        elapsed = (perf_counter()-began)*MS+image_ms
        if elapsed*LOW_SPEC_RATIO > MAX_LOW_SPEC_MS:
            raise RuntimeError('観測処理の時間超過')
        state.setdefault('attempt_ms', []).append(elapsed*LOW_SPEC_RATIO)
        if remaining is not None and not state['corrected']:
            state['corrected'] = True
            state['current']['after']['remaining'] = remaining
            times = evidence+[f['t'] for f in state['formulas']]+[f['t'] for f in state['previous']]
            state['after'].append(record('observation', stamp, elapsed, state['current']['after'], times))
    state['previous'].append({k: frame.get(k) for k in ('t', 'state', 'board', 'next')})


def replay(inp: dict, stream: dict, policy: dict, native: object) -> dict:
    state = initialize(inp, policy, native)
    messages = [(s['t_sec'], 0, s) for s in stream['formulas']]
    messages += [(s['t'], 1, s) for s in stream['scores']]
    messages += [(s['t'], 2, s) for s in stream['frames']]
    for stamp, kind, value in sorted(messages, key=lambda v: v[:2]):
        if stamp < inp['input_sec']:
            raise ValueError('配送順違反')
        if kind == 0:
            on_formula(state, value, inp, policy, native)
        elif kind == 1:
            on_score(state, value, inp, policy)
        else:
            on_frame(state, value, stream['initial_next'], inp, policy, native)
    for version in ('before', 'after'):
        ready = inp['input_sec']
        for row in state[version]:
            row['dependency_ready_sec'] = ready
            row['publish_sec'] = max(row['input_sec'], ready)+row['low_spec_ms']/MS
            ready = row['publish_sec']
    return {k: state[k] for k in ('before', 'after', 'repairs', 'rejections', 'attempt_ms') if k in state}
