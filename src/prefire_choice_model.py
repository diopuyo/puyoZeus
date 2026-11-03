"""催促・本線の混合予測。絶対量を左右別に保持し、表示直前だけ差を取る。"""
from __future__ import annotations
import numpy as np
from scipy.special import expit
from src.prefire_growth_model import (expanded_curve, distribution, board_state, MAX_K,
    MAX_RETURN_K, HORIZONS, COUNT_UNIT)
from src.b3_timing import STAGE_SEC_DEFAULT, TAIL_SEC_DEFAULT, SEC_PER_HAND_DEFAULT
from src.exchange_projection import hands_for

BRANCHES = ('provocation', 'main')
MATERIAL_SCALE = 78.
FIRE_SCALE = 1000.
NORMALIZATION_RATE = 70.


def branch_record(record: dict, name: str) -> dict:
    return dict(record, **record['branches'][name])


def preserved(record: dict) -> dict:
    """『h手までに返せる量』を保持してから外挿の基準を作る。"""
    score, chain, best = [], [], 0
    for i, value in enumerate(record['scores']):
        if value > record['scores'][best]:
            best = i
        score.append(record['scores'][best])
        chain.append(record['chains'][best])
    return dict(record, scores=score, chains=chain)


def curves(record: dict, growth: dict, discount: float = 0.) -> tuple[np.ndarray, np.ndarray]:
    values = [expanded_curve(branch_record(record, name), growth['epsilon'], discount) for name in BRANCHES]
    # 催促は未計算域で本線相当の火力へ変えない。実測prefix末尾の候補を保持する。
    last = len(record['scores'])-1
    values[0][0][last+1:] = values[0][0][last]
    values[0][1][last+1:] = values[0][1][last]
    return np.asarray([v[0] for v in values]), np.asarray([v[1] for v in values])


def features(own: dict, other: dict) -> np.ndarray:
    def local(record: dict) -> list[float]:
        material, garbage, _ = board_state(record)
        quick = record['branches']['provocation']['scores'][0]/NORMALIZATION_RATE
        main = record['branches']['main']['scores'][0]/NORMALIZATION_RATE
        return [material/MATERIAL_SCALE, garbage/MATERIAL_SCALE,
                record['request']['pending']/FIRE_SCALE, quick/FIRE_SCALE, main/FIRE_SCALE]
    return np.asarray(local(own)+local(other))


def probability(own: dict, other: dict, model: dict) -> float:
    classifier = model['choice']
    return float(expit(features(own, other)@np.asarray(classifier['coef'])+classifier['intercept']))


def receiver_basis(record: dict, growth: dict, discount: float) -> np.ndarray:
    curve = expanded_curve(preserved(record), growth['epsilon'])[0]
    available = np.maximum.accumulate(curve)
    horizons = np.arange(MAX_RETURN_K+1)
    # 受け側の長手数係数は独立。現在能力は減衰させず、増分だけを実測返却量でfit。
    return np.column_stack((np.full(len(curve), available[0]),
                            (available-available[0])*np.exp(-discount*horizons)))


def direction(own: dict, other: dict, model: dict, stamp: float) -> dict:
    growth = model['growth']
    attack, chains = curves(own, growth, model['discount'])
    attack, chains = attack[:, :MAX_K+1], chains[:, :MAX_K+1]
    duration = np.where(chains > 0, STAGE_SEC_DEFAULT*chains+TAIL_SEC_DEFAULT, 0.)
    own_busy = max(0., own.get('busy_until', stamp)-stamp)
    other_busy = max(0., other.get('busy_until', stamp)-stamp)
    available_sec = np.maximum(0., HORIZONS*SEC_PER_HAND_DEFAULT+duration+own_busy-other_busy)
    at = np.array([[min(MAX_RETURN_K, hands_for(float(s), SEC_PER_HAND_DEFAULT)) for s in row]
                   for row in available_sec])
    basis = receiver_basis(other, growth, model['receiver_discount'])
    returned = (basis@np.asarray(model['receiver_beta']))[at]
    p = probability(own, other, model)
    probabilities = np.asarray([p, 1-p])[:, None]*distribution(own, growth)[None, :]
    net_a = np.maximum(0., attack-own['request']['pending'])
    net_b = np.maximum(0., returned-other['request']['pending'])
    mean = lambda value: float(np.sum(probabilities*value))
    return dict(attack=mean(net_a), returned=mean(net_b), a_gross=mean(attack), b_gross=mean(returned),
        expected_k=mean(HORIZONS), expected_response_k=mean(at), provocation_probability=p,
        curve=np.sum(np.asarray([p, 1-p])[:, None]*attack, axis=0).tolist(),
        branches=[dict(name=name, probability=float(weight), attack=float(distribution(own, growth)@attack[i]),
            K=float(distribution(own, growth)@(at[i]-HORIZONS)),
            duration_sec=float(distribution(own, growth)@duration[i])) for i, (name, weight) in
            enumerate(zip(BRANCHES, (p, 1-p)))], receiver_basis=np.sum(probabilities[:, :, None]*basis[at], axis=(0, 1)).tolist(),
        input_sec=own['input_sec'], receiver_input_sec=other['input_sec'])


def predict_pair(records: tuple, model: dict, stamp: float, current_states: list | None = None) -> dict:
    if not all(records):
        return dict(a=0., b=0., sides=None, available=False, reason='入力未到達')
    if any(r['ready_sec'] > stamp or max(r['request']['evidence_times']) > stamp for r in records):
        raise ValueError('未来の入力または未公開値')
    sides = [direction(records[i], records[1-i], model, stamp) for i in (0, 1)]
    return dict(a=(sides[0]['attack']-sides[1]['attack'])/COUNT_UNIT,
        b=(sides[0]['returned']-sides[1]['returned'])/COUNT_UNIT, sides=sides, available=True, reason='有効')
