"""積み増し曲線の外挿、状態条件付き発火手数分布、両側同時進行。"""
from __future__ import annotations
import numpy as np
from src.scoring import CHAIN_POWER_TABLE, compute_effective_rate
from src.b3_timing import STAGE_SEC_DEFAULT, TAIL_SEC_DEFAULT, SEC_PER_HAND_DEFAULT
from src.exchange_projection import hands_for

MAX_K = 12
MAX_RETURN_K = 48
STATE_EDGES = (20, 40, 55)
OJAMA_EDGE = 6
FIRE_EDGE = 100.
SAFE_CELLS = 73
GROUP_SIZE = 4.
PAIR_SIZE = 2.
SCORE_UNIT = 10.
COUNT_UNIT = 30.
SCORE_TABLE = np.r_[0., np.cumsum(np.maximum(1., CHAIN_POWER_TABLE)*GROUP_SIZE*SCORE_UNIT)]
HORIZONS = np.arange(MAX_K+1)


def board_state(record: dict, edges: tuple = STATE_EDGES, garbage_edge: float = OJAMA_EDGE,
                fire_edge: float = FIRE_EDGE) -> tuple[int, int, int]:
    board = np.asarray(record['request']['board'])
    material = int(np.count_nonzero((board >= 1) & (board <= 5)))
    garbage = int(np.count_nonzero(board == 9))
    rate = compute_effective_rate(record['request']['elapsed_sec'], from_first_move=True)
    state = int(np.searchsorted(edges, material))*4+int(garbage > garbage_edge)*2
    return material, garbage, state+int(record['scores'][0]/rate > fire_edge)


def expanded_curve(record: dict, epsilon: float, discount: float = 0.) -> tuple[np.ndarray, np.ndarray]:
    """Rustが完了した点は保持。未計算の先だけ材料増加分を連鎖数へ外挿。"""
    scores, chains = np.asarray(record['scores']), np.asarray(record['chains'])
    material, garbage, _ = board_state(record)
    last = len(scores)-1
    horizon = np.arange(MAX_RETURN_K+1)
    index = np.minimum(horizon, last)
    score, chain = scores[index].copy(), chains[index].copy()
    additional = np.minimum(PAIR_SIZE*np.maximum(0, horizon-last),
                            max(0., SAFE_CELLS-garbage-material-PAIR_SIZE*last))
    anchor_chain = np.interp(scores[-1], SCORE_TABLE, np.arange(len(SCORE_TABLE)))
    grown = anchor_chain+epsilon*additional/GROUP_SIZE
    tail = horizon > last
    score[tail] = np.interp(grown[tail], np.arange(len(SCORE_TABLE)), SCORE_TABLE)
    chain[tail] = grown[tail]
    rate = compute_effective_rate(max(0., record['request']['elapsed_sec']), from_first_move=True)
    return score/rate*np.exp(-discount*horizon), chain


def distribution(record: dict, model: dict) -> np.ndarray:
    material, garbage, state = board_state(record, tuple(model.get('state_edges', STATE_EDGES)),
        model.get('garbage_edge', OJAMA_EDGE), model.get('fire_edge', FIRE_EDGE))
    probabilities = np.array(model['state_probabilities'].get(str(state), model['global_probabilities']))
    # 空き容量を越す積み増しには質量を置かない。k=0は必ず残る。
    capacity = max(0, int((SAFE_CELLS-material-garbage)//PAIR_SIZE))
    probabilities[HORIZONS > capacity] = 0.
    if probabilities.sum() == 0:
        probabilities[0] = 1.
    return probabilities/probabilities.sum()


def direction(own: dict, other: dict, model: dict, discount: float = 0.,
              aggregation: str = 'expectation') -> dict:
    a, chains = expanded_curve(own, model['epsilon'], discount)
    b, _ = expanded_curve(other, model['epsilon'], discount)
    local_chains = chains[:MAX_K+1]
    duration = np.where(local_chains > 0, STAGE_SEC_DEFAULT*local_chains+TAIL_SEC_DEFAULT, 0.)
    wait = np.array([hands_for(float(seconds), SEC_PER_HAND_DEFAULT) for seconds in duration])
    returned_at = np.minimum(MAX_RETURN_K, HORIZONS+wait)
    probabilities = distribution(own, model)
    attack, returned = a[:MAX_K+1], b[returned_at]
    net_a = np.maximum(0., attack-own['request']['pending'])
    net_b = np.maximum(0., returned-other['request']['pending'])
    if aggregation == 'maximum':
        allowed = probabilities > 0
        k = int(np.argmax(np.where(allowed, net_a-net_b, -np.inf)))
        probabilities = np.eye(MAX_K+1)[k]
    elif aggregation != 'expectation':
        raise ValueError('脅威集約の種類が不正')
    return dict(attack=float(probabilities@net_a), returned=float(probabilities@net_b),
        a_gross=float(probabilities@attack), b_gross=float(probabilities@returned),
        expected_k=float(probabilities@HORIZONS), expected_response_k=float(probabilities@returned_at),
        curve=attack.tolist(), probabilities=probabilities.tolist(),
        input_sec=own['input_sec'], receiver_input_sec=other['input_sec'])


def predict_pair(records: tuple[dict | None, dict | None], model: dict,
                 discount: float = 0., aggregation: str = 'expectation') -> dict:
    if any(record is None for record in records):
        return dict(a=0., b=0., sides=None)
    sides = [direction(records[i], records[1-i], model, discount, aggregation) for i in (0, 1)]
    return dict(a=(sides[0]['attack']-sides[1]['attack'])/COUNT_UNIT,
                b=(sides[0]['returned']-sides[1]['returned'])/COUNT_UNIT, sides=sides)
