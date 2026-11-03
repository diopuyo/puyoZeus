"""明示呼出し専用。穴の補修を接地判定より先に行い、fit固定の順位器を使う。"""
from __future__ import annotations

from itertools import combinations
from math import log, exp
from typing import Any
from src.exchange_count_prediction import CELLS, COLS, simulate, validate_times
from src.exchange_count_v34 import candidates as baseline, grounded, delta_distance

MAX_BASES, MAX_REPAIRS, MAX_DOUBLE_CELLS = 8, 1536, 6
LOG_SCALE, MAX_CHAIN = 100., 20.
MAIN_SCORE, PRIOR_FLOOR = 1000, 1e-6
FEATURE_NAMES = ('観測対数差', '連鎖長', '色ぷよ量', '消去割合', '残る色ぷよ量',
                 '初段消去数', '初段消去深さ', '充填量と連鎖長', '初段後の消去量')


def alternatives(board: list[int], sources: list[dict], colors: list[int]) -> list[tuple]:
    result = []
    for cell, value in enumerate(board):
        votes = {s['board'][cell] for s in sources if s['board'][cell] in (0, 9, *colors)}
        hole = value == 0 and cell >= COLS and board[cell-COLS] != 0
        if value not in (0, 9, *colors) or hole:
            votes.update((0, 9, *colors))
        if cell < COLS:
            continue  # 隠し段は既存の履歴分布で生成する。
        for color in sorted(votes-{value}):
            result.append((cell, color, int(hole or value not in (0, 9, *colors))))
    return sorted(result, key=lambda x: (-x[2], x[0], x[1]))


def repaired_boards(sources: list[dict], colors: list[int], bases: list[list[int]]) -> list[list[int]]:
    result, seen = [], set()
    for board in bases[:MAX_BASES]:
        options = alternatives(board, sources, colors)
        chosen = list(dict.fromkeys(i for i, _, _ in options))[:MAX_DOUBLE_CELLS]
        doubles = [(a, b) for a, b in combinations(options, 2)
                   if a[0] != b[0] and a[0] in chosen and b[0] in chosen]
        for changes in [(), *((o,) for o in options), *doubles]:
            value = board.copy()
            for cell, color, _ in changes:
                value[cell] = color
            # 読めなかった隠し段は既存候補に任せ、ここでは空の仮説を一つだけ残す。
            value[:COLS] = [c if c in (0, 9, *colors) else 0 for c in value[:COLS]]
            value = grounded(value)
            if tuple(value) in seen or any(c not in (0, 9, *colors) for c in value):
                continue
            seen.add(tuple(value))
            result.append(value)
            if len(result) >= MAX_REPAIRS:
                return result
    return result


def candidates(inp: dict, policy: dict, native: Any) -> dict:
    stamp, sources = inp['input_sec'], inp['sources']
    validate_times(stamp, sources)
    out = baseline(sources, inp['legacy'], inp['colors'], policy, stamp,
                   native, inp['history'], inp['pair'])
    bases = [s['board'] for s in reversed(sources)]
    bases += [v['origin'] for v in sorted(out['candidates'], key=lambda v: -v['log_weight'])]
    probs, existing = out['probabilities'], {tuple(v['origin']) for v in out['candidates']}
    boards = repaired_boards(sources, inp['colors'], bases)
    boards += pair_boards(inp, native)
    for board in boards:
        if tuple(board) in existing:
            continue
        value = simulate(board, native)
        if value['erased']:
            weight = sum(log(probs[i].get(c, policy['prob_floor'])) for i, c in enumerate(board))
            weight -= policy['delta_weight']*delta_distance(board, inp['history'], inp['pair'])
            out['candidates'].append(dict(value, log_weight=weight))
    return out


def pair_boards(inp: dict, native: Any) -> list[list[int]]:
    pair = inp['pair']
    if not pair.get('identified') or not pair.get('pair'):
        return []
    validate_times(inp['input_sec'], [dict(t=pair['observed_sec'])])
    sources = inp['sources']
    image = next((s for s in sources if s['kind'] == 'image'), None)
    cnn = next((s for s in sources if s['kind'] == 'cnn'), None)
    output, seen = [], set()
    for source in (cnn, image):
        if source is None:
            continue
        board = source['board'].copy()
        board[:COLS] = [c if c in (0, 9, *inp['colors']) else 0 for c in board[:COLS]]
        board = grounded(board)
        if any(c not in (0, 9, *inp['colors']) for c in board):
            continue
        variants = [board]
        for cell in range(COLS):
            if board[cell+COLS] == 0:
                continue
            for color in (0, 9, *inp['colors']):
                changed = board.copy()
                changed[cell] = color
                variants.append(changed)
        for value in variants:
            for _, _, placed, _ in native.enumerate_placements_py(value, *pair['pair'], False):
                if tuple(placed) not in seen:
                    seen.add(tuple(placed))
                    output.append(placed)
    return output


def features(value: dict, peak: float, native: Any) -> list[float]:
    colored = sum(value['remaining'][:-1])+sum(value['erased'])
    erased, length = sum(value['erased']), len(value['erased'])
    first = native.simulate_chain_trace_py(value['origin'], True)[0]
    cells = [r for _, group, _ in first[2] for r, _ in group]
    return [max(-1., (value['log_weight']-peak)/LOG_SCALE), length/MAX_CHAIN,
        colored/CELLS, erased/max(1, colored), sum(value['remaining'][:-1])/CELLS,
        value['erased'][0]/CELLS, sum(cells)/max(1, len(cells))/(CELLS/COLS-1),
        colored/CELLS*length/MAX_CHAIN, (erased-value['erased'][0])/CELLS]


def main_features(inp: dict, values: list[dict], native: Any) -> list[float]:
    occupancy = max((sum(c in inp['colors'] for c in s['board']) for s in inp['sources']), default=0)/CELLS
    if not values:
        return [occupancy, 0., 0., occupancy*occupancy, 0.]
    best = max(values, key=lambda v: v['log_weight'])
    f = features(best, best['log_weight'], native)
    return [occupancy, f[5], f[6], occupancy*occupancy, occupancy*f[6]]


def rank(values: list[dict], policy: dict, native: Any, inp: dict | None = None) -> list[dict]:
    if not values:
        return []
    if 'main_prior' in policy:
        prior = policy['main_prior']
        z = prior['intercept']+sum(a*b for a, b in zip(main_features(inp, values, native), prior['weights']))
        probability = min(1-PRIOR_FLOOR, max(PRIOR_FLOOR, 1/(1+exp(-z))))
        return [dict(v, observation_log_weight=v['log_weight'], main_probability=probability,
            log_weight=v['log_weight']+prior['strength']*log(probability if v['score'] >= MAIN_SCORE
                else 1-probability)) for v in values]
    peak = max(v['log_weight'] for v in values)
    weights = policy['ranking_weights']
    return [dict(v, observation_log_weight=v['log_weight'],
        log_weight=sum(a*b for a, b in zip(features(v, peak, native), weights))) for v in values]
