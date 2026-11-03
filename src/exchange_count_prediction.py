"""レビュー専用の個数分布。既定の本番経路には接続しない。物理はRustのみ。"""
from __future__ import annotations
from src import prefire_work_budget as work_budget

from collections import Counter
from math import exp, log
from time import perf_counter
from typing import Any

COLS, CELLS, SCORE_UNIT, MS = 6, 78, 10, 1000
COUNT_COLORS = (1, 2, 3, 4, 5, 9)
LOW_SPEC_RATIO = 1.6473854230
DEADLINE_SEC = .5/LOW_SPEC_RATIO


def histogram(board: list[int]) -> list[int]:
    counts = Counter(board)
    return [counts[c] for c in COUNT_COLORS]


def simulate(board: list[int], native: Any, compact: bool = False) -> dict:
    """Python側は数え上げだけを行い、連鎖・重力を進めない。"""
    if len(board) != CELLS or any(c not in (0, *COUNT_COLORS) for c in board):
        raise ValueError('不正な盤面')
    summary = getattr(native, 'simulate_chain_summary_py', None) if compact else None
    if summary is not None:
        erased, scores, final = summary(board, True)
        final = final if erased else board
    else:
        trace = native.simulate_chain_trace_py(board, True)
        erased = [sum(len(g[1]) for g in step[2]) for step in trace]
        scores = [step[4] for step in trace]
        final = trace[-1][1] if trace else board
    return dict(origin=board, erased=erased, scores=scores,
                bonuses=[s//(n*SCORE_UNIT) for s, n in zip(scores, erased)],
                score=sum(scores), final=final, remaining=histogram(final))


def validate_times(stamp: float, evidence: list[dict]) -> None:
    if any(e['t'] > stamp for e in evidence):
        raise ValueError('未来の観測は予測へ渡せません')


def cell_probabilities(sources: list[dict], colors: list[int], weight: float) -> list[dict]:
    """CNN・画像の観測票から候補確率を作る。試合外色をおじゃまへ決め打ちしない。"""
    result = []
    modality_counts = Counter(s['kind'] for s in sources)
    for cell in range(CELLS):
        votes: Counter = Counter()
        for source in sources:
            if cell < COLS and source['kind'] == 'image':
                continue  # 隠し段を画像で観測したことにしない。
            w = weight if source['kind'] == 'image' else 1-weight
            w /= modality_counts[source['kind']]
            value = source['board'][cell]
            if value in (0, 9, *colors):
                votes[value] += w
        if not votes or not sum(votes.values()):
            votes.update(dict.fromkeys((0, 9, *colors), 1.))
        total = sum(votes.values())
        result.append({c: v/total for c, v in votes.items() if v > 0})
    return result


def beam_boards(probabilities: list[dict], width: int) -> list[tuple]:
    beam = [(0., [])]
    for distribution in probabilities:
        beam = sorted(((score+log(p), board+[color]) for score, board in beam
                       for color, p in distribution.items()), key=lambda x: -x[0])[:width]
    return beam


def candidates(sources: list[dict], legacy: list[dict], colors: list[int],
               policy: dict, stamp: float, native: Any) -> dict:
    """到着済み情報だけから生成。ラベル・終了盤・実得点の引数は持たない。"""
    began = perf_counter()
    validate_times(stamp, sources)
    probabilities = cell_probabilities(sources, colors, policy['image_weight']) if sources else []
    boards = beam_boards(probabilities, policy['beam_width']) if probabilities else []
    for source in sources:
        board = source['board']
        if all(c in (0, 9, *colors) for c in board):
            boards.append((sum(log(probabilities[i].get(c, policy['prob_floor']))
                               for i, c in enumerate(board)), board))
    boards.extend((sum(log(probabilities[i].get(c, policy['prob_floor']))
                       for i, c in enumerate(v['origin'])) if probabilities else 0., v['origin']) for v in legacy)
    unique, output = set(), []
    for score, board in sorted(boards, key=lambda x: -x[0]):
        if (len(unique) >= work_budget.CANDIDATE_SIMULATIONS if work_budget.enabled() else perf_counter()-began >= DEADLINE_SEC):
            break
        if tuple(board) in unique:
            continue
        unique.add(tuple(board))
        result = simulate(board, native)
        if result['erased']:
            output.append(dict(result, log_weight=score))
    elapsed = (perf_counter()-began)*MS
    return dict(candidates=output, elapsed_ms=elapsed, low_spec_ms=elapsed*LOW_SPEC_RATIO,
                exhausted=len(unique) < len({tuple(b) for _, b in boards}),
                input_sec=stamp, evidence_times=[s['t'] for s in sources])


def condition(values: list[dict], formulas: list[dict], stamp: float) -> tuple[list[dict], bool]:
    """段ごとに消去数で絞る。全滅なら矛盾を明示し、候補ゼロを確定値にしない。"""
    validate_times(stamp, formulas)
    accepted = [v for v in values if all(s['step'] <= len(v['erased']) and
                v['erased'][s['step']-1] == s['erased'] for s in formulas)]
    return accepted, bool(formulas and not accepted)


def distribution(values: list[dict], temperature: float = 1.) -> list[dict]:
    if not values:
        return []
    peak = max(v['log_weight'] for v in values)
    weights = [exp((v['log_weight']-peak)/temperature) for v in values]
    total = sum(weights)
    return [dict(v, probability=w/total) for v, w in zip(values, weights)]


def interval(values: list[dict], key: str, index: int | None = None) -> dict | None:
    """候補の全支持区間。較正済み90%区間と称さず、実測被覆率を別に採点する。"""
    if not values:
        return None
    numbers = [v[key] if index is None else (v[key][index] if index < len(v[key]) else 0) for v in values]
    mean = sum(v['probability']*x for v, x in zip(values, numbers))
    return dict(value=mean, low=min(numbers), high=max(numbers))
