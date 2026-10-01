"""候補ごとの非線形なS3・着弾勝率を求めてから確率重みで平均する。"""
from __future__ import annotations
from collections import Counter
from contextlib import contextmanager
from dataclasses import replace
from itertools import product
from math import prod
from typing import Any, Iterator
import numpy as np
from src.exchange_event_evaluator import ExchangeEndInput, evaluate_exchange_event
from src.scoring import score_to_ojama


def scenarios(engine: Any, chains: list) -> list[tuple]:
    """同じ最終得点の候補を先に合算し、不要なモデル再評価を省く。"""
    distributions = []
    for chain in chains:
        entry = engine.active(chain)
        weights: Counter = Counter()
        if entry:
            for option in entry['options']:
                score = max(option['score'], chain.formula_total or 0., chain.score_delta or 0.)
                weights[score] += option['weight']
        else:
            weights[chain.provisional_score] = 1.
        distributions.append(tuple(weights.items()))
    return [(tuple(score for score, _ in row), prod(w for _, w in row))
            for row in product(*distributions)]


def totals(chains: list, scores: tuple) -> np.ndarray:
    """対称化前の1P/2P順で得点を集計する。"""
    return np.array([sum(s for c,s in zip(chains,scores) if c.side == side) for side in ('1P','2P')])


def weighted_s3(engine: Any, tracker: Any, event: ExchangeEndInput) -> float:
    """モデル入力の平均ではなく、各候補のモデル勝率を平均する。"""
    chains = tracker.current.chains if tracker.current else []
    if not any(engine.active(c) for c in chains):
        return evaluate_exchange_event(event, tracker.models)
    return sum(weight*evaluate_exchange_event(replace(event, scores_after=totals(chains, scores)), tracker.models)
               for scores, weight in scenarios(engine, chains))


@contextmanager
def conditional_scores(engine: Any, chains: list, scores: tuple) -> Iterator[None]:
    """局所計算だけに候補得点を渡し、公開予測と分布は必ず元に戻す。"""
    saved = []
    try:
        for chain, score in zip(chains, scores):
            entry = engine.active(chain)
            if entry is None:
                continue
            saved.append((chain, entry, chain.predicted_final_score, entry['stats']['mean_send']))
            chain.predicted_final_score = score
            entry['stats']['mean_send'] = score_to_ojama(score, elapsed_sec=entry['elapsed']).ojama_count
        yield
    finally:
        for chain, entry, score, send in saved:
            chain.predicted_final_score = score
            entry['stats']['mean_send'] = send


def weighted_landing(projection: Any, overlay: Any, snapshot: Any, latest: tuple,
                     hands: tuple, base: dict, stamp: float) -> tuple | None:
    """死亡判定とは独立に、候補ごとのS3と着弾の合成確率を平均する。"""
    from src.exchange_event_landing import logit_mean
    tracker = overlay.tracker
    # 終了後に死亡記録だけが残った場合は、起点のないS3を作らず呼出元の通常経路を使う。
    if tracker.firing is None:
        return None
    engine = getattr(tracker, 'hidden_row_belief', None)
    record = tracker.current or projection.death_record
    if engine is None or record is None or not any(engine.active(c) for c in record.chains):
        return None
    probability, gfe_mean, counter = 0., 0., {}
    dropped = (snapshot.total_dropped_to_p1, snapshot.total_dropped_to_p2)
    event = ExchangeEndInput(tracker.firing, np.zeros(2), np.zeros(2), tracker._score_elapsed)
    for scores, weight in scenarios(engine, record.chains):
        with conditional_scores(engine, record.chains, scores):
            incoming = projection._incoming(tracker, dropped, record)
            s3 = evaluate_exchange_event(replace(event, scores_after=totals(record.chains, scores)), tracker.models)
            if not any(incoming):
                probability += weight*s3
                gfe_mean += weight*s3
                continue
            gfe, counter = projection._probability_inputs(overlay, snapshot, latest, incoming, hands, stamp)
            probability += weight*logit_mean(s3, gfe)
            gfe_mean += weight*gfe
    return probability, gfe_mean, dict(counter, hidden_row_weighted=True)
