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


def capped(rows: list[tuple], cap: int | None) -> list[tuple]:
    """候補が cap 個を超えるとき、重みの大きい順 (同点は元の順) に cap 個へ絞り、合計の重みは保つ。

    既定 (cap=None) は従来どおり全候補。決定的で、壁時計に依存しない。
    """
    if cap is None or len(rows) <= cap:
        return rows
    keep = sorted(sorted(range(len(rows)), key=lambda i: (-rows[i][1], i))[:cap])
    total, mass = sum(w for _, w in rows), sum(rows[i][1] for i in keep)
    scale = total / mass if mass > 0 else 1.0
    return [(rows[i][0], rows[i][1] * scale) for i in keep]


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
    rows = [(tuple(score for score, _ in row), prod(w for _, w in row))
            for row in product(*distributions)]
    return capped(rows, getattr(engine, 'scenario_cap', None))


def totals(chains: list, scores: tuple) -> np.ndarray:
    """対称化前の1P/2P順で得点を集計する。"""
    return np.array([sum(s for c,s in zip(chains,scores) if c.side == side) for side in ('1P','2P')])


# 同じ側別得点の組は同じ S3 勝率 (純関数)。候補列挙で繰り返し現れるため 1 回だけ評価する (出力同一)。
EXACT_S3_CACHE = True


def s3_probability(cache: dict, event: ExchangeEndInput, chains: list, scores: tuple, models: Any) -> float:
    """側別得点が同じ候補の S3 勝率を再利用する。cache は 1 回の呼出し内だけで使う。"""
    after = totals(chains, scores)
    key = after.tobytes()
    if not EXACT_S3_CACHE or key not in cache:
        cache[key] = evaluate_exchange_event(replace(event, scores_after=after), models)
    return cache[key]


def weighted_s3(engine: Any, tracker: Any, event: ExchangeEndInput) -> float:
    """モデル入力の平均ではなく、各候補のモデル勝率を平均する。"""
    chains = tracker.current.chains if tracker.current else []
    if not any(engine.active(c) for c in chains):
        return evaluate_exchange_event(event, tracker.models)
    cache: dict = {}
    return sum(weight*s3_probability(cache, event, chains, scores, tracker.models)
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
    engine = getattr(tracker, 'hidden_row_belief', None)
    record = tracker.current or projection.death_record
    if engine is None or record is None or not any(engine.active(c) for c in record.chains):
        return None
    if tracker.firing is None:
        # 撃ち合いの記録 (death_record) だけが残り発火入力が無い時刻では S3 を組めない。
        # 旧実装は ExchangeEndInput で ValueError になり再生が止まった (zenchi 第32〜33試合、2026-10-01)。
        # 加重なし (呼出元の確率をそのまま使う) へ落とす。発火入力がある時刻の値は変わらない。
        return None
    probability, gfe_mean, counter = 0., 0., {}
    dropped = (snapshot.total_dropped_to_p1, snapshot.total_dropped_to_p2)
    event = ExchangeEndInput(tracker.firing, np.zeros(2), np.zeros(2), tracker._score_elapsed)
    s3_cache: dict = {}
    for scores, weight in scenarios(engine, record.chains):
        with conditional_scores(engine, record.chains, scores):
            incoming = projection._incoming(tracker, dropped, record)
            s3 = s3_probability(s3_cache, event, record.chains, scores, tracker.models)
            if not any(incoming):
                probability += weight*s3
                gfe_mean += weight*s3
                continue
            gfe, counter = projection._probability_inputs(overlay, snapshot, latest, incoming, hands, stamp)
            probability += weight*logit_mean(s3, gfe)
            gfe_mean += weight*gfe
    return probability, gfe_mean, dict(counter, hidden_row_weighted=True)
