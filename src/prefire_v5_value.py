"""Phase 5 の有限範囲ミニマックス。待つ手も設置後の同じ評価器へ渡す。"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Callable

import numpy as np

from src import prefire_v5_search as search
from src.exchange_event_evaluator import evaluate_exchange_event
from src.indicators_v2 import SEC_PER_HAND
from src.ojama_accounting import ON_FIELD_CAP

ValueFunction = Callable[[search.Exchange, float], float]


def pending_counts(overlay: Any) -> tuple[int, int] | None:
    """本番の未処理台帳を優先し、無ければ観測の両側残量を読む。純差にしない。"""
    safety = getattr(overlay._landing_projection, 'safety', None)
    # ledger_enabledは表示への採用フラグ。observeはOFFでも両側台帳を更新する。
    if safety is not None:
        return tuple(int(v) for v in safety.ledger.pending)
    snapshot = overlay._snapshots[-1][1]
    if not all(hasattr(snapshot, f'pending_p{i}') for i in (1, 2)):
        return None
    return tuple(int(getattr(snapshot, f'pending_p{i}_uncapped', None)
                     if getattr(snapshot, f'pending_p{i}_uncapped', None) is not None
                     else getattr(snapshot, f'pending_p{i}')) for i in (1, 2))


def evaluate(overlay: Any, exchange: search.Exchange, elapsed: float) -> float:
    """着弾後の盤面・未使用NEXT・残予告をG_feへ渡す。係数や較正は変えない。"""
    boards = tuple(search.sim._board(p.board) for p in exchange.sides)
    queues = np.array([(*p.queue[:4], *(0 for _ in range(max(0, 4-len(p.queue)))))
                       for p in exchange.sides], dtype=int)
    m0 = overlay._m0(np.stack([b._grid for b in boards]), queues)
    left, right = (p.pending for p in exchange.sides)
    snapshot = SimpleNamespace(**vars(overlay._snapshots[-1][1]))
    snapshot.pending_p1, snapshot.pending_p2 = left, right
    snapshot.pending_p1_uncapped, snapshot.pending_p2_uncapped = left, right
    snapshot.net_ojama_balance = right-left
    snapshot.net_balance_capped = min(right, ON_FIELD_CAP)-min(left, ON_FIELD_CAP)
    snapshot.forecast_p1, snapshot.forecast_p2 = left, right
    event = overlay._build_static(boards, snapshot, elapsed, m0)
    return float(evaluate_exchange_event(event, overlay.tracker.models))


def choose(states: tuple[search.Position, search.Position], attacker: int, elapsed: float,
           value_fn: ValueFunction, depth: int = search.SEARCH_DEPTH,
           k: int | None = search.TOP_K) -> tuple[float, search.Position, search.Position] | None:
    """攻撃側はmax/min、受け側は逆。待機も実際に置いた末端局面で競う。"""
    options = tuple(search.candidates(p, depth, k) for p in states)
    if not all(options):
        return None
    outer = max if attacker == 0 else min
    inner = min if attacker == 0 else max
    choices = []
    for attack in options[attacker]:
        replies = []
        for response in options[1-attacker]:
            pair = (attack, response) if attacker == 0 else (response, attack)
            time = elapsed + max(0, attack.consumed-1) * SEC_PER_HAND
            exchange = search.resolve(*pair, attacker, time)
            replies.append((value_fn(exchange, time), attack, response))
        choices.append(inner(replies, key=lambda item: item[0]))
    return outer(choices, key=lambda item: item[0])
