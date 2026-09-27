"""E15: 未来の終点によらず、観測済みSTABLE更新ごとの行を作る。"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator

import numpy as np
import pandas as pd

from scripts import train_exchange_event_models_v2_20260927 as v2
from src.exchange_event_count_features import CountObservation, side_features, orient, completion
from src.exchange_event_features import arrival_features, prefire_side_features, score_features, fill_phase, D_COLUMNS

SIDES = ("1P", "2P")


@dataclass
class Chain:
    """発火後の得点代理だけを持ち、盤面の参照先は過去に限定する。"""
    side: int
    trigger: float
    observed: float
    pre: int
    last_tag: float
    score: float = np.nan
    final: bytes | None = None


def chains_for_game(raw: dict, ids: np.ndarray) -> list[Chain]:
    """終点得点は各連鎖単位で分離し、次の発火分を取り込まない。"""
    result = []
    for side, label in enumerate(SIDES):
        own = ids[raw["side"][ids] == label]
        tagged = own[np.isfinite(raw["chain_trigger_sec"][own]) & (raw["chain_mechanism"][own] != "")]
        triggers = np.unique(raw["chain_trigger_sec"][tagged])
        stable = own[raw["chain_mechanism"][own] == ""]
        for position, trigger in enumerate(triggers):
            tags = tagged[raw["chain_trigger_sec"][tagged] == trigger]
            before = stable[raw["t_sec"][stable] < trigger]
            pre = int(before[-1]) if len(before) else -1
            chain = Chain(side, float(trigger), float(raw["t_sec"][tags].min()), pre,
                          float(raw["t_sec"][tags].max()))
            next_trigger = triggers[position + 1] if position + 1 < len(triggers) else np.inf
            after = stable[(raw["t_sec"][stable] > chain.last_tag)
                           & (raw["t_sec"][stable] < next_trigger)]
            if pre >= 0 and len(after):
                left, right = raw["score"][[pre, after[0]]]
                if left >= 0 and right >= 0:
                    chain.score = float(max(0, right - left))
            result.append(chain)
    return sorted(result, key=lambda chain: chain.observed)


@dataclass
class Exchange:
    """1回の撃ち合いの発火時特徴と、時系列更新状態。"""
    number: int
    trigger: float
    pre: np.ndarray
    chains: list[Chain] = field(default_factory=list)
    last_tag: float = -np.inf
    base: np.ndarray | None = None
    arrival: np.ndarray | None = None
    source: int = 0


def samples(raw: dict, ids: np.ndarray) -> Iterator[tuple[Exchange, float, np.ndarray, list[Chain]]]:
    """タグ到着で開き、両側が無タグに戻るまでの各STABLE更新を列挙する。"""
    chains = chains_for_game(raw, ids)
    pending = iter(chains)
    next_chain = next(pending, None)
    latest, stable_times = np.full(2, -1, int), np.full(2, -np.inf)
    busy = [False, False]
    exchange, number = None, 0
    for stamp in np.unique(raw["t_sec"][ids]):
        batch = ids[raw["t_sec"][ids] == stamp]
        while next_chain is not None and next_chain.observed <= stamp:
            if exchange is None:
                number += 1
                before = ids[(raw["t_sec"][ids] < next_chain.trigger) & (raw["chain_mechanism"][ids] == "")]
                pre = [before[raw["side"][before] == side] for side in SIDES]
                exchange = Exchange(number, next_chain.trigger,
                                    np.array([p[-1] if len(p) else -1 for p in pre]))
            exchange.chains.append(next_chain)
            next_chain = next(pending, None)
        changed = False
        for i in batch:
            side = int(raw["side"][i] == "2P")
            busy[side] = raw["chain_mechanism"][i] != ""
            if raw["chain_mechanism"][i] != "":
                if exchange is not None:
                    exchange.last_tag = float(stamp)
                continue
            if latest[side] < 0 or not np.array_equal(raw["grids"][latest[side]], raw["grids"][i]) or any(
                    raw[key][latest[side]] != raw[key][i] for key in v2.QUEUE):
                changed = True
            latest[side], stable_times[side] = i, stamp
        if exchange is not None:
            active = [next(c for c in reversed(exchange.chains) if c.side == side)
                      for side in range(2) if busy[side] and any(c.side == side for c in exchange.chains)]
            if changed and np.all(latest >= 0):
                yield exchange, float(stamp), latest.copy(), active
            if not any(busy) and min(stable_times) > exchange.last_tag:
                exchange = None


def initial_features(exchange: Exchange, raw: dict, rows: pd.DataFrame,
                     design: np.ndarray, start: float) -> bool:
    """Dも発火前に観測済みの同試合行だけから選ぶ。終点採否を使わない。"""
    if exchange.base is not None:
        return True
    past = rows[rows.t_sec < exchange.trigger]
    if past.empty or (exchange.pre < 0).any():
        return False
    row = past.iloc[-1]
    exchange.source = int(row.sign < 0)
    exchange.base = np.nan_to_num(np.array(design[int(row.row_id)]), nan=0.)
    elapsed = max(0., exchange.trigger - start)
    sides = np.stack([prefire_side_features(raw["grids"][i], np.array(v2.queue_at(raw, i)), elapsed)
                      for i in exchange.pre])
    firing = tuple(any(c.side == side and c.trigger == exchange.trigger for c in exchange.chains)
                   for side in range(2))
    exchange.arrival = arrival_features(sides, firing, exchange.source)
    return True


def features(exchange: Exchange, stamp: float, latest: np.ndarray, active: list[Chain],
             raw: dict, start: float) -> tuple[np.ndarray, np.ndarray]:
    """既発火分の得点だけを加え、進行中の側は予測完走盤面を使う。"""
    grids = raw["grids"][latest].copy()
    for chain in active:
        if chain.pre < 0:
            raise ValueError("連鎖の発火前盤面が未取得")
        if chain.final is None:
            _, _, chain.final = completion(raw["grids"][chain.pre].tobytes(), True, chain.trigger-start)
        grids[chain.side] = np.frombuffer(chain.final, np.int8).reshape(grids.shape[1:])
    totals = np.array([sum(c.score for c in exchange.chains if c.side == side) for side in range(2)])
    elapsed = max(0., exchange.trigger-start)
    observation = CountObservation(grids, np.array([v2.queue_at(raw, i) for i in latest]),
                                   stamp-start, live=True, score_elapsed_sec=elapsed)
    counts = orient(side_features(observation, (False, False), np.zeros(2), totals), exchange.source)
    s1 = np.r_[exchange.base, exchange.arrival]
    s3 = np.r_[s1, score_features(np.zeros(2), totals, elapsed, exchange.source), counts]
    initial_counts = orient(side_features(observation, (False, False)), exchange.source)
    return np.r_[s1, initial_counts], s3


def game_rows(raw: dict, ids: np.ndarray, base_rows: pd.DataFrame,
              design: np.ndarray) -> Iterator[tuple[dict, np.ndarray, np.ndarray]]:
    """時刻・盤面参照時刻を行へ保存し、その時点の充填率で中盤を選ぶ。"""
    start = float(raw["t_sec"][ids].min())
    game = int(raw["game_idx"][ids[0]])
    for exchange, stamp, latest, active in samples(raw, ids):
        if not initial_features(exchange, raw, base_rows, design, start):
            continue
        own_total = np.count_nonzero(raw["grids"][latest[0]]) / v2.iv.BOARD_ROWS / v2.BOARD_COLS
        opp_total = np.count_nonzero(raw["grids"][latest[1]]) / v2.iv.BOARD_ROWS / v2.BOARD_COLS
        phase = fill_phase(own_total, own_total-opp_total)
        if phase != v2.v1.MIDDLE:
            continue
        label = float(raw["won"][latest[0]])
        if label not in (0., 1.):
            continue
        if raw["side"][latest[0]] != "1P":
            raise ValueError("左右順が不一致")
        first, third = features(exchange, stamp, latest, active, raw, start)
        info = dict(game=game, exchange=exchange.number, t_sec=stamp, phase=phase,
                    label=label, sign=1-2*exchange.source, trigger_sec=exchange.trigger,
                    board_sec_1p=float(raw["t_sec"][latest[0]]),
                    board_sec_2p=float(raw["t_sec"][latest[1]]), fired_chains=len(exchange.chains))
        yield info, first, third
