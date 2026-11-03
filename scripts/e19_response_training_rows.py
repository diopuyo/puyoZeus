"""148原票の観測順序が識別できる応手机会を抽出する。"""
from __future__ import annotations

from collections import Counter
from pathlib import Path
import json
import math
import numpy as np
from scripts import train_exchange_event_models_v2_20260927 as data
from scripts.e15_training_rows_20260928 import chains_for_game
from src.board import COLOR_OJAMA, COLOR_UNKNOWN
from src.exchange_event_count_features import completion, NativeSimulator
from src.exchange_event_landing import remaining_hands, RESPONSE_BEAM_WIDTH
from src.indicators_v2 import NEAR_FUTURE_KNOWN_HAND_SLOTS, near_future_fire_power
from src.board import Board
from src.landing_counter_probability import response_features

OUT = Path("logs/e19/train")


def estimated_send(grid: np.ndarray, queue: np.ndarray, elapsed: float, hands: int) -> float:
    """E18のcalculate_chain_scoreと同じ厳密得点・探索幅をnativeでも維持する。"""
    board = Board.from_list(grid.tolist())
    level = hands-NEAR_FUTURE_KNOWN_HAND_SLOTS
    result = near_future_fire_power(board, tuple(queue[:2]), tuple(queue[2:]), elapsed_sec=elapsed,
        simulator=NativeSimulator(), k_levels=(level,), beam_width=RESPONSE_BEAM_WIDTH,
        resolve_before_death=True, use_exact_score=True)
    return float(result.values[level].raw)


def arrival_intervals(raw: dict, own: np.ndarray) -> list[tuple[float, float]]:
    """無タグ確定盤面のおじゃま増加を、前後の観測時刻で区間化する。"""
    stable = own[raw["chain_mechanism"][own] == ""]
    result = []
    for left, right in zip(stable, stable[1:]):
        before, after = raw["grids"][[left, right]]
        if np.any(before == COLOR_UNKNOWN) or np.any(after == COLOR_UNKNOWN):
            continue
        if np.count_nonzero(after == COLOR_OJAMA) > np.count_nonzero(before == COLOR_OJAMA):
            result.append((float(raw["t_sec"][left]), float(raw["t_sec"][right])))
    return result


def observed_label(stamp: float, arrivals: list[tuple[float, float]], replies: list[float],
                   next_attack: float, game_end: float) -> tuple[int | None, str, float, float]:
    """着弾区間内の発火・落下未観測・次撃後の落下を打ち切り扱いにする。"""
    interval = next(((lo, hi) for lo, hi in arrivals if hi > stamp), None)
    if interval is None:
        return None, "landing_unobserved", np.nan, np.nan
    lo, hi = interval
    if lo < stamp or hi >= min(next_attack, game_end):
        return None, "landing_unattributed", lo, hi
    reply = next((t for t in replies if t > stamp), np.inf)
    if reply <= lo:
        return 1, "fire_before_landing_interval", lo, hi
    if reply > hi:
        return 0, "no_fire_before_landing_interval", lo, hi
    return None, "fire_inside_landing_interval", lo, hi


def candidate(raw: dict, ids: np.ndarray, chain: object, chains: list) -> dict | None:
    """相手発火の初観測時に得られた確定盤面だけから説明変数を作る。"""
    if chain.pre < 0:
        return None
    receiver, stamp = 1-chain.side, chain.observed
    own = ids[(raw["side"][ids] == data.SIDES[receiver]) & (raw["t_sec"][ids] <= stamp)]
    stable = own[raw["chain_mechanism"][own] == ""]
    if not len(stable):
        return None
    idx, start = int(stable[-1]), float(raw["t_sec"][ids].min())
    elapsed = max(0., chain.trigger-start)
    incoming, count, _ = completion(raw["grids"][chain.pre].tobytes(), True, elapsed)
    incoming = math.floor(incoming)
    if incoming <= 0 or np.any(raw["grids"][idx] == COLOR_UNKNOWN):
        return None
    # 当該側が既に発火していれば、追加応手の母数へ混ぜず打ち切る。
    active = raw["chain_mechanism"][own[-1]] != ""
    if active:
        return None
    hands = remaining_hands(count, chain.trigger, stamp)
    grid, queue = raw["grids"][idx], np.array(data.queue_at(raw, idx))
    send = max(0, math.floor(estimated_send(grid, queue, elapsed, hands)))
    if send < incoming:
        return None
    features = response_features(grid, queue, send, incoming, hands, stamp-float(raw["t_sec"][idx]), False)
    own_ids = ids[raw["side"][ids] == data.SIDES[receiver]]
    replies = [c.trigger for c in chains if c.side == receiver]
    next_attack = min((c.trigger for c in chains if c.side == chain.side and c.trigger > stamp), default=np.inf)
    label, reason, lo, hi = observed_label(stamp, arrival_intervals(raw, own_ids), replies,
                                          next_attack, float(raw["t_sec"][ids].max()))
    return dict(t_sec=stamp, trigger=chain.trigger, receiver=data.SIDES[receiver], label=label,
        reason=reason, landing_lo=lo, landing_hi=hi, send=send, incoming=incoming, hands=hands,
        board_sec=float(raw["t_sec"][idx]), features=features.tolist())


def video_job(path_text: str) -> dict:
    """全148動画を走査し、候補と除外理由を動画単位で保存する。"""
    path = Path(path_text)
    dest = OUT/"videos"/(path.stem+".json")
    if dest.exists():
        return json.loads(dest.read_text())
    raw = data.read_npz(path)
    rows, counts = [], Counter()
    for game in np.unique(raw["game_idx"]):
        ids = np.flatnonzero(raw["game_idx"] == game)
        ids = ids[np.argsort(raw["t_sec"][ids], kind="stable")]
        chains = chains_for_game(raw, ids)
        counts["attacks"] += len(chains)
        for number, chain in enumerate(chains):
            row = candidate(raw, ids, chain, chains)
            if row is None:
                counts["not_eligible"] += 1
                continue
            counts[row["reason"]] += 1
            rows.append(dict(video_id="video_"+path.stem, game=int(game), exchange=number, **row))
    value = dict(video=path.stem, counts=dict(counts), rows=rows)
    data.v1.save_json(dest, value)
    print(path.stem, dict(counts), flush=True)
    return value
