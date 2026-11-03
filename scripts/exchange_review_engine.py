"""追補4のF/quick_counterとad4を明示的に呼ぶレビュー専用アダプター。"""
from __future__ import annotations

import time
from typing import Any

from scripts.exchange_review_io import grid, consumption
from scripts.exchange_review_seed import extend_tapes
from src.board import COLOR_EMPTY, COLOR_OJAMA, COLOR_UNKNOWN

WIDTH, DEPTH, LEVELS = 4, 6, 3
HORIZON, QUICK_HANDS = 12, 2
TARGET, TARGETS, DEADLINES = 12600, [4200, 12600], [6, 12]
BUDGET_MS, FOREGROUND_LIMIT_MS, THREADS = 2500, 500, 4
RATE = 70
MATCH_COLOR_COUNT = 4
HAND_MS = 733
CHAIN_MS = [0, 2633, 3867, 5325, 6667, 8100, 9500, 10725, 11967, 13550, 11458, 13892, 14083]
CHAIN_MS += [round((3.3 + .89 * n) * 1000) for n in range(13, 20)]


def leaf_value(node: dict) -> int | None:
    """追補4と同じminimax。打切りの暫定値を完全探索と区別して保持する。"""
    if node.get('unavailable') or node['outcome'].get('truncated') or not node['outcome']['coverage']:
        return None
    children = [v for child in node['children'] if (v := leaf_value(child)) is not None]
    if children:
        return min(children) if node['side'] == 0 else max(children)
    return node['balance']


def choose(tree: dict) -> dict:
    ranks = {'main': 0, 'counter': 1, 'quick_counter': 2}
    valid = [(n, leaf_value(n)) for n in tree['own']]
    valid = [(n, v) for n, v in valid if v is not None]
    if not valid:
        return dict(candidate=None, policy=None, balance=None, truncated=True)
    node, value = min(valid, key=lambda p: (-p[1], ranks[p[0]['policy']],
                                           p[0]['deadline_hands'], p[0]['target_score']))
    return dict(candidate=[node['policy'], node['deadline_hands'], node['target_score']],
        policy=node['policy'], balance=value,
        truncated=bool(tree.get('budget_exhausted') or node['truncated']),
        preparation=node.get('preparation'))


def simulate_attack(data: dict, native: Any) -> tuple[list[int], float | None]:
    before = grid(data['attacker_board'])
    result = native.simulate_chain_py(before, True)
    after = list(result.final_grid)
    return after, consumption(before, after) if result.chain_count else None


def fast(row: dict, seed: dict, native: Any, min_hands: int = 1) -> dict:
    began = time.perf_counter()
    data = row['exchange_input']
    side = 1 - int(data['attacker'])
    known = seed['tape'][seed['indices'][side]:] if seed.get('indices') else data['known_receiver']
    pairs, board = [tuple(p) for p in known], grid(data['receiver_board'])
    k = min(HORIZON, max(min_hands, int(row['hands_exp'])))
    incoming = int(row['incoming'])
    opponent, ratio = simulate_attack(data, native)
    f = native.ama_f_py(board, pairs, k, WIDTH, DEPTH, data['colors'],
                       policy='counter', incoming=incoming)
    q = native.ama_f_py(board, pairs, k, WIDTH, DEPTH, data['colors'],
        policy='quick_counter', incoming=incoming, target_score=TARGET,
        opponent_consumption=ratio, quick_visible=QUICK_HANDS)
    # 本線仮定のquick_counterは探索を省くため、2手量の観測値を別に測る。
    probe = native.ama_f_py(board, pairs, QUICK_HANDS, WIDTH, DEPTH, data['colors'],
        policy='quick_counter', incoming=incoming, target_score=TARGET,
        opponent_consumption=0., quick_visible=QUICK_HANDS)
    elapsed = (time.perf_counter() - began) * 1000
    return dict(f=f, quick=q, compute_ms=elapsed, over_500ms=elapsed > FOREGROUND_LIMIT_MS,
                k=k, incoming=incoming, opponent_consumption=ratio, opponent_board=opponent,
                two_hand_return=probe['quick_max_return'] if probe['quick_complete'] else None)


def recursive(row: dict, seed: dict, foreground: dict, native: Any) -> dict:
    if not seed.get('indices'):
        return dict(status='種/手番未特定', compute_ms=0, candidate=None, truncated=True)
    data = row['exchange_input']
    if any(c not in data['colors'] for pair in seed['tape'] for c in pair):
        return dict(status='特定種と観測4色が不一致', compute_ms=0, candidate=None, truncated=True)
    attack, receiver = int(data['attacker']), 1 - int(data['attacker'])
    began = time.perf_counter()
    tree = native.ama_exchange_py(grid(data['receiver_board']), foreground['opponent_board'],
        [tuple(p) for p in seed['tape']], (seed['indices'][receiver], seed['indices'][attack]),
        foreground['k'], WIDTH, DEPTH, data['colors'], LEVELS, HAND_MS, CHAIN_MS, TARGETS, DEADLINES,
        incoming=foreground['incoming'], both=False, budget_ms=BUDGET_MS,
        parallel=True, num_threads=THREADS, advance_opponent=True,
        opponent_busy_ms=max(0, round(data['remaining_sec'] * 1000)))
    elapsed = (time.perf_counter() - began) * 1000
    return dict(status='計算済み', compute_ms=elapsed, **choose(tree), tree=tree)


def compute_event(row: dict, prefix: dict, native: Any) -> dict:
    """事後の実測値を引数に受け取らない。未来の結果を予測へ混ぜる口を持たない。"""
    stamp = float(row['t_sec'])
    result = dict(event_id=row['event_id'], game_idx=row['game_idx'], input_sec=stamp,
        trigger_sec=row['trigger_sec'], receiver=row['receiver'], attacker=row['attacker'],
        episode=row['evidence']['chain_episode_id'], status='欠測', seed=prefix)
    data = row.get('exchange_input')
    if data is None:
        return result | dict(reason='保存exchange_inputなし')
    if data.get('receiver_busy_sec', 0) > 0:
        return result | dict(reason='受け側連鎖中: 初期相殺/完走の二重計上を避け欠測')
    if any(COLOR_UNKNOWN in grid(data[k]) for k in ('attacker_board', 'receiver_board')):
        return result | dict(reason='UNKNOWN盤面')
    if len(data['colors']) != MATCH_COLOR_COUNT:
        return result | dict(reason='4色未確定')
    allowed = {COLOR_EMPTY, COLOR_OJAMA, *data['colors']}
    if any(c not in allowed for key in ('attacker_board', 'receiver_board') for c in grid(data[key])):
        return result | dict(reason='盤面に試合4色以外の色あり')
    seed = extend_tapes(prefix, row)
    if seed.get('indices') and any(c not in data['colors'] for pair in seed['tape'] for c in pair):
        return result | dict(reason='特定種と観測4色が不一致')
    foreground = fast(row, seed, native)
    background = recursive(row, seed, foreground, native)
    return result | dict(status='計算済み', seed=seed, foreground=foreground, background=background,
        fast_ready_sec=stamp + foreground['compute_ms'] / 1000,
        background_ready_sec=stamp + background['compute_ms'] / 1000,
        rate_note='方式F/再帰は追補4と同じ70点固定', exchange_input=data)
