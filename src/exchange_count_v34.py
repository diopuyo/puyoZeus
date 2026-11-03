"""明示呼出し専用。時系列票・盤の支持・隠し段の履歴で個数候補を復元する。"""
from __future__ import annotations

from collections import Counter
from math import log
from time import perf_counter
from typing import Any
from src.exchange_count_prediction import (
    CELLS, COLS, COUNT_COLORS, LOW_SPEC_RATIO, MS, beam_boards, simulate, validate_times,
)

PAIR_SIZE, HIDDEN = 2, 6
MAX_LOW_SPEC_MS = 500


def grounded(board: list[int]) -> list[int]:
    """接地していない操作中組・空中の演出を発火起点から外す。重力計算ではない。"""
    result = board.copy()
    for col in range(COLS):
        gap = False
        for cell in range(CELLS-COLS+col, -1, -COLS):
            gap = gap or board[cell] == 0
            if gap:
                result[cell] = 0
    return result


def probabilities(sources: list[dict], colors: list[int], history: dict, policy: dict) -> list[dict]:
    valid = (0, 9, *colors)
    images = [s for s in sources if s['kind'] == 'image']
    latest = max((s['t'] for s in images), default=None)
    images = [s for s in images if s['t'] == latest]
    cnn = next((s for s in sources if s['kind'] == 'cnn'), None)
    result = []
    for cell in range(CELLS):
        if cell < HIDDEN:
            result.append(hidden_distribution(cell, cnn, history, valid, policy['prob_floor']))
            continue
        votes: Counter = Counter()
        occupied = bool(cnn and cnn['board'][cell] != 0)
        for index, image in enumerate(images):
            value = grounded(image['board'])[cell]
            if value in valid:
                mass = policy['image_weight']/max(1, len(images))
                # 試合外色でも「何かがある」という観測は捨てず、色だけ不明とする。
                votes[value] += mass * (policy['empty_factor'] if occupied and value == 0 else 1.)
        if cnn and cnn['board'][cell] in valid:
            value = grounded(cnn['board'])[cell]
            votes[value] += 1-policy['image_weight']
        if not votes or not sum(votes.values()):
            votes.update(dict.fromkeys(valid, 1.))
        total = sum(votes.values())
        result.append({c: p/total for c, p in votes.items() if p > 0})
    return result


def hidden_distribution(cell: int, cnn: dict | None, history: dict,
                        valid: tuple, floor: float) -> dict:
    """画像票を使わず、同一試合で到着した隠し段の経過を確率で残す。"""
    counts = history.get('hidden_history', [{} for _ in range(HIDDEN)])[cell]
    votes = {c: float(counts.get(str(c), counts.get(c, 0))) for c in valid}
    if cnn and cnn['board'][cell] in valid:
        votes[cnn['board'][cell]] += 1.
    # 履歴に未観測の色を不可能と断定しない。事後の落下盤による穴埋めは禁止。
    total = sum(votes.values())
    return {c: (p+floor)/(total+floor*len(valid)) for c, p in votes.items()}


def history_boards(history: dict, pair: dict, colors: list[int], native: Any) -> list[list[int]]:
    result = []
    for item in history.get('history', []):
        board = item['board']
        if any(c not in (0, 9, *colors) for c in board):
            continue
        result.append(board)
        if pair.get('identified') and pair.get('pair'):
            result.extend(p[2] for p in native.enumerate_placements_py(board, *pair['pair'], False))
    return result


def delta_distance(board: list[int], history: dict, pair: dict) -> int:
    """既知組の2個追加と直前STABLEの保存がどれだけ一致するかを観測する。"""
    distances = []
    for item in history.get('history', []):
        before = item['board']
        changed = [i for i, (a, b) in enumerate(zip(before, board)) if a != b]
        damage = sum(before[i] != 0 for i in changed)
        added = [board[i] for i in changed if before[i] == 0 and board[i] != 0]
        pair_error = int(bool(pair.get('identified')) and sorted(added) != sorted(pair['pair']))
        distances.append(damage+min(len(added), abs(len(added)-PAIR_SIZE))+pair_error)
    return min(distances, default=0)


def candidates(sources: list[dict], legacy: list[dict], colors: list[int], policy: dict,
               stamp: float, native: Any, history: dict, pair: dict) -> dict:
    began = perf_counter()
    evidence = sources+history.get('history', [])
    if history.get('history_latest') is not None:
        evidence = evidence+[dict(t=history['history_latest'])]
    if pair.get('observed_sec') is not None:
        evidence = evidence+[dict(t=pair['observed_sec'])]
    validate_times(stamp, evidence)
    probs = probabilities(sources, colors, history, policy)
    boards = [b for _, b in beam_boards(probs, policy['beam_width'])]
    boards += [v['origin'] for v in legacy]+history_boards(history, pair, colors, native)
    boards += [s['board'] for s in sources if all(c in (0, 9, *colors) for c in s['board'])]
    unique = {tuple(grounded(b)) for b in boards}
    ranked = [(sum(log(probs[i].get(c, policy['prob_floor'])) for i, c in enumerate(b))
               -policy['delta_weight']*delta_distance(list(b), history, pair), b) for b in unique]
    result = []
    for weight, board in sorted(ranked, key=lambda v: (-v[0], v[1])):
        value = simulate(list(board), native)
        if value['erased']:
            result.append(dict(value, log_weight=weight))
    elapsed = (perf_counter()-began)*MS
    return dict(candidates=result, probabilities=probs, elapsed_ms=elapsed,
                low_spec_ms=elapsed*LOW_SPEC_RATIO, evidence_times=[s['t'] for s in evidence])


def condition(values: list[dict], formulas: list[dict], stamp: float) -> tuple[list[dict], bool]:
    validate_times(stamp, formulas)
    selected = [v for v in values if all(s['step'] <= len(v['erased']) and
        v['erased'][s['step']-1] == s['erased'] and v['bonuses'][s['step']-1] == s['bonus'] for s in formulas)]
    return selected, bool(formulas and not selected)
