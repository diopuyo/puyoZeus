"""明示呼出し専用の逐次候補復元。初回v3.4・本番経路は変更しない。"""
from __future__ import annotations
from src import prefire_work_budget as work_budget

from math import log
from time import perf_counter
from typing import Any, Iterator

from src.exchange_count_prediction import COLS, CELLS, simulate, validate_times
from src.exchange_count_v34 import condition

MAX_EDITS = 2
SEARCH_SECONDS = .22
MAX_SIMULATIONS = 2400
MISSING_PENALTY = 100


def mismatch(value: dict, formulas: list[dict]) -> int:
    """届いた式だけで探索順を決める。未着の段数を目的にしない。"""
    error = 0
    for step in formulas:
        index = step['step']-1
        if index >= len(value['erased']):
            error += MISSING_PENALTY
        else:
            error += abs(value['erased'][index]-step['erased'])
            error += abs(value['bonuses'][index]-step['bonus'])
    return error


def neighbours(value: dict, colors: list[int], probabilities: list[dict],
               floor: float) -> Iterator[tuple]:
    """元候補から異なるマスだけを変更し、2マス以内の証跡を保つ。"""
    board = value['origin']
    used = {c['cell'] for c in value.get('changes', [])}
    choices = []
    for cell, before in enumerate(board):
        if cell in used:
            continue
        # 空の上空への追加は直下が埋まっている場合か隠し段に限定する。
        if not before and cell >= COLS and cell+COLS < CELLS and not board[cell+COLS]:
            continue
        for after in (0, 9, *colors):
            if before == after:
                continue
            delta = log(probabilities[cell].get(after, floor))-log(probabilities[cell].get(before, floor))
            choices.append((delta, cell, after))
    for delta, cell, after in sorted(choices, reverse=True):
        edited = board.copy()
        edited[cell] = after
        yield edited, value['log_weight']+delta, value.get('changes', [])+[
            dict(cell=cell, before=board[cell], after=after)]


def repair(values: list[dict], formulas: list[dict], stamp: float, colors: list[int],
           probabilities: list[dict], policy: dict, native: Any) -> dict:
    validate_times(stamp, formulas)
    began, tested, found, seen = perf_counter(), 0, [], set()
    seeds = sorted(values, key=lambda v: (mismatch(v, formulas), -v['log_weight']))[:policy['bases']]
    frontier = [dict(v, changes=[]) for v in seeds]
    stopped = False
    for depth in range(1, min(MAX_EDITS, policy['edits'])+1):
        next_frontier = []
        # 各近傍を交互に調べ、最初の種だけで予算を使い切らない。
        streams = [iter(neighbours(v, colors, probabilities, policy['prob_floor'])) for v in frontier]
        while streams and not stopped:
            live = []
            for stream in streams:
                item = next(stream, None)
                if item is None:
                    continue
                live.append(stream)
                board, weight, changes = item
                if tuple(board) in seen:
                    continue
                if tested >= MAX_SIMULATIONS or (not work_budget.enabled() and perf_counter()-began >= SEARCH_SECONDS):
                    stopped = True
                    break
                seen.add(tuple(board))
                value = dict(simulate(board, native), log_weight=weight, changes=changes)
                tested += 1
                error = mismatch(value, formulas)
                if not error:
                    found.append(value)
                next_frontier.append((error, -weight, tuple(board), value))
            streams = live
        if found or stopped:
            break
        frontier = [v[-1] for v in sorted(next_frontier, key=lambda v: v[:3])[:policy['repair_beam']]]
    return dict(candidates=sorted(found, key=lambda v: -v['log_weight']), tested=tested,
                budget_exhausted=stopped, evidence_times=[f['t'] for f in formulas])


def update(pool: list[dict], anchors: list[dict], formulas: list[dict], stamp: float,
           colors: list[int], probabilities: list[dict], policy: dict, native: Any) -> dict:
    accepted, conflict = condition(pool, formulas, stamp)
    if not conflict:
        return dict(candidates=accepted, tested=0, budget_exhausted=False,
                    evidence_times=[f['t'] for f in formulas])
    return repair(anchors, formulas, stamp, colors, probabilities, policy, native)
