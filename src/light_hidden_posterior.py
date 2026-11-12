"""既存の隠し段信念を凍結し、可視尤度と合わせる外側の包み。"""
from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from math import prod
from itertools import product
from types import SimpleNamespace
from typing import Any

import numpy as np

from src.board import Board
from src.board_state_machine import BoardState
from src.hidden_row_belief import HiddenRowBelief, prior
from src.light_consumption import pair_value

SIDES = 2
MAX_CANDIDATES = 3
SELECTED_MASS = .95


class HiddenPrior:
    """推定器は HiddenRowPrefire と同じ HiddenRowBelief。発火後は事前を変えない。"""

    def __init__(self, simulator: Any) -> None:
        self.belief = HiddenRowBelief(simulator)
        self.frozen: Any = None
        self.audit: list[dict] = []
        self.cells: dict = {}
        self.uniform_cells: dict = {}

    def observe(self, t: float, side: dict, colors: tuple) -> None:
        state = side.get('state', {})
        state = state.get('state') if isinstance(state, dict) else state
        from src.fire_board import clean
        board = clean(side.get('confirmed_board'))
        stable = state == 'STABLE'
        view = SimpleNamespace(state=BoardState.STABLE if stable else BoardState.CHAIN,
            confirmed_board=Board.from_list(board.tolist()) if stable and board is not None else None,
            midchain_board=None,
            next_pair=pair_value(side.get('tape_next_pair', side.get('next_pair'))),
            dnext_pair=pair_value(side.get('tape_dnext_pair', side.get('dnext_pair'))))
        self.belief.observe(view, colors, t)

    def freeze(self) -> None:
        self.frozen = deepcopy(self.belief)
        self.cells = {}
        self.uniform_cells = {}

    def weights(self, before: np.ndarray, hidden: tuple, colors: tuple) -> tuple[float, float]:
        # 発火前までの履歴だけで、異なる可視候補には既存の snapshot の規則を使う。
        key = before.tobytes()
        if key not in self.cells:
            belief = deepcopy(self.frozen)
            belief.advance(before, colors, belief.stamp or 0., belief.pair)
            self.cells[key] = belief.snapshot(before, colors)
        cells = self.cells[key]
        uniform_key = (key, colors)
        if uniform_key not in self.uniform_cells:
            self.uniform_cells[uniform_key] = [prior(before, c, colors) for c in range(before.shape[1])]
        uniform = self.uniform_cells[uniform_key]
        return (prod(cell.probs.get(color, 0.) for cell, color in zip(cells, hidden)),
                prod(cell.probs.get(color, 0.) for cell, color in zip(uniform, hidden)))


def assign_priors(physical: Any, origins: dict) -> None:
    """可視の設置候補は等確率。重複生成の回数を隠し段の確率に足さない。"""
    before_keys = {key for values in origins.values() for key, _ in values}
    denominator = len(before_keys)
    for candidate in physical.candidates:
        values = origins[candidate.board.tobytes()]
        records = []
        for (raw, _), hidden in values:
            before = np.frombuffer(raw, dtype=np.int8).reshape(candidate.board.shape)
            weight, uniform = physical.hidden_prior.weights(before, hidden, physical.palette[1:])
            records.append(dict(hidden=hidden, weight=weight/denominator,
                                uniform=uniform/denominator))
        candidate.hidden_origins = tuple(records)
        candidate.prior = sum(row['weight'] for row in records)


def posterior(physical: Any) -> list[dict]:
    evidence = physical.evidence
    if evidence is None or not len(evidence.retained) or physical.rejected:
        return []
    ids = [int(i) for i in evidence.retained if physical.candidates[int(i)].prior > 0]
    if not ids:
        return []
    logs = np.array([np.log(physical.candidates[i].prior)+evidence.scores[i] for i in ids])
    weights = np.exp(logs-logs.max())
    weights /= weights.sum()
    return [dict(index=i, weight=float(weight), candidate=physical.candidates[i])
            for i, weight in zip(ids, weights)]


def total_groups(physical: Any) -> list[dict]:
    grouped: dict[int, list] = defaultdict(list)
    for row in posterior(physical):
        grouped[sum(row['candidate'].steps)].append(row)
    result = []
    for total, members in grouped.items():
        # 同点内の段列・終端は一貫した最大確率候補を使い、重みは全員分を合算する。
        selected = max(members, key=lambda row: (row['weight'], -row['index']))
        candidate = selected['candidate']
        result.append(dict(total=total, weight=sum(row['weight'] for row in members),
            steps=candidate.steps, final=candidate.final.tolist(), chain_id=physical.chain_id,
            observed=physical.observed, members=len(members)))
    return sorted(result, key=lambda row: (-row['weight'], row['total']))


def scenarios(owners: list, limit: int = MAX_CANDIDATES) -> list[dict]:
    choices = []
    for owner in owners:
        physical = getattr(owner, '_light_physics', None)
        groups = total_groups(physical) if physical is not None and physical.active else []
        choices.append(groups or [None])
    if not any(len(group) > 1 for group in choices):
        return []
    ranked = sorted((dict(sides=pair, weight=prod(row['weight'] for row in pair if row))
                     for pair in product(*choices)), key=lambda row: -row['weight'])
    selected, mass = [], 0.
    for row in ranked[:limit]:
        selected.append(row)
        mass += row['weight']
        if mass >= SELECTED_MASS:
            break
    for row in selected:
        row.update(weight=row['weight']/mass, retained_mass=mass)
    return selected
