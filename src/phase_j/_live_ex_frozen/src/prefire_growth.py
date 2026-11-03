"""予測専用の片側確定盤と積み増し曲線。本番からは呼ばず、既定OFF。"""
from __future__ import annotations
from src.phase_j._live_ex_frozen.src import prefire_work_budget as work_budget

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, asdict
from time import perf_counter
from typing import Any
import numpy as np
from src.phase_j._live_ex_frozen.src.prefire_worker import LOW_SPEC_RATIO, MAX_LOW_SPEC_SEC, DEATH_CELL
from src.phase_j._live_ex_frozen.src.production_config import PREFIRE_V4_ENABLED_DEFAULT

ROLLOUT_COUNT = 4
CURVE_HORIZON = 24
VISIBLE_PAIRS = 3
DEPTH = 3
WIDTH = 1
COMPUTE_BUDGET_SEC = .05
MS = 1000.
PALETTE_SIZE = 4
CELL_COUNT = 78


@dataclass(frozen=True)
class SideInput:
    """一方が確定した瞬間の入力。教師や未来の発火側は保持しない。"""
    game: int
    side: int
    input_sec: float
    board: tuple[int, ...]
    known: tuple[tuple[int, int], ...]
    colors: tuple[int, ...]
    pending: int
    elapsed_sec: float
    evidence_times: tuple[float, ...]
    placements: int
    seed: int


def validate_side(request: SideInput) -> None:
    if not request.evidence_times or not np.isfinite(request.input_sec):
        raise ValueError('確定時刻の証跡なし')
    if any(not np.isfinite(t) or t > request.input_sec for t in request.evidence_times):
        raise ValueError('未来の入力')
    if request.side not in (0, 1) or request.game <= 0 or request.pending < 0:
        raise ValueError('側・試合・予告が不正')
    if len(set(request.colors)) != PALETTE_SIZE or any(c not in range(1, 6) for c in request.colors):
        raise ValueError('観測4色未確定')
    if len(request.board) != CELL_COUNT or any(c not in (0, 9, *request.colors) for c in request.board):
        raise ValueError('不明色または不正盤面')
    if request.board[DEATH_CELL] or len(request.known) < 2:
        raise ValueError('死亡盤または既知組不足')
    if len(request.known) > VISIBLE_PAIRS or any(len(p) != 2 or any(c not in request.colors for c in p)
                                               for p in request.known):
        raise ValueError('既知組の可視範囲・色矛盾')


def rollout_curve(task: tuple, native: Any) -> list:
    request, pairs, deadline = task
    return native.ama_growth_py(list(request.board), pairs, work_budget.CURVE_HANDS if work_budget.enabled() else CURVE_HORIZON,
        list(request.colors), WIDTH, DEPTH, budget_ms=None if work_budget.enabled() else max(0., deadline-perf_counter())*MS,
        visible_pairs=len(request.known))


def calculate_curve(request: SideInput, native: Any, pool: ThreadPoolExecutor,
                    enabled: bool = PREFIRE_V4_ENABLED_DEFAULT) -> dict | None:
    """完了した共通prefixだけを公開。未完の点を0にしない。"""
    if not enabled:
        return None
    began = perf_counter()
    validate_side(request)
    rng = np.random.default_rng(request.seed)
    tasks = []
    for _ in range(ROLLOUT_COUNT):
        unknown = rng.choice(request.colors, size=(CURVE_HORIZON+VISIBLE_PAIRS-len(request.known), 2))
        pairs = list(request.known)+[tuple(map(int, p)) for p in unknown]
        tasks.append((request, pairs, began+COMPUTE_BUDGET_SEC))
    samples = list(pool.map(lambda task: rollout_curve(task, native), tasks))
    length = min(map(len, samples))
    scores = np.array([[p[1] for p in sample[:length]] for sample in samples])
    chains = np.array([[p[2] for p in sample[:length]] for sample in samples])
    seconds = perf_counter()-began
    return dict(request=asdict(request), game=request.game, side=request.side,
        input_sec=request.input_sec, ready_sec=request.input_sec+seconds*LOW_SPEC_RATIO,
        elapsed_ms=seconds*MS, low_spec_ms=seconds*MS*LOW_SPEC_RATIO,
        over_budget=False if work_budget.enabled() else seconds*LOW_SPEC_RATIO > MAX_LOW_SPEC_SEC,
        scores=scores.mean(axis=0).tolist() if length else [],
        chains=chains.mean(axis=0).tolist() if length else [],
        sample_scores=scores.tolist(), computed_k=length-1,
        exhausted=length < (work_budget.CURVE_HANDS if work_budget.enabled() else CURVE_HORIZON)+1)


class SideSnapshots:
    """片側ごとの最新確定盤。既存の両者STABLE層とは共有しない。"""
    def __init__(self) -> None:
        self.game: int | None = None
        self.boards: list[tuple | None] = [None, None]
        self.stable = [False, False]

    def observe(self, game: int, sides: list[dict]) -> list[int]:
        if self.game != game:
            self.game, self.boards, self.stable = game, [None, None], [False, False]
        changed = []
        for side, raw in enumerate(sides):
            stable = raw['state'] == 'STABLE' and raw['board'] is not None
            board = tuple(np.asarray(raw['board']).reshape(-1).astype(int)) if stable else None
            if stable and (board != self.boards[side] or not self.stable[side]):
                self.boards[side] = board
                changed.append(side)
            self.stable[side] = stable
        return changed
