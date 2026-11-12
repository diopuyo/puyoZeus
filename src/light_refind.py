"""既知の盤を物理で進め、見えるようになった未知の升だけを観測で選ぶ。"""
from __future__ import annotations
from collections import Counter, deque
from dataclasses import dataclass, replace
import itertools
from typing import Any, Callable, Iterator
import numpy as np
from src.board import Board
from src.chain import ChainSimulator
from src.fire_board import clean, step_scores
from src.light_consumption import enabled, pair_value
from src.light_likelihood import CandidateEvidence
from src.light_base_history import ObservedBases

ROWS, COLS = 13, 6
COLOR_COUNT, HISTORY_SECONDS = 4, 4.
MAX_ORIGIN_COMBINATIONS = 8192
AfterSpec = tuple[np.ndarray, list[tuple[int, int]], tuple[bytes, tuple[int, ...]]]
MIN_FRAMES, MIN_WINDOW_SECONDS, MIN_POP_SECONDS = 3, .06, .3
LOWER_BOUND_SOURCE = 'light_physics_lower'
# この候補経路だけ、消去群・順序が同一の既存の高速探索を使う。
SIMULATOR = ChainSimulator(cache_enabled=False, exclude_hidden_row_from_pop=True, fast_groups=True)


def placements(base: np.ndarray, pair: tuple[int, int]) -> Iterator[np.ndarray]:
    for first in range(COLS):
        for second in (first, first-1, first+1):
            if not 0 <= second < COLS:
                continue
            for order in (pair, pair[::-1]):
                board = base.copy()
                for column, color in zip((first, second), order):
                    occupied = np.flatnonzero(board[:, column])
                    row = (int(occupied[0]) if len(occupied) else ROWS)-1
                    if row < 0:
                        break
                    board[row, column] = color
                else:
                    yield board


def after_first(board: np.ndarray, product: int, hidden: set[int]) -> tuple[np.ndarray, list] | None:
    value = Board.from_list(board.tolist())
    groups = SIMULATOR.find_erasable_groups(value)
    description = [(int(group.color), group.size) for group in groups]
    if not description or step_scores([description])[0] != product:
        return None
    SIMULATOR._erase_fast(value, groups)
    SIMULATOR._gravity_fast(value)
    after, slots = value._grid.copy(), []
    for column in range(COLS):
        if column in hidden:
            occupied = np.flatnonzero(after[:, column])
            row = (int(occupied[0]) if len(occupied) else ROWS)-1
            if row >= 0:
                # まだ落ちていない隠し段も未知のまま候補に残す。
                slots.append((row, column))
    return after, slots


def after_specs(base: np.ndarray, pairs: set, product: int) -> Iterator[AfterSpec]:
    """色の直積を展開する前に、最初の式に一致する盤と未知升を求める。"""
    before_seen = set()
    options = itertools.chain((base,), *(placements(base, pair) for pair in sorted(pairs)))
    for before in options:
        hidden = {column for column in range(COLS) if base[1, column] or before[0, column]}
        before = before.copy()
        before[0] = 0  # 隠し段へ入る発火組も、色はNEXTから決めず未知に戻す。
        key = (before.tobytes(), tuple(sorted(hidden)))
        if key in before_seen:
            continue
        before_seen.add(key)
        physical = after_first(before, product, hidden)
        if physical is None:
            continue
        after, slots = physical
        source = (before.astype(np.int8).tobytes(), tuple(sorted(hidden)))
        yield after, slots, source


def after_boards(base: np.ndarray, pairs: set, palette: tuple, product: int,
                 origins: dict | None = None, specs: list[AfterSpec] | None = None) -> Iterator[np.ndarray]:
    after_seen = set()
    for after, slots, source in after_specs(base, pairs, product) if specs is None else specs:
        for colors in itertools.product(palette, repeat=len(slots)):
            candidate = after.copy()
            hidden_colors = [0]*COLS
            for (row, column), color in zip(slots, colors):
                candidate[row, column] = color
                hidden_colors[column] = color
            key = candidate.tobytes()
            if origins is not None:
                origins.setdefault(key, set()).add((source, tuple(hidden_colors)))
            if key not in after_seen:
                after_seen.add(key)
                yield candidate


@dataclass
class Candidate:
    board: np.ndarray
    steps: tuple[int, ...]
    final: np.ndarray
    prior: float = 1.
    hidden_origins: tuple = ()


class PhysicalRefind:
    def __init__(self, trace: Callable) -> None:
        self.trace = trace
        self.base: np.ndarray | None = None
        self.base_t: float | None = None
        self.pairs: deque = deque()
        self.colors: Counter = Counter()
        self.next_seen: tuple | None = None
        self.next_frames = 0
        self.chain_id: float | None = None
        self.candidates: list[Candidate] = []
        self.matrix = np.empty((0, ROWS, COLS), np.int8)
        self.votes = np.empty(0, np.int32)
        self.frames = 0
        self.window_t: float | None = None
        self.best: Candidate | None = None
        self.rejected = False
        self.active = False
        self.palette: tuple = ()
        self.evidence: CandidateEvidence | None = None
        self.base_history = ObservedBases()
        self.base_versions: list[np.ndarray] = []
        from src.light_hidden_posterior import HiddenPrior
        self.hidden_prior = HiddenPrior(SIMULATOR)
        self.observed: tuple = ()
        self.work_items = 0
        self.work_limited = False
        self.work_limit_hits = 0

    def remember(self, t: float, side: dict) -> None:
        pair = pair_value(side.get('light_next_pair', side.get('tape_next_pair')))
        self.next_frames = self.next_frames+1 if pair == self.next_seen else 1
        self.next_seen = pair
        if pair is not None and self.next_frames == MIN_FRAMES:
            self.pairs.append((t, pair))
            self.colors.update(pair)
        while self.pairs and self.pairs[0][0] < t-HISTORY_SECONDS:
            self.pairs.popleft()
        state = side.get('state', {})
        state = state.get('state') if isinstance(state, dict) else state
        if state != 'STABLE':
            return
        board = clean(side.get('confirmed_board'))
        if board is None:
            return
        board[0] = 0
        self.base, self.base_t = board, t
        self.base_history.remember(t, clean(side.get('raw_cnn_board')))
        self.colors.update(int(c) for c in board.flat if 1 <= c <= 5)

    def start(self, chain_id: float, product: int, active: bool) -> None:
        self.chain_id, self.active = chain_id, active
        self.candidates, self.best, self.rejected = [], None, False
        self.evidence = None
        self.base_versions = []
        self.frames, self.window_t = 0, None
        self.work_items, self.work_limited = 0, False
        self.palette = (0, *sorted(c for c, _ in self.colors.most_common(COLOR_COUNT)))
        if self.base is None or len(self.palette) != COLOR_COUNT+1:
            return
        self.base_versions = self.base_history.variants(chain_id,self.base)
        # 上端が埋まる盤では、STABLEへの入り方に依存する流入旗だけで候補を止めない。
        self.active = active or any(np.any(base[1]) for base in self.base_versions)
        if not self.active:
            return
        pairs = {pair for _, pair in self.pairs}
        self.hidden_prior.freeze()
        origins: dict = {}
        self.collect_candidates(pairs, product, origins)
        if not self.candidates and not self.work_limited:
            # 最初の式との物理的矛盾がある時だけ、繰り返し観測した別色も試す。
            self.base_versions = self.base_history.variants(
                chain_id, self.base, allow_color_conflicts=True)
            self.collect_candidates(pairs, product, origins)
        if self.candidates:
            from src.light_hidden_posterior import assign_priors
            assign_priors(self, origins)
            self.matrix = np.stack([candidate.board for candidate in self.candidates])
            self.votes = np.zeros(len(self.candidates), np.int32)
            self.evidence = CandidateEvidence(self.candidates)

    def collect_candidates(self, pairs: set, product: int, origins: dict) -> None:
        """全ての盤に同じ物理条件を適用し、可視尤度で絞る前の集合を作る。"""
        specs = {}
        for base in self.base_versions:
            for spec in after_specs(base, pairs, product):
                specs.setdefault(spec[2], spec)
        self.work_items = sum(len(self.palette)**len(spec[1]) for spec in specs.values())
        if self.work_items > MAX_ORIGIN_COMBINATIONS:
            # 一部だけを選ぶと確率が変わるため、この集合は丸ごと基礎経路へ戻す。
            self.work_limited = True
            return
        for board in after_boards(self.base, pairs, self.palette, product, origins, list(specs.values())):
            groups, final = self.trace(board)
            self.candidates.append(Candidate(board, (product, *step_scores(groups, 1)), final))
    def observe(self, t: float, side: dict, forecast: Any, active: bool) -> Candidate | None:
        formula = side.get('formula') or {}
        valid = bool(formula.get('valid')) and formula.get('product') is not None
        if forecast is not None and forecast.chain_id != self.chain_id:
            self.start(forecast.chain_id, forecast.observed[0], active)
        if not valid and (forecast is None or not self.active):
            self.remember(t, side)
        colors = tuple(sorted(c for c, _ in self.colors.most_common(COLOR_COUNT)))
        self.hidden_prior.observe(t, side, colors)
        if self.work_limited and forecast is not None and forecast.chain_id == self.chain_id:
            # 発火時の入力集合は固定。同じ集合で上限に当たる更新も毎回数える。
            self.work_limit_hits += 1
        if self.rejected or forecast is None:
            return None
        observed = tuple(forecast.observed)
        self.observed = observed
        if self.evidence is None:
            return None
        if not any(c.steps[:len(observed)] == observed for c in self.candidates):
            self.rejected = True
            self.evidence = None
            return None
        probs = side.get('light_cell_probs') if not valid and t-self.chain_id >= MIN_POP_SECONDS else None
        self.evidence.update(t, observed, probs)
        self.frames, self.window_t = self.evidence.frames, self.evidence.first_t
        return None  # 確実でない一つの盤は選ばない。


def update_physical_refind(owner: Any, t: float, raw: Any, side: dict | None) -> None:
    """旗OFFでは確保も更新もせず、基礎の予測履歴は書き換えない。"""
    if not enabled():
        return
    if getattr(owner, '_light_physics', None) is None:
        owner._light_physics = PhysicalRefind(owner.trace)
    observation = dict(side or {}, raw_cnn_board=raw)
    owner._light_physics.observe(t, observation, owner.current(t),
                                bool(getattr(owner, '_light_hidden_fire', False)))
    if owner._light_physics.work_limited:
        # B3と凍結EXが共用する公開時の抑制も、この更新はOFFの経路へ戻す。
        owner._light_hidden_fire = False


def physical_forecast(owner: Any, forecast: Any) -> Any:
    """否定できない候補の最小得点だけを公開し、既存refindを下げない。"""
    physical = getattr(owner, '_light_physics', None)
    if not enabled() or physical is None or physical.rejected or not physical.active:
        return forecast
    evidence = physical.evidence
    if (evidence is None or evidence.steps is None or forecast.chain_id != physical.chain_id
            or evidence.steps[:len(forecast.observed)] != forecast.observed):
        return forecast
    if forecast.has_prediction and forecast.total >= sum(evidence.steps):
        return forecast
    fields = dict(steps=evidence.steps, final_board=evidence.final, source=LOWER_BOUND_SOURCE, lower_bound=True)
    if hasattr(forecast, 'combos'):
        fields.update(combos=(), exp_total=None, upside=None, n_comb=0, comb_mass=None, comb_ms=None)
    return replace(forecast, **fields)
