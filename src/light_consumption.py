"""光による状態誤判定と独立にNEXTの消費と隠し段の不確かさを記録する。"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
import os
from typing import Any

ENV_FLAG = 'PUYO_LIGHT_ROBUST_CELLS'
STABLE_FRAMES = 3
STABLE_SECONDS = 0.06
LANDING_WAIT_SECONDS = 0.45
FORMULA_WINDOW_SECONDS = 1.5
VISIBLE_COLUMNS = 6
BOARD_CELLS = 78
OJAMA = 9
PAIR_SIZE = 2
FIRST_STEP_MIN_SECONDS = 0.3


def enabled() -> bool:
    """製品既定を変えず、明示された旗だけで有効にする。"""
    return os.environ.get(ENV_FLAG) == '1'


def guarded_forecast(forecast: Any, active: bool) -> Any:
    """基礎の予測履歴を変えず、隠れた流入がある1段目の公開だけを抑える。"""
    if not enabled() or not active or len(forecast.observed) != 1:
        return forecast
    if forecast.source == 'light_physics_lower' or len(forecast.steps) > len(forecast.observed):
        return forecast
    return replace(forecast, source='none', steps=forecast.observed, final_board=None)


def pair_value(value: Any) -> tuple[int, int] | None:
    if isinstance(value, dict):
        value = value.get('tuple')
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        return None
    if not all(isinstance(c, int) and 1 <= c <= 5 for c in value):
        return None
    return tuple(value)


def visible_board(value: Any) -> tuple[int, ...] | None:
    while isinstance(value, dict):
        value = value.get('board', value.get('array', value.get('grid')))
    if hasattr(value, '_grid'):
        value = value._grid.tolist()
    if value is None:
        return None
    try:
        cells = tuple(int(c) for row in value for c in row)
    except (ValueError, TypeError):
        return None
    return cells[VISIBLE_COLUMNS:] if len(cells) == BOARD_CELLS else None


def supported_ojama(before: tuple | None, after: tuple | None) -> int:
    """空き升から、下が埋まったおじゃまへ変わった升だけを数える。"""
    if before is None or after is None:
        return 0
    count = 0
    for index, (old, new) in enumerate(zip(before, after)):
        if old != 0 or new != OJAMA:
            continue
        below = after[index + VISIBLE_COLUMNS::VISIBLE_COLUMNS]
        count += all(c in (1, 2, 3, 4, 5, OJAMA) for c in below)
    return count


@dataclass
class ConsumptionSide:
    """色・列を推測せず、観測した組の変化だけを持つ。"""
    advances: int = 0
    accepted: tuple | None = None
    candidate: tuple | None = None
    candidate_frames: int = 0
    candidate_since: float = 0.0
    candidate_board: tuple | None = None
    stable_board: tuple | None = None
    pending: list[tuple[float, tuple | None]] = field(default_factory=list)
    hidden_unknown: bool = False
    top_columns: set[int] = field(default_factory=set)
    fire_t: float | None = None
    formula_seen: bool = False
    clear_reason: str = ''
    last_formula: float = float('-inf')
    last_t: float = float('-inf')

    def _next(self, t: float, value: Any) -> bool:
        pair = pair_value(value)
        if pair is None or pair != self.candidate:
            self.candidate, self.candidate_frames = pair, int(pair is not None)
            self.candidate_since, self.candidate_board = t, self.stable_board
            return False
        self.candidate_frames += 1
        if self.candidate_frames < STABLE_FRAMES or t-self.candidate_since < STABLE_SECONDS:
            return False
        if pair == self.accepted:
            return False
        changed = self.accepted is not None
        self.accepted = pair
        if changed:
            self.advances += 1
            self.pending.append((t + LANDING_WAIT_SECONDS, self.candidate_board))
        return changed

    def _settled(self, t: float, board: tuple | None, state: str) -> None:
        if state != 'STABLE' or board is None:
            return
        if self.top_columns and all(board[c] == 0 for c in self.top_columns):
            self._clear('top_empty')
        pending = []
        for due, before in self.pending:
            if t >= due and before is not None:
                changes = [(a, b) for a, b in zip(before, board) if a != b]
                placed = all(a == 0 and 1 <= b <= 5 for a, b in changes)
                columns = {c for c in range(VISIBLE_COLUMNS) if board[c] in (1, 2, 3, 4, 5)}
                if placed and len(changes) < PAIR_SIZE and columns:
                    self.hidden_unknown = True
                    self.top_columns.update(columns)
            else:
                pending.append((due, before))
        self.pending = pending
        self.stable_board = board

    def _clear(self, reason: str) -> None:
        self.hidden_unknown = False
        self.top_columns.clear()
        self.clear_reason = reason

    def _chain(self, t: float, raw: dict, state: str) -> bool:
        formula = raw.get('formula') or {}
        valid = bool(formula.get('valid')) and formula.get('product') is not None
        if valid:
            if not self.formula_seen:
                self.fire_t = t
            self.formula_seen = True
            self.pending.clear()
        if self.formula_seen and not valid and t-self.fire_t >= FIRST_STEP_MIN_SECONDS:
            self._clear('first_step_finished')
            self.formula_seen, self.fire_t = False, None
        return valid or state in ('CHAIN', 'GRAVITY_SETTLE')

    def observe(self, t: float, raw: dict, opponent_chain: bool) -> dict:
        if t <= self.last_t:
            return self.fields(False, False, 0)
        self.last_t = t
        self.clear_reason = ''
        state = (raw.get('state') or {}).get('state', '')
        board = visible_board(raw.get('confirmed_board'))
        changed = self._next(t, raw.get('tape_next_pair', raw.get('next_pair')))
        added = supported_ojama(self.stable_board, visible_board(raw.get('raw_cnn_board')))
        suspect = state == 'OJAMA_FALL' and opponent_chain and added == 0
        chain = self._chain(t, raw, state)
        if chain or added:
            self.pending.clear()
        if not chain:
            self._settled(t, board, state)
        return self.fields(changed, suspect, added)

    def fields(self, changed: bool, suspect: bool, added: int) -> dict:
        return dict(light_next_consumed=changed,
                    light_tape_advances=self.advances,
                    light_hidden_inflow_unknown=self.hidden_unknown,
                    light_hidden_top_columns=sorted(self.top_columns),
                    light_hidden_clear_reason=self.clear_reason,
                    light_ojama_suspect=suspect, light_supported_ojama=added)


class LightConsumptionTracker:
    """保存要求と生の認識の両方で使う、試合単位の記帳。"""
    def __init__(self) -> None:
        self.game: Any = None
        self.sides = [ConsumptionSide(), ConsumptionSide()]

    def observe(self, packet: dict) -> dict:
        game, t = packet['game'], float(packet['t'])
        if game != self.game:
            self.game, self.sides = game, [ConsumptionSide(), ConsumptionSide()]
        for state, raw in zip(self.sides, packet['raw']):
            formula = raw.get('formula') or {}
            if formula.get('valid'):
                state.last_formula = t
        result = []
        for index, raw in enumerate(packet['raw'][:2]):
            other = t-self.sides[1-index].last_formula < FORMULA_WINDOW_SECONDS
            fields = self.sides[index].observe(t, raw, other)
            result.append(dict(raw, **fields))
        return dict(packet, raw=result)


def annotate_sides(owner: Any, sides: list[dict], pairs: tuple, t: float, game: Any) -> list[dict]:
    """式を読む層で別台帳を作り、既存の観測列には書き戻さない。"""
    if not enabled():
        return sides
    if not hasattr(owner, '_light_consumption'):
        owner._light_consumption = LightConsumptionTracker()
    raw = [dict(s, state=dict(state=s['state']), tape_next_pair=pair) for s, pair in zip(sides, pairs)]
    packet = owner._light_consumption.observe(dict(t=t, game=game, raw=raw))
    return [dict(s, light_next_pair=pair_value(r.get('tape_next_pair')),
                 **{k: v for k, v in r.items() if k.startswith('light_')})
            for s, r in zip(sides, packet['raw'])]
