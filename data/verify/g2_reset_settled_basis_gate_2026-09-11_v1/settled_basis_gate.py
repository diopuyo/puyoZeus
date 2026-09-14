"""resetを跨いだ未認証落下手の完了後にだけ確率baseline候補を観測記録する小コア。

このコアは観測の記録と資格判定だけを行う。旧FIFO/current/Counterには触れず、
未認証手をlandもconsumeもしない。返す候補はあくまで基準候補であり、
quality/physical/accounting/production/current のいずれの権限も持たない。
入力observationの資格（原J同callであること等）は実caller adapterが保証する前提で、
このコア自体を本番権限とは呼ばない。偽の原J認証をここで自作しない。

原Recovery (recovery_connection.py) の「整数盤面にUNKNOWNがあれば拒否」は変更しない。
本コアは別経路であり、可視領域のUNKNOWNは拒否する一方、隠し行のUNKNOWNは
「真値が整数ではなく確率分布として保持されている」ことを要求して受理する。
UNKNOWNを0へ変換することはしない（変換された入力は拒否する）。
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
import math
from src.board import BOARD_ROWS, BOARD_COLS, HIDDEN_ROWS, COLOR_UNKNOWN
from src.probabilistic_board import PROB_COLORS

STRIDE = 2  # 原保存runの時計stride。新しい閾値ではなく既存観測間隔。
SUM_TOLERANCE = 1e-12
ACTUAL_CALL_STAGE = 'actual_J_complete_before_cleanup'
STATE_TSUMO_FALL = 'tsumo_fall'
STATE_STABLE = 'stable'

Grid = tuple[tuple[int, ...], ...]
Dist = tuple[tuple[int, float], ...]


def require(ok: bool, reason: str) -> None:
    if not ok: raise ValueError('settled_basis_gate:' + reason)


@dataclass(frozen=True)
class CallObservation:
    """実caller adapterが原Jの1 callから組み立てる観測。資格保証は呼出側の責務。"""
    call_token: str
    frame: int
    scope: tuple[Any, ...]
    epoch: int
    generation: int
    state: str
    exception: str | None
    match_active: bool
    effect_gate_window_active: bool | None
    origin_present: bool
    landing_grace_expired: bool
    observation_stage: str
    board_present: bool
    raw_grid: Grid | None = None
    cnn_grid: Grid | None = None
    sm_confirmed_grid: Grid | None = None
    returned_grid: Grid | None = None
    visible_probability: tuple[Dist, ...] | None = None
    hidden_probability: tuple[Dist, ...] | None = None


@dataclass(frozen=True)
class BasisCandidate:
    """新しい確率baselineの候補。初期frame/sourceは発行元の同callに一致する契約。"""
    scope: tuple[Any, ...]
    frame: int
    source_call_token: str
    source_observation_stage: str
    raw_grid: Grid
    hidden_probability: tuple[Dist, ...]
    entry_hidden_probability: tuple[Dist, ...]
    tsumo_fall_frames: tuple[int, ...]
    hidden_is_distribution: bool = field(default=True, init=False)
    quality_gate_clear: bool = field(default=False, init=False)
    physical_certified: bool = field(default=False, init=False)
    integer_current_permission: bool = field(default=False, init=False)
    accounting_permission: bool = field(default=False, init=False)
    production_permission: bool = field(default=False, init=False)
    current_permission: bool = field(default=False, init=False)
    display_update_permission: bool = field(default=False, init=False)
    both_sides_stable_confirmed: bool = field(default=False, init=False)
    consumed_next_token: None = field(default=None, init=False)


def check_scope(scope: tuple[Any, ...]) -> None:
    require(type(scope) is tuple and len(scope) == 7, 'scope_type')
    require(all(type(v) is str and bool(v) for v in scope[:2]), 'scope_head')
    require(all(type(v) is int and v >= 0 for v in scope[2:6]), 'scope_ints')
    require(type(scope[6]) is str and scope[6] in ('1P', '2P'), 'scope_side')


def check_grid(value: Any, name: str) -> Grid:
    require(type(value) is tuple and len(value) == BOARD_ROWS, name + '_rows')
    require(all(type(row) is tuple and len(row) == BOARD_COLS for row in value), name + '_cols')
    require(all(type(c) is int and c in (*PROB_COLORS, COLOR_UNKNOWN)
                for row in value for c in row), name + '_color')
    return value


def check_distributions(value: Any, count: int, name: str) -> tuple[Dist, ...]:
    require(type(value) is tuple and len(value) == count, name + '_length')
    for dist in value:
        require(type(dist) is tuple and bool(dist), name + '_empty')
        require(all(type(e) is tuple and len(e) == 2 and type(e[0]) is int and e[0] in PROB_COLORS
                    and type(e[1]) is float and math.isfinite(e[1]) and e[1] > 0 for e in dist), name + '_entry')
        require(len({e[0] for e in dist}) == len(dist), name + '_duplicate_color')
        require(abs(math.fsum(e[1] for e in dist) - 1.0) <= SUM_TOLERANCE, name + '_sum')
    return value


def visible_mismatch(raw: Grid, visible: tuple[Dist, ...]) -> str | None:
    """可視セルは実PBがpointmassで、raw整数と一致していなければならない。"""
    for index in range((BOARD_ROWS - HIDDEN_ROWS) * BOARD_COLS):
        color = raw[HIDDEN_ROWS + index // BOARD_COLS][index % BOARD_COLS]
        if color == COLOR_UNKNOWN: return 'visible_unknown'
        if visible[index] != ((color, 1.0),): return 'probability_nonpointmass'
    return None


def hidden_mismatch(raw: Grid, hidden: tuple[Dist, ...], entry: tuple[Dist, ...]) -> str | None:
    """隠しは分布のまま保たれること。UNKNOWNを0へ潰した入力をここで拒否する。"""
    for column in range(BOARD_COLS):
        color, dist = raw[0][column], hidden[column]
        if color == COLOR_UNKNOWN:
            if len(dist) < 2: return 'hidden_unknown_without_distribution'
        elif dist != ((color, 1.0),):
            return 'hidden_integer_probability_mismatch'
        if len(entry[column]) >= 2 and len(dist) < 2: return 'hidden_distribution_lost'
    return None


class SettledBasisGate:
    """reset直後scopeに束ねた観測専用の状態遷移。発行は最大1回。"""

    def __init__(self, scope: tuple[Any, ...], reset_frame: int, deadline: int) -> None:
        # deadlineは据置の絶対frame上限（最後に許される観測frame）。無条件に緩めない。
        check_scope(scope)
        require(type(reset_frame) is int and type(deadline) is int, 'clock_type')
        require(0 <= reset_frame < deadline, 'clock_order')
        self.scope, self.reset_frame, self.deadline = scope, reset_frame, deadline
        self.state = 'AWAIT_FALL'
        self.last_frame: int | None = None
        self.entry_hidden: tuple[Dist, ...] | None = None
        self.tsumo_fall_frames: tuple[int, ...] = ()
        self.seen_tokens: set[str] = set()
        self.candidate: BasisCandidate | None = None
        self.rows: list[dict[str, Any]] = []

    def observe(self, obs: CallObservation) -> BasisCandidate | None:
        """観測を1件記録して候補を返す。外部構造は一切変更しない。"""
        reason = self._admit(obs)
        if reason is None: reason = self._advance(obs)
        issued = reason is None
        if issued: self.candidate = self._issue(obs)
        self.rows.append(dict(kind='settled_basis_observation', frame=obs.frame, state=obs.state,
                              gate_state=self.state, call_token=obs.call_token, reason=reason,
                              issued=issued, mutated_fifo=False, mutated_current=False,
                              mutated_counter=False, consumed_next_token=None))
        return self.candidate if issued else None

    def _admit(self, obs: CallObservation) -> str | None:
        """同call資格・時計stride・scope不変・例外なしを見る。違反は復帰不能にする。"""
        require(type(obs) is CallObservation, 'observation_type')
        if self.state in ('ISSUED', 'BROKEN'):
            return 'candidate_already_issued' if self.state == 'ISSUED' else 'gate_broken'
        if obs.call_token in self.seen_tokens: return self._break('call_token_reused')
        self.seen_tokens.add(obs.call_token)
        if obs.observation_stage != ACTUAL_CALL_STAGE: return self._break('not_actual_J_call_stage')
        if obs.exception is not None: return self._break('exception_present')
        if obs.scope != self.scope: return self._break('scope_changed')
        if obs.epoch != self.scope[2] or obs.generation != self.scope[5]: return self._break('epoch_or_generation_changed')
        expected = self.reset_frame + STRIDE if self.last_frame is None else self.last_frame + STRIDE
        if obs.frame != expected: return self._break('stride_broken')
        self.last_frame = obs.frame
        if obs.frame > self.deadline: return self._break('deadline_exceeded')
        return None

    def _advance(self, obs: CallObservation) -> str | None:
        """reset後の原TSUMO_FALL観測→原STABLE完了だけを完了と認める。"""
        if not obs.board_present: return 'board_absent'
        if self.entry_hidden is None:
            self.entry_hidden = check_distributions(obs.hidden_probability, BOARD_COLS, 'entry_hidden')
        if obs.state == STATE_TSUMO_FALL:
            self.tsumo_fall_frames += (obs.frame,)
            self.state = 'AWAIT_SETTLE'
            return 'tsumo_fall_in_progress'
        if self.state == 'AWAIT_FALL': return 'awaiting_tsumo_fall_after_reset'
        if obs.state != STATE_STABLE: return self._break('fall_interrupted')
        return self._eligible(obs)

    def _eligible(self, obs: CallObservation) -> str | None:
        """原STABLE時の資格。raw=CNN=SM=返却、実PB可視pointmass、隠しは分布のまま。"""
        if not obs.landing_grace_expired: return 'landing_grace_pending'
        if not obs.match_active or obs.effect_gate_window_active is not False: return 'inactive_or_window'
        if obs.origin_present: return 'origin_present'
        raw = check_grid(obs.raw_grid, 'raw')
        if raw != check_grid(obs.cnn_grid, 'cnn') or raw != check_grid(obs.sm_confirmed_grid, 'sm'):
            return 'raw_SM_mismatch'
        if raw != check_grid(obs.returned_grid, 'returned'): return 'returned_mismatch'
        visible = check_distributions(obs.visible_probability, (BOARD_ROWS - HIDDEN_ROWS) * BOARD_COLS, 'visible')
        hidden = check_distributions(obs.hidden_probability, BOARD_COLS, 'hidden')
        assert self.entry_hidden is not None
        return visible_mismatch(raw, visible) or hidden_mismatch(raw, hidden, self.entry_hidden)

    def _break(self, reason: str) -> str:
        self.state = 'BROKEN'
        return reason

    def _issue(self, obs: CallObservation) -> BasisCandidate:
        """候補の初期frame/sourceは、この発行を起こした同callに一致する。"""
        require(self.candidate is None and self.state == 'AWAIT_SETTLE', 'reissue_rejected')
        assert obs.raw_grid is not None and obs.hidden_probability is not None and self.entry_hidden is not None
        value = BasisCandidate(self.scope, obs.frame, obs.call_token, obs.observation_stage,
                               obs.raw_grid, obs.hidden_probability, self.entry_hidden,
                               self.tsumo_fall_frames)
        require(value.frame == obs.frame and value.source_call_token == obs.call_token, 'source_call_contract')
        require(value.source_observation_stage == obs.observation_stage, 'source_stage_contract')
        self.state = 'ISSUED'
        return value
