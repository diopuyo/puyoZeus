"""v4.9e: 連続値 (b) の出力不変の高速化。候補盤のB3評価を1更新1回にまとめ、候補選びを再利用する。

v4.9d の `continuous_values` と同じ値を返す。違いは (1) 必要な候補盤を先に全部決め、B3をまとめて評価する、
(2) 応手列挙を `prefire_responses_v49e` (出力不変の高速版、任意で作業量の打切り) に差し替える、
(3) 同じ応手集合・予告・繰越・換算レートの代表選び (`pick_candidates`) を更新をまたいで再利用する。
証明 (a) と保持は v4.9d のまま継承する。
"""
from __future__ import annotations

from typing import Any, Callable
from src.board import Board, BOARD_COLS, COLOR_EMPTY, COLOR_UNKNOWN, DEATH_COL, DEATH_ROW
from src.chain import OJAMA_ROW_SIZE
from src.exchange_event_landing import remaining_hands
from src.prefire_landing_v49d import (AGGREGATORS, AGG_BASE, AGG_REP, AGG_MEAN, ContinuousInput, LandingV49D,
                                      StageClock, grid_board, margin_elapsed, first_alive, strata_bounds, SIDES)
from src.prefire_landing_v49d import candidate
from src.prefire_responses_v49d import (ARRANGEMENT_SAMPLES, RESPONSE_MAX_HANDS, RESPONSE_NODE_LIMIT, arrangement_columns, cancel_amount,
                                        landed_boards, landing_count, feasible_responses as feasible_responses_v49d)
from src.prefire_responses_v49e import Replies, ranked_responses, compact
from src.scoring import compute_effective_rate

Many = Callable[[list[tuple[list, list]]], list[float]]
PICK_MEMO_LIMIT = 2048
_PICK_MEMO: dict[tuple, tuple] = {}


class BatchMemo:
    """同じ (盤面, 予告) のB3評価を1度だけ行い、未評価の候補を1回にまとめて評価する。"""

    def __init__(self, many: Many, boards: list[list[int]], pending: list[float], receiver: int) -> None:
        self.many, self.boards, self.pending, self.receiver = many, boards, pending, receiver
        self.values: dict[tuple, float] = {}
        self.evaluated = 0

    @staticmethod
    def key(board: Board, left: int) -> tuple:
        return (board._grid.tobytes(), int(left))

    def request(self, board: Board, left: int) -> tuple[list, list]:
        boards = [list(b) for b in self.boards]
        boards[self.receiver] = board._grid.ravel().tolist()
        pending = list(self.pending)
        pending[self.receiver] = float(left)
        return boards, pending

    def prefetch(self, items: list[tuple[Board, int]]) -> None:
        todo: dict[tuple, tuple[Board, int]] = {}
        for board, left in items:
            key = self.key(board, left)
            if key not in self.values and key not in todo:
                todo[key] = (board, left)
        if todo:
            results = self.many([self.request(b, l) for b, l in todo.values()])
            self.evaluated += len(results)
            for key, value in zip(todo, results):
                self.values[key] = float(value)

    def get(self, board: Board, left: int) -> float:
        key = self.key(board, left)
        if key not in self.values:
            self.prefetch([(board, left)])
        return self.values[key]


def death_column_ojama(drop: int) -> int:
    """着弾 `drop` 個 (代表の端数列) のうち、窒息判定の列 (DEATH_COL) に落ちるおじゃまの個数。"""
    if drop <= 0:
        return 0
    return drop//OJAMA_ROW_SIZE+int(DEATH_COL in arrangement_columns(drop)[0])


def dead_after_landing(raw: bytes, ojama_in_column: int) -> bool:
    """応手の盤面 (バイト列) へ着弾したあと窒息するか。盤面を作らず、窒息判定の列だけを見る。

    おじゃまは列の空きセルを下から順に埋める。窒息判定のセル (DEATH_ROW) が、元から埋まっている
    (UNKNOWN は判定不能で窒息としない)、または、その下の空きセル数が落ちる個数以内なら窒息。
    """
    column = raw[DEATH_COL::BOARD_COLS]
    cell = column[DEATH_ROW]
    if cell != COLOR_EMPTY:
        return cell != COLOR_UNKNOWN
    return column[DEATH_ROW+1:].count(COLOR_EMPTY) < ojama_in_column


def first_alive_fast(ranked: Any, pending: int, carry: int, elapsed: float) -> dict | None:
    """v4.9d の `first_alive` と同じ候補。窒息する応手は盤面を作らずに読み飛ばし、最初に生き残る応手だけを v4.9d の手順で作る。"""
    if ARRANGEMENT_SAMPLES != 1 or not isinstance(ranked, Replies):
        return first_alive(ranked, pending, carry, elapsed)
    landing: dict[int, int] = {}
    for raw, points in zip(ranked.grids, ranked.points):
        if points not in landing:
            left = max(0, pending-cancel_amount(points, carry, elapsed))
            landing[points] = death_column_ojama(landing_count(left))
        if dead_after_landing(raw, landing[points]):
            continue
        found = candidate((Replies.board_of(raw), points), pending, carry, elapsed)
        if found is not None:
            return found
    return None


def pick_candidates_v49e(ranked: Any, rest: int, ctx: ContinuousInput, clock: StageClock,
                         need_top: bool, need_strata: bool) -> tuple[dict | None, list]:
    """v4.9d の `pick_candidates` と同じ代表 (順位最上位で生き残る応手・3層それぞれの代表と層の大きさ)。

    3層は応手列全体を順に分けた区間なので、全体で最初に生き残る応手は、層の代表のうち最初に見つかったもの。
    層を求めるときは最上位の応手を別に探さない (同じ順位を重ねて数えない)。
    """
    with clock.time('landing'):
        picks = []
        for lo, hi in (strata_bounds(len(ranked)) if need_strata else []):
            pick = first_alive_fast(ranked[lo:hi], rest, ctx.carry, ctx.elapsed)
            if pick is not None:
                picks.append((pick, hi-lo))
        if not need_top:
            top = None
        elif need_strata:
            top = picks[0][0] if picks else None
        else:
            top = first_alive_fast(ranked, rest, ctx.carry, ctx.elapsed)
    return top, picks


def cached_picks(found_key: tuple, ranked: Any, rest: int, ctx: ContinuousInput, clock: StageClock,
                 need_top: bool, need_strata: bool) -> tuple[dict | None, list]:
    """代表選びを、応手集合・予告・繰越・換算レートが同じなら更新をまたいで再利用する。"""
    key = (found_key, int(rest), int(ctx.carry), compute_effective_rate(ctx.elapsed), bool(need_top), bool(need_strata))
    if key not in _PICK_MEMO:
        if len(_PICK_MEMO) >= PICK_MEMO_LIMIT:
            _PICK_MEMO.pop(next(iter(_PICK_MEMO)))
        _PICK_MEMO[key] = pick_candidates_v49e(ranked, rest, ctx, clock, need_top, need_strata)
    return _PICK_MEMO[key]


def continuous_values_v49e(ctx: ContinuousInput, many: Many, clock: StageClock, variants: tuple,
                           native: Any, work_limit: float | None, lean: bool = True,
                           max_hands: int = RESPONSE_MAX_HANDS) -> dict:
    """v4.9d の `continuous_values` と同じ値 (集約方法ごとの勝率)。B3は必要な候補盤を全部決めてから1回で評価する。"""
    side, pending = ctx.receiver, int(ctx.pending[ctx.receiver])
    memo = BatchMemo(many, ctx.boards, ctx.pending, side)
    board = grid_board(ctx.boards[side])
    if pending <= 0 or COLOR_UNKNOWN in ctx.boards[side]:
        value = memo.get(board, pending)
        return dict(values={k: value for k in variants}, responses=0, strata=[], complete=True, nodes=0, enum={})
    start, rest = board, pending
    if ctx.immediate:
        start, rest = landed_boards(board, landing_count(pending))[0], pending-landing_count(pending)
    with clock.time('landing'):
        landed = landed_boards(start, landing_count(rest))[0]
    base = (landed, rest-landing_count(rest))
    if AGG_REP not in variants and AGG_MEAN not in variants:
        values = {AGG_BASE: memo.get(*base)} if AGG_BASE in variants else {}
        return dict(values=values, responses=0, strata=[], complete=True, nodes=0, enum={})
    with clock.time('enumerate'):
        if lean:
            found = ranked_responses(start, ctx.queue, ctx.hands, ctx.colors, native, RESPONSE_NODE_LIMIT, work_limit, max_hands)
        else:
            found = feasible_responses_v49d(start, ctx.queue, ctx.hands, ctx.elapsed, ctx.colors)
            found = dict(found, ranked=compact(found['replies']))
        ranked = found['ranked']
    found_key = (start._grid.tobytes(), tuple(ctx.queue), int(ctx.hands), tuple(ctx.colors), work_limit, lean, int(max_hands))
    top, picks = cached_picks(found_key, ranked, rest, ctx, clock, AGG_REP in variants, AGG_MEAN in variants)
    wanted = ([base] if AGG_BASE in variants else [])
    wanted += [(top['board'], top['left'])] if AGG_REP in variants and top is not None else []
    wanted += [(c['board'], c['left']) for c, _ in picks] if AGG_MEAN in variants else []
    memo.prefetch(wanted)
    values: dict[str, float] = {}
    if AGG_BASE in variants:
        values[AGG_BASE] = memo.get(*base)
    if AGG_REP in variants:
        values[AGG_REP] = memo.get(top['board'], top['left']) if top is not None else memo.get(*base)
    if AGG_MEAN in variants:
        values[AGG_MEAN] = (sum(w*memo.get(c['board'], c['left']) for c, w in picks)/sum(w for _, w in picks)
                            if picks else memo.get(*base))
    enum = {k: found[k] for k in ('sims', 'steps', 'work', 'truncated') if k in found}
    return dict(values=values, responses=len(ranked), strata=[w for _, w in picks],
                complete=found['complete'], nodes=found['nodes'], enum=dict(enum, b3_evals=memo.evaluated))


class LandingV49E(LandingV49D):
    """証明 (a) と保持は v4.9d のまま。連続値 (b) だけ、B3まとめ評価と列挙の高速版に差し替える。"""

    def __init__(self, work_limit: float | None = None, lean: bool = True, max_hands: int = RESPONSE_MAX_HANDS) -> None:
        super().__init__()
        self.work_limit, self.lean, self.max_hands = work_limit, lean, max_hands
        self.last_enum: dict = {}

    def continuous(self, engine: Any, value: dict, evaluate: Callable, clock: StageClock,
                   variants: tuple = AGGREGATORS) -> dict:
        state = engine.state
        receiver = max(SIDES, key=lambda i: (value['pending'][i], -i))
        schedule, active = value['landing_schedule'][receiver], value['board_prediction_active']
        immediate = bool(active[receiver] and schedule['certain'])
        other = engine.chains[1-receiver]
        hands = (remaining_hands(len(other['best']['scores']), other['started'], state.t)
                 if other and active[1-receiver] else 1)
        queue = tuple(state.sides[receiver].queue)
        queue = queue if immediate else (0, 0)+queue
        known = state.colors.known()[1]
        ctx = ContinuousInput(boards=value['boards'], pending=value['pending'], receiver=receiver,
            queue=queue, hands=hands, immediate=immediate,
            elapsed=margin_elapsed(state.t, value['margin_origin_sec']),
            carry=int(state.sl_snap[f'carry_p{receiver+1}']),
            colors=tuple(known) if len(known) == 4 else (1, 2, 3, 4))
        result = continuous_values_v49e(ctx, evaluate.many, clock, variants, engine.native, self.work_limit, self.lean,
                                          self.max_hands)
        self.last_enum = result.pop('enum')
        return dict(result, receiver=receiver, hands=hands, immediate=immediate)
