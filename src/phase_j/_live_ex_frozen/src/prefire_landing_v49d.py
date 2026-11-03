"""v4.9d: 死亡証明 (a) と連続値 (b) を分離する。

(a) 死亡証明は、入力契約 (`prefire_proof_contract`) を通った入力だけで、許容されるすべての返しを含めても
    死ぬときに限り 0/1 を出す。保持は根拠 (予告の量・連鎖世代・着弾) と結び付け、反証が来たら解除する。
(b) 連続値は、実際に起こりうる応手と、その応手自身が生む相殺量を結び付けた盤で固定B3を評価する。
    証明が未決でも捨てない。集約方法は事前登録の3案 (`AGGREGATORS`)。
打切りはノード数・楽観返しの呼出し回数・手数だけ。壁時計では結果を変えない。
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from hashlib import sha256
from itertools import combinations
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Iterator
import numpy as np
from src.phase_j._live_ex_frozen.src.board import Board, BOARD_ROWS, BOARD_COLS, COLOR_UNKNOWN
from src.phase_j._live_ex_frozen.src.exchange_event_multilanding import prove_multilanding, Search, SearchCutoff
from src.phase_j._live_ex_frozen.src.exchange_event_landing import ExchangeLandingProjection, remaining_hands
from src.phase_j._live_ex_frozen.src.exchange_fast_expand import drop_ojama_fast
from src.phase_j._live_ex_frozen.src.indicators_v2 import SEC_PER_HAND
from src.phase_j._live_ex_frozen.src.post_counter_geometry import all_landings_dead
from src.phase_j._live_ex_frozen.src.scoring import (compute_effective_rate, MARGIN_TIME_START_SEC,
                         MARGIN_TIME_START_FROM_FIRST_MOVE_SEC)
from src.phase_j._live_ex_frozen.src.prefire_certificate_v49d import Basis, judge, KEEP, DROP
from src.phase_j._live_ex_frozen.src.prefire_proof_contract import ContractTally
from src.phase_j._live_ex_frozen.src.prefire_responses_v49d import (feasible_responses, cancel_amount, landed_boards,
                                        landing_count, representative)

SIDES = (0, 1)
PAIR_SIZE = 2
# --- 事前登録の集約方法 (REPORT「事前登録」節と一致させる。結果を見て変えない) ---
AGG_BASE, AGG_REP, AGG_MEAN = 'base', 'rep', 'mean'
AGGREGATORS = (AGG_BASE, AGG_REP, AGG_MEAN)
MEAN_STRATA = 3
# --- 決定的な作業量の上限 (壁時計は使わない) ---
# 付録C 100入力の実測: 死亡証明39件のうち37件は700ノード・楽観返し1回以内。
# 上限は最悪の応答時間 (約200ms) を抑える値 (ノード約35us、楽観返し約23ms)。
PROOF_NODE_LIMIT = 3000
OPTIMISTIC_CALL_LIMIT = 4
# 再証明の猶予: 着弾で根拠が正当に変わった後、契約を通る再証明が来るまで保持してよい時間 (2手ぶん)。
RECERTIFY_GRACE_SEC = 2*SEC_PER_HAND
CODE_FILES = ('src/prefire_landing_v49d.py', 'src/prefire_responses_v49d.py',
              'src/prefire_proof_contract.py', 'src/prefire_facts_v49d.py',
              'src/exchange_event_multilanding.py', 'src/exchange_landing_capacity.py')


def code_digest(root: Path | None = None) -> str:
    """証明経路のソースのSHA (改行差を吸収)。結果の再利用キーに含め、コードが変われば無効化する。"""
    base = root or Path(__file__).resolve().parents[1]
    digest = sha256()
    for name in CODE_FILES:
        digest.update((base/name).read_bytes().replace(b'\r\n', b'\n'))
    return digest.hexdigest()[:16]


CODE_SHA = code_digest()


class StageClock:
    """工程ごとの所要時間 (秒) と呼出し回数。時間は計測だけに使い、結果は変えない。"""

    def __init__(self) -> None:
        self.seconds: dict[str, float] = {}
        self.calls: dict[str, int] = {}

    @contextmanager
    def time(self, stage: str) -> Iterator[None]:
        began = perf_counter()
        try:
            yield
        finally:
            self.seconds[stage] = self.seconds.get(stage, 0.)+perf_counter()-began
            self.calls[stage] = self.calls.get(stage, 0)+1

    def snapshot(self) -> dict:
        return dict(ms={k: v*1000. for k, v in self.seconds.items()}, calls=dict(self.calls))


@dataclass
class TimedSearch(Search):
    """応手列挙・着地を工程別に計測する証明用の探索器 (結果は基底と同一)。"""
    clock: StageClock | None = field(default=None, compare=False)

    def responses(self, board: Board, queue: tuple, hands: int) -> list:
        with self.clock.time('enumerate'):
            return super().responses(board, queue, hands)

    def land(self, reply: Board, dropped: int, columns: tuple) -> Board:
        with self.clock.time('landing'):
            return super().land(reply, dropped, columns)


# 出力を変えない高速化 (全行一致を確認済みの場合だけ有効): 楽観返しは時刻を得点換算レートだけで使うため、
# 同じ (盤面・ツモ・手数・レート) の結果を更新をまたいで再利用する。False で従来の毎回計算。
EXACT_OPTIMISTIC_MEMO = True
OPTIMISTIC_MEMO_LIMIT = 8192
_OPTIMISTIC_MEMO: dict[tuple, float] = {}


def optimistic_response(board: Board, queue: np.ndarray, hands: int, elapsed: float) -> float:
    """楽観返し。結果は盤面・ツモ・手数と、時刻から決まる換算レートだけで決まる。"""
    if not EXACT_OPTIMISTIC_MEMO:
        return ExchangeLandingProjection._optimistic_response(None, board, queue, hands, elapsed)
    key = (board._grid.tobytes(), tuple(int(v) for v in queue), int(hands), compute_effective_rate(elapsed))
    if key not in _OPTIMISTIC_MEMO:
        if len(_OPTIMISTIC_MEMO) >= OPTIMISTIC_MEMO_LIMIT:
            _OPTIMISTIC_MEMO.pop(next(iter(_OPTIMISTIC_MEMO)))
        _OPTIMISTIC_MEMO[key] = ExchangeLandingProjection._optimistic_response(None, board, queue, hands, elapsed)
    return _OPTIMISTIC_MEMO[key]


def margin_elapsed(now: float, origin: float | None) -> float:
    """マージンタイム経過秒 (v4.9cと同じ換算)。"""
    elapsed = now-origin if origin is not None else 0.
    return elapsed+MARGIN_TIME_START_SEC-MARGIN_TIME_START_FROM_FIRST_MOVE_SEC


def prove_death(board: Board, queue: tuple, pending: int, hands: int, elapsed: float,
                immediate: bool, clock: StageClock) -> dict:
    """許容されるすべての返しを含めても窒息するときだけ dead。安い個数判定を先に行う。"""
    if np.any(board._grid == COLOR_UNKNOWN) or board.is_dead():
        return dict(dead=False, reason='invalid_start', nodes=0, optimistic_calls=0)
    first = landing_count(pending) if immediate else 0
    with clock.time('count'):
        if immediate and all_landings_dead(board, first):
            return dict(dead=True, reason='capacity_immediate', nodes=0, optimistic_calls=0,
                        drop=first, min_drop=first)
        starts = ([drop_ojama_fast(board, first, c) for c in combinations(range(BOARD_COLS), first % BOARD_COLS)]
                  if immediate else [board])
    calls = [0]

    def optimistic(b: Board, q: np.ndarray, h: int, e: float) -> float:
        calls[0] += 1
        if calls[0] > OPTIMISTIC_CALL_LIMIT:
            raise SearchCutoff('optimistic_limit')
        with clock.time('optimistic'):
            return optimistic_response(b, q, h, e)

    def factory(*args: Any, **kwargs: Any) -> TimedSearch:
        return TimedSearch(*args, clock=clock, **kwargs)

    nodes, proofs = 0, []
    for current in starts:
        if current.is_dead():
            continue
        left = pending-first
        if left <= 0:
            return dict(dead=False, reason='landing_complete', nodes=nodes, optimistic_calls=calls[0])
        proof = prove_multilanding(current, queue, left, max(1, hands), elapsed, optimistic,
            node_limit=max(0, PROOF_NODE_LIMIT-nodes), count_first=True, search_factory=factory)
        nodes += min(proof['nodes'], max(0, PROOF_NODE_LIMIT-nodes))
        proofs.append(proof)
        if not proof['dead']:
            return dict(dead=False, reason=proof['reason'], nodes=nodes, optimistic_calls=calls[0])
    reason = 'capacity_all_branches' if all(p['reason'].startswith('capacity') for p in proofs) else 'all_responses_dead'
    return dict(dead=True, reason=reason, nodes=nodes, optimistic_calls=calls[0], drop=first,
                min_drop=first_drop_floor(proofs, first))


def first_drop_floor(proofs: list[dict], first: int) -> int:
    """証明が仮定した最初の着弾量の下限。観測された着弾がこれを下回れば「予測と違う着弾」。"""
    drops: list[int] = []
    for proof in proofs:
        if 'drop' in proof:
            drops.append(proof['drop'])
        rounds = proof.get('rounds') or []
        drops += [row.get('dropped', row.get('drop')) for row in (rounds[0] if rounds else [])
                  if row.get('dropped', row.get('drop')) is not None]
    return min(drops) if drops else first


# ---------------------------------------------------------------------------
# 連続値 (b)
# ---------------------------------------------------------------------------

Evaluate = Callable[[list[list[int]], list[float]], float]


@dataclass
class ContinuousInput:
    """連続値の計算に必要な入力。受け側1つぶん。"""
    boards: list[list[int]]
    pending: list[float]
    receiver: int
    queue: tuple
    hands: int
    immediate: bool
    elapsed: float
    carry: int
    colors: tuple


class Memo:
    """同じ (盤面, 予告) のB3評価を、1回の更新内で1度だけ行う。"""

    def __init__(self, evaluate: Evaluate, boards: list[list[int]], pending: list[float], receiver: int) -> None:
        self.evaluate, self.boards, self.pending, self.receiver = evaluate, boards, pending, receiver
        self.values: dict[tuple, float] = {}

    def __call__(self, board: Board, left: int) -> float:
        key = (board._grid.tobytes(), int(left))
        if key not in self.values:
            boards = [list(b) for b in self.boards]
            boards[self.receiver] = board._grid.ravel().tolist()
            pending = list(self.pending)
            pending[self.receiver] = float(left)
            self.values[key] = float(self.evaluate(boards, pending))
        return self.values[key]


def grid_board(cells: list[int]) -> Board:
    return Board.from_list(np.asarray(cells).reshape(BOARD_ROWS, BOARD_COLS).tolist())


def candidate(reply: tuple[Board, int], pending: int, carry: int, elapsed: float) -> dict | None:
    """応手1つ: その応手自身の相殺量で予告を減らし、最初の着弾を落として窒息しなければ候補にする。"""
    board, points = reply
    cancel = cancel_amount(points, carry, elapsed)
    left = max(0, pending-cancel)
    drop = landing_count(left)
    alive = [b for b in landed_boards(board, drop) if not b.is_dead()]
    if not alive:
        return None
    return dict(points=points, cancel=min(cancel, pending), board=alive[0], left=left-drop)


def first_alive(ranked: list, pending: int, carry: int, elapsed: float) -> dict | None:
    """順位の高い応手から探して、最初に生き残る候補。"""
    for reply in ranked:
        found = candidate(reply, pending, carry, elapsed)
        if found is not None:
            return found
    return None


def strata_bounds(size: int, count: int = MEAN_STRATA) -> list[tuple[int, int]]:
    """応手の順位 (相殺量の大きい順) を等頻度で `count` 層に分ける区間。"""
    parts = min(count, size)
    edges = [round(i*size/parts) for i in range(parts+1)] if parts else [0]
    return [(lo, hi) for lo, hi in zip(edges, edges[1:]) if hi > lo]


def continuous_values(ctx: ContinuousInput, evaluate: Evaluate, clock: StageClock,
                      variants: tuple = AGGREGATORS) -> dict:
    """指定した集約方法の連続値。応手が1つも得られなければ、基準盤 (応手なし) に戻す。

    `variants` に無い集約方法は計算しない (採用版の速度測定では選んだ1つだけを計算する)。
    同じ集約方法の値は、他の集約方法を同時に計算するかどうかに依らず同一。
    """
    side, pending = ctx.receiver, int(ctx.pending[ctx.receiver])
    memo = Memo(evaluate, ctx.boards, ctx.pending, side)
    board = grid_board(ctx.boards[side])
    if pending <= 0 or COLOR_UNKNOWN in ctx.boards[side]:
        # 予告なし、または未確定色を含む盤面は、物理を進めず現在の盤面と予告でそのまま評価する。
        value = memo(board, pending)
        return dict(values={k: value for k in variants}, responses=0, strata=[], complete=True, nodes=0)
    start, rest = board, pending
    if ctx.immediate:
        first = landing_count(pending)
        start, rest = landed_boards(board, first)[0], pending-first

    def base() -> float:
        with clock.time('landing'):
            landed = landed_boards(start, landing_count(rest))[0]
        return memo(landed, rest-landing_count(rest))

    values: dict[str, float] = {}
    if AGG_BASE in variants:
        values[AGG_BASE] = base()
    if AGG_REP in variants or AGG_MEAN in variants:
        with clock.time('enumerate'):
            found = feasible_responses(start, ctx.queue, ctx.hands, ctx.elapsed, ctx.colors)
            ranked = representative(found['replies'])
        top, picks = pick_candidates(ranked, rest, ctx, clock, AGG_MEAN in variants)
        if AGG_REP in variants:
            values[AGG_REP] = memo(top['board'], top['left']) if top is not None else base()
        if AGG_MEAN in variants:
            values[AGG_MEAN] = (sum(w*memo(c['board'], c['left']) for c, w in picks)/sum(w for _, w in picks)
                                if picks else base())
        return dict(values=values, responses=len(ranked), strata=[w for _, w in picks],
                    complete=found['complete'], nodes=found['nodes'])
    return dict(values=values, responses=0, strata=[], complete=True, nodes=0)


def pick_candidates(ranked: list, rest: int, ctx: ContinuousInput, clock: StageClock,
                    need_strata: bool) -> tuple[dict | None, list[tuple[dict, int]]]:
    """代表1手 (順位の最上位で生き残る応手) と、3層それぞれの代表と層の大きさ。"""
    with clock.time('landing'):
        top = first_alive(ranked, rest, ctx.carry, ctx.elapsed)
        picks = []
        for lo, hi in (strata_bounds(len(ranked)) if need_strata else []):
            pick = first_alive(ranked[lo:hi], rest, ctx.carry, ctx.elapsed)
            if pick is not None:
                picks.append((pick, hi-lo))
    return top, picks


# ---------------------------------------------------------------------------
# 証明 (a) の統括: 契約 → 保持の判定 → 証明 → 連続値 (b)
# ---------------------------------------------------------------------------

CACHE_LIMIT = 4096


class LandingV49D:
    """死亡証明の保持・再証明と、証明結果の再利用。勝敗ラベルは受け取らない。"""

    def __init__(self) -> None:
        self.game: int | None = None
        self.certs: dict[int, dict] = {}
        self.cache: dict[tuple, dict] = {}
        self.tally = ContractTally()
        self.released: dict[str, int] = {}

    def reset(self, game: int) -> None:
        """試合境界だけで証明と再利用結果を捨てる。撃ち合いの切替では根拠の判定に任せる。"""
        if game != self.game:
            self.game, self.certs, self.cache = game, {}, {}

    def basis(self, side: int, engine: Any, tracker: Any, value: dict, stamp: float,
              hands: int, min_drop: int) -> Basis:
        core, snap = engine.state.sl.core, engine.state.sl_snap
        return Basis(pending=int(value['pending_observed'][side]),
            receiver_seq=core.st[side].chain_seq, sender_seq=core.st[1-side].chain_seq,
            dropped=int(snap[f'dropped_p{side+1}']), placements=tracker.placements[side],
            hands=hands, min_drop=min_drop, stamp=stamp)

    def proof(self, side: int, engine: Any, value: dict, clock: StageClock) -> dict:
        """契約を通った入力だけを証明する。同じ根拠の入力は結果を再利用する (予告が減れば別の根拠)。"""
        state, core = engine.state, engine.state.sl.core
        elapsed = margin_elapsed(state.t, value['margin_origin_sec'])
        queue, pending = (0, 0)+tuple(state.sides[side].queue), int(value['pending'][side])
        board = grid_board(value['boards'][side])
        batches = tuple((b.n, b.src_side, b.chain_id, b.confirmed, b.confirmed_t) for b in core.batches[side])
        key = (side, board._grid.tobytes(), queue, pending, batches, 1, False,
               compute_effective_rate(elapsed), CODE_SHA, PROOF_NODE_LIMIT, OPTIMISTIC_CALL_LIMIT)
        hit = key in self.cache
        if not hit:
            if len(self.cache) >= CACHE_LIMIT:
                self.cache.pop(next(iter(self.cache)))
            self.cache[key] = prove_death(board, queue, pending, 1, elapsed, False, clock)
        return dict(self.cache[key], cache_hit=hit, incoming=pending, hands=1)

    def side(self, side: int, engine: Any, tracker: Any, value: dict, clock: StageClock) -> dict:
        """受け側1つぶんの契約・保持・証明。"""
        state, stamp = engine.state, engine.state.t
        facts = tracker.facts(side, state, state.sl.core, value, value['board_prediction_active'])
        reason = self.tally.add(facts)
        audit = dict(side=side, pending=facts.pending, admitted=reason is None and facts.pending > 0,
                     reject=reason, guard=facts.guard_reason, attack=facts.attack_reason,
                     held=False, released=None, dead=False, reason='no_pending' if facts.pending <= 0 else None,
                     nodes=0, optimistic_calls=0, cache_hit=False)
        now = self.basis(side, engine, tracker, value, stamp, 1, 0)
        cert = self.certs.get(side)
        if cert is not None:
            verdict, why = judge(Basis(**cert['basis']), now, reason is None, RECERTIFY_GRACE_SEC)
            if verdict == DROP:
                self.certs.pop(side)
                audit['released'] = why
                self.released[why] = self.released.get(why, 0)+1
            elif verdict == KEEP:
                return dict(audit, held=True, dead=True, reason='certified_hold')
            else:
                self.certs.pop(side)
        if facts.pending <= 0 or reason is not None:
            return audit
        proof = self.proof(side, engine, value, clock)
        audit.update(dead=proof['dead'], reason=proof['reason'], nodes=proof['nodes'],
                     optimistic_calls=proof['optimistic_calls'], cache_hit=proof['cache_hit'])
        if proof['dead']:
            self.certs[side] = dict(proof=dict(reason=proof['reason'], nodes=proof['nodes']), issued=stamp,
                basis=self.basis(side, engine, tracker, value, stamp, 1, proof.get('min_drop', 0)).as_dict())
        return audit

    def continuous(self, engine: Any, value: dict, evaluate: Evaluate, clock: StageClock,
                   variants: tuple = AGGREGATORS) -> dict:
        """連続値 (b)。受け側 (予告のある側) の応手を実際の相殺量と結び付けて評価する。"""
        state, core = engine.state, engine.state.sl.core
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
            carry=int(engine.state.sl_snap[f'carry_p{receiver+1}']),
            colors=tuple(known) if len(known) == 4 else (1, 2, 3, 4))
        return dict(continuous_values(ctx, evaluate, clock, variants), receiver=receiver, hands=hands,
                    immediate=immediate)

    def apply(self, engine: Any, tracker: Any, value: dict, clock: StageClock) -> dict:
        """両側を判定し、死亡の確定した側を value に載せる。"""
        self.reset(engine.state.game_idx)
        audits = [self.side(side, engine, tracker, value, clock) for side in SIDES]
        value.update(landing_proofs=audits, death_sides=sorted(self.certs),
                     proof_nodes=sum(a['nodes'] for a in audits if not a['cache_hit']),
                     optimistic_calls=sum(a['optimistic_calls'] for a in audits if not a['cache_hit']))
        return value
