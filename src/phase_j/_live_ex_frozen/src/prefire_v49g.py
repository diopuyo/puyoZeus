"""受け連鎖の完走後へ既存証明を適用する採用済みv4.9g。配信版への接続は別管理。"""
from __future__ import annotations

from typing import Any
from src.phase_j._live_ex_frozen.src.prefire_v49f import PostV49F
from src.phase_j._live_ex_frozen.src.prefire_landing_v49e import LandingV49E
from src.phase_j._live_ex_frozen.src.prefire_receiver_v49g import ReceiverTracker, projected_receiver
from src.phase_j._live_ex_frozen.src.prefire_proof_contract import blockers
from src.phase_j._live_ex_frozen.src.prefire_landing_v49d import (
    StageClock, grid_board, margin_elapsed, prove_death, CACHE_LIMIT)
from src.phase_j._live_ex_frozen.src.scoring import compute_effective_rate
from src.phase_j._live_ex_frozen.src.production_config import PREFIRE_V49G_ENABLED_DEFAULT, PREFIRE_V49G_RECEIVER_OPEN_DEFAULT

REPLACED = frozenset(('receiver_chain_open', 'receiver_not_current', 'board_not_resting'))


def receiver_blockers(facts: Any, engine: Any, side: int) -> list[str]:
    """受け側の完走で置換する条件以外は維持する。相手連鎖の未完走は解除しない。"""
    found = [reason for reason in blockers(facts) if reason not in REPLACED]
    other = engine.chains[1-side]
    own_only = (engine.state.sl.core.st[1-side].chain is None
                and (other is None or other['finished']))
    if own_only:
        found = [reason for reason in found if reason != 'opponent_unfinished']
    return found


def physical_proof(ctx: dict, clock: StageClock) -> dict:
    """NEXT1から全応手を検証。新しい全消しボーナスは次の返しにだけ使う。"""
    from src.phase_j._live_ex_frozen.src.prefire_responses_v49d import landing_count
    board = grid_board(ctx['board'])
    first = ctx.get('first_drop', landing_count(ctx['pending']) if ctx['immediate'] else 0)
    if not ctx['future_credit'] and (not first or first == landing_count(ctx['pending'])):
        return prove_death(board, ctx['queue'], ctx['pending'], 1, ctx['elapsed'], ctx['immediate'], clock)
    return after_initial(ctx, first, clock)


def after_initial(ctx: dict, first: int, clock: StageClock) -> dict:
    """割り込みの古い予告だけを落とす。端数列の全分岐で作業量上限を共有する。"""
    from itertools import combinations
    from src.phase_j._live_ex_frozen.src.board import BOARD_COLS
    from src.phase_j._live_ex_frozen.src.exchange_fast_expand import drop_ojama_fast
    from src.phase_j._live_ex_frozen.src.exchange_event_multilanding import prove_multilanding, SearchCutoff
    from src.phase_j._live_ex_frozen.src.prefire_landing_v49d import PROOF_NODE_LIMIT, OPTIMISTIC_CALL_LIMIT, optimistic_response
    board, nodes, calls = grid_board(ctx['board']), 0, [0]
    if board.is_dead():
        return dict(dead=False, reason='invalid_start', nodes=0, optimistic_calls=0)

    def optimistic(*args: Any) -> float:
        calls[0] += 1
        if calls[0] > OPTIMISTIC_CALL_LIMIT:
            raise SearchCutoff('optimistic_limit')
        with clock.time('optimistic'):
            return optimistic_response(*args)

    for cols in combinations(range(BOARD_COLS), first % BOARD_COLS):
        landed = drop_ojama_fast(board, first, cols) if first else board
        if landed.is_dead():
            continue
        left = ctx['pending']-first
        if left <= 0:
            return dict(dead=False, reason='landing_complete', nodes=nodes, optimistic_calls=calls[0])
        proof = prove_multilanding(landed, ctx['queue'], left, 1, ctx['elapsed'], optimistic,
            credit=ctx['future_credit'], node_limit=max(0, PROOF_NODE_LIMIT-nodes), count_first=True)
        nodes += min(proof['nodes'], max(0, PROOF_NODE_LIMIT-nodes))
        if not proof['dead']:
            return dict(proof, nodes=nodes, optimistic_calls=calls[0])
    return dict(dead=True, reason='receiver_after_initial', nodes=nodes, optimistic_calls=calls[0])


class LandingV49G(LandingV49E):
    """追加証明は更新ごとに再確認し、旧証明の保持根拠に混ぜない。"""

    def __init__(self, diagnose_guarded: bool = False) -> None:
        super().__init__(work_limit=100, lean=True)
        self.receiver_cache: dict[tuple, dict] = {}
        self.guard_probe = LandingV49E(work_limit=100, lean=True)
        self.diagnose_guarded = diagnose_guarded

    def solve(self, ctx: dict, clock: StageClock) -> dict:
        key = (tuple(ctx['board']), tuple(ctx['queue']), ctx['pending'], ctx['immediate'], ctx.get('first_drop'),
               ctx['future_credit'], compute_effective_rate(ctx['elapsed']))
        hit = key in self.receiver_cache
        if not hit:
            if len(self.receiver_cache) >= CACHE_LIMIT:
                self.receiver_cache.pop(next(iter(self.receiver_cache)))
            self.receiver_cache[key] = physical_proof(ctx, clock)
        return dict(self.receiver_cache[key], cache_hit=hit)

    def diagnose(self, side: int, engine: Any, tracker: Any, value: dict, clock: StageClock) -> dict:
        facts = tracker.facts(side, engine.state, engine.state.sl.core, value, value['board_prediction_active'])
        found = blockers(facts) if facts.pending > 0 else []
        result = dict(blockers=found, eligible=False, physical_dead=False, reason=None)
        if 'receiver_chain_open' in found:
            ctx, reason = projected_receiver(engine, tracker, side)
            if ctx is None:
                return dict(result, reason=reason)
            ctx['elapsed'] = margin_elapsed(ctx['end'], value['margin_origin_sec'])
            remaining = receiver_blockers(facts, engine, side)
            # 元射影の予告不足は、独立の完走後台帳で再確認済み。
            remaining = [r for r in remaining if r != 'confirmed_short']
            if remaining and not self.diagnose_guarded:
                return dict(result, context=ctx, remaining_blockers=remaining,
                            reason='receiver_guards_reject', diagnostic_skipped=True)
            proof = self.solve(ctx, clock)
            return dict(result, context=ctx, proof=proof, physical_dead=proof['dead'],
                        eligible=not remaining, remaining_blockers=remaining, reason=proof['reason'])
        if 'board_guard' in found and self.diagnose_guarded:
            proof = self.guard_probe.proof(side, engine, value, clock)
            return dict(result, proof=proof, physical_dead=proof['dead'], reason=proof['reason'])
        return result

    def guarded_diagnosis(self, side: int, engine: Any, tracker: Any,
                          value: dict, clock: StageClock) -> dict:
        """診断の失敗で既存証明を消さず、追加証明だけを棄権して例外を原票へ残す。"""
        try:
            return self.diagnose(side, engine, tracker, value, clock)
        except Exception as exc:  # 診断異常は受領票で必ず検収し、成功とは扱わない。
            engine._note('receiver_diagnostic', engine.state.t, exc)
            return dict(blockers=[], eligible=False, physical_dead=False,
                        reason='diagnostic_error', error=repr(exc))

    def apply(self, engine: Any, tracker: Any, value: dict, clock: StageClock) -> dict:
        super().apply(engine, tracker, value, clock)
        diagnostics = [self.guarded_diagnosis(s, engine, tracker, value, clock) for s in (0, 1)]
        value['receiver_open_diagnostics'] = diagnostics
        for side, detail in enumerate(diagnostics):
            if not (detail['eligible'] and detail['physical_dead']):
                continue
            audit = value['landing_proofs'][side]
            proof = detail['proof']
            audit.update(dead=True, held=False, admitted=True, reject=None,
                         reason='receiver_completed_'+proof['reason'], cache_hit=proof['cache_hit'],
                         nodes=proof['nodes'], optimistic_calls=proof['optimistic_calls'])
            value['death_sides'] = sorted(set(value['death_sides']) | {side})
            value['evidence_times'] = [*value['evidence_times'], detail['context']['evidence_sec']]
        value['proof_nodes'] = sum(a['nodes'] for a in value['landing_proofs'] if not a['cache_hit'])
        value['optimistic_calls'] = sum(a['optimistic_calls'] for a in value['landing_proofs'] if not a['cache_hit'])
        return value


class PostV49G(PostV49F):
    """v4.9fを基点とし、採用済みの受け連鎖証明を既定で有効にする。"""

    def __init__(self, predictor: Any, table: Any, policy: dict, native: Any,
                 enabled: bool = PREFIRE_V49G_ENABLED_DEFAULT, pre_model: dict | None = None,
                 receiver_chain_open: bool = PREFIRE_V49G_RECEIVER_OPEN_DEFAULT,
                 diagnose_guarded: bool = False) -> None:
        super().__init__(predictor, table, policy, native, enabled, pre_model)
        if receiver_chain_open:
            self.tracker = ReceiverTracker(native)
            self.landing = LandingV49G(diagnose_guarded)
