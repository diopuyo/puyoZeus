"""受け側の発火前盤を完走し、死亡証明へ渡す既定OFF試作の入力。"""
from __future__ import annotations

from typing import Any
import numpy as np
from src.phase_j._live_ex_frozen.src.exchange_count_ledger import FORMULA_SCORE_DELAY_SEC, STEP_SEC, gross
from src.phase_j._live_ex_frozen.src.prefire_facts_v49d import FactsTracker, CHAIN_LINK_SEC, PHYSICS_SLACK
from src.phase_j._live_ex_frozen.src.prefire_pending_v47 import fork_ledger, attack, conditional_landing
from src.phase_j._live_ex_frozen.src.b3_all_clear_credit import BONUS_OJAMA

SIDES = (0, 1)
PAIR_SIZE = 2


class ReceiverTracker(FactsTracker):
    """既存のガードを維持し、到着済みの連鎖前盤と式の照合結果を追加する。"""

    def reset(self) -> None:
        super().reset()
        self.completions: list[dict | None] = [None, None]

    def _physics(self, side: int, raw: dict, stamp: float) -> None:
        super()._physics(side, raw, stamp)
        event = raw.get('chain_event')
        if not event:
            return
        ns = event['namespace']
        board = [int(c) for c in np.asarray(ns['before_board']['board']['array']).ravel()]
        solved = self._solve(board)
        count, reported = int(ns['chain_count']), int(ns['total_score'])
        valid = (solved is not None and 0 < count <= solved['chain_count']
                 and abs(solved['prefix'][count-1]-reported) <= PHYSICS_SLACK)
        self.completions[side] = dict(trigger=float(ns['trigger_sec']), seen=stamp,
            board=board, solved=solved, count=count, reported=reported, valid=valid)


def completion_input(engine: Any, tracker: ReceiverTracker, side: int) -> tuple[dict | None, str | None]:
    """現在の台帳連鎖と一致し、観測済み段を説明できる物理完走だけを返す。"""
    core, stamp = engine.state.sl.core, engine.state.t
    record, known = core.st[side].chain, tracker.completions[side]
    if record is None or known is None:
        return None, 'receiver_completion_missing'
    if engine.state.sides[side].state != 'CHAIN':
        return None, 'receiver_not_chaining'
    if known['seen'] > stamp or known['trigger'] > stamp:
        return None, 'receiver_future_evidence'
    if abs(record.t_start-known['trigger']) > CHAIN_LINK_SEC:
        return None, 'receiver_chain_identity'
    if not known['valid']:
        return None, 'receiver_prefix_mismatch'
    if any(step.product <= 0 for step in record.steps):
        return None, 'receiver_formula_missing'
    solved = known['solved']
    scores = np.diff([0]+solved['prefix']).astype(int).tolist()
    if len(record.steps) > len(scores):
        return None, 'receiver_step_overrun'
    if any(abs(step.product-score) > PHYSICS_SLACK for step, score in zip(record.steps, scores)):
        return None, 'receiver_ledger_mismatch'
    return dict(known, scores=scores), None


def remaining_send(engine: Any, side: int, known: dict) -> tuple[int, float]:
    """全連鎖の送りから既に台帳へ反映済みの生成を引く。全消しは固定30個。"""
    core, stamp = engine.state.sl.core, engine.state.t
    trigger, scores = known['trigger'], known['scores']
    history = next((r for r in reversed(engine.ledger_history) if r[0] < trigger), None)
    if history is None:
        raise ValueError('発火前台帳がない')
    record = core.st[side].chain
    times = [record.steps[k].t_score if k < len(record.steps) and record.steps[k].t_score is not None
             else max(stamp, trigger+FORMULA_SCORE_DELAY_SEC+k*STEP_SEC) for k in range(len(scores))]
    total, _ = gross(scores, times, int(history[3+side]), core.origin_sec)
    if core.defer_all_clear:
        total += core.all_clear[side].forecast(trigger)
    # 前の連鎖の遅着得点を今回の反映分へ混ぜない。分割された連鎖は返しを多めに見積もる。
    reflected = min(core.st[side].generated_total-history[1+side],
                    sum(step.generated for step in record.steps))
    return max(0, int(total-reflected)), max(times)


def projected_receiver(engine: Any, tracker: ReceiverTracker, side: int) -> tuple[dict | None, str | None]:
    """連鎖終了までは落とさず、新しい予告から相殺した後の盤面・NEXT1を返す。"""
    known, reason = completion_input(engine, tracker, side)
    if known is None:
        return None, reason
    core = engine.state.sl.core
    if core.origin_sec is None:
        return None, 'origin_unknown'
    if core.defer_all_clear and core.all_clear[side].uncertain_points:
        return None, 'receiver_all_clear_uncertain'
    try:
        remaining, end = remaining_send(engine, side, known)
    except ValueError:
        return None, 'receiver_baseline_missing'
    ledger = fork_ledger(core)
    attack(ledger, side, remaining, engine.state.t)
    pending = ledger._pending(side)
    if ledger._pending(side, True) < pending:
        return None, 'confirmed_short'
    schedule = conditional_landing(ledger, side, known['trigger'], end)
    board = known['solved']['final_board']
    # この連鎖で獲得する全消しは今回の相殺へ入れず、次の返しへ楽観的に与える。
    future_credit = BONUS_OJAMA if not any(board) else 0
    return dict(board=board, pending=pending, queue=tuple(engine.state.sides[side].queue),
        immediate=bool(schedule['certain'] and schedule['count']), end=end,
        first_drop=schedule['count'] if schedule['certain'] else 0,
        remaining_send=remaining, future_credit=future_credit, evidence_sec=known['seen'],
        trigger=known['trigger'], scores=known['scores'], schedule=schedule), None
