"""v4.9c用の因果的な完走・会計接続。確定盤面と得点完了を分離する。"""
from __future__ import annotations

from typing import Any
from src.phase_j._live_ex_frozen.src.exchange_count_ledger import gross, FORMULA_SCORE_DELAY_SEC, STEP_SEC
from src.phase_j._live_ex_frozen.src.prefire_post_v47 import PostV47
from src.phase_j._live_ex_frozen.src.prefire_pending_v47 import project

SIDES = (0, 1)


def chain_records(core: Any, side: int, trigger: float) -> list:
    """現在と完了済みの台帳から、当該発火以後の記録だけを参照する。"""
    records = list(core.chains)
    current = core.st[side].chain
    if current is not None and current not in records:
        records.append(current)
    return [r for r in records if r.side == side and r.t_start >= trigger]


def reconciled_send(chain: dict, core: Any, side: int) -> tuple[int, list[float]]:
    """予測の得点を現在台帳と同じ起点・端数・観測反映時刻で換算する。"""
    scores, trigger = chain['best']['scores'], chain['started']
    observed = [step.t_score for record in chain_records(core, side, trigger)
                for step in record.steps if step.t_score is not None]
    times = [observed[k] if k < len(observed) else
             trigger+FORMULA_SCORE_DELAY_SEC+k*STEP_SEC for k in range(len(scores))]
    total, _ = gross(scores, times, int(chain['carry_at_fire']), core.origin_sec)
    if getattr(core, 'defer_all_clear', False):
        total += core.all_clear[side].forecast(trigger)
    return total, times


class PhysicsV49C(PostV47):
    """モデルを変更せず、v4.9cでだけ台帳の完走信号を盤面寿命へ接続する。"""

    def _finish_chain(self, side: int, stamp: float, source: dict) -> None:
        chain = self.chains[side]
        if chain is None:
            return
        records = chain_records(self.state.sl.core, side, chain['started'])
        ended = any(r.t_end is not None and r.t_end <= stamp for r in records)
        if ended and source['state']['state'] == 'STABLE':
            chain['finished'] = True
            chain['physical_end_sec'] = max(r.t_end for r in records if r.t_end is not None)

    def projection(self) -> dict:
        value = super().projection()
        core, amounts, audit = self.state.sl.core, [], []
        for side, chain in enumerate(self.chains):
            remaining, detail = 0, None
            if chain is not None and core.origin_sec is not None:
                total, times = reconciled_send(chain, core, side)
                reflected = core.st[side].generated_total-chain['generated_at_fire']
                remaining = max(0, total-reflected) if not chain['finished'] else 0
                detail = dict(original=chain.get('external_sent'), total=total,
                    reflected=reflected, remaining=remaining, times=times,
                    origin=core.origin_sec, carry=chain['carry_at_fire'],
                    physical_end_sec=chain.get('physical_end_sec'))
            amounts.append(remaining)
            audit.append(detail)
        triggers = [c['started'] if c and not c['finished'] else None for c in self.chains]
        ends = [max(self.state.t, c['started']+FORMULA_SCORE_DELAY_SEC+
                    (len(c['best']['scores'])-1)*STEP_SEC) if c and not c['finished'] else None
                for c in self.chains]
        value.update(project(core, amounts, self.state.t, triggers, ends),
                     remaining=amounts, reconciliation=audit,
                     current_boards=[s.board.ravel().tolist() for s in self.state.sides],
                     board_prediction_active=[self._resolving(i) for i in SIDES])
        return value
