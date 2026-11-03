"""v4.7専用の台帳接続。既存の得点換算・相殺・着弾部品を再利用する。"""
from __future__ import annotations

from copy import copy, deepcopy
from typing import Sequence
from src.phase_j._live_ex_frozen.src.b3_score_ledger import ScoreLedger, ScoreLedgerV2, _Batch, MODE_TIMING
from src.phase_j._live_ex_frozen.src.scoring import OJAMA_MAX_DROP_PER_TURN

SIDES = (0, 1)


class ConnectedLedger(ScoreLedger):
    """観測NEXT移動を初手起点へ接続。画像の未来や予測得点は受け取らない。"""

    def __init__(self, defer_all_clear: bool = False, native: object = None) -> None:
        super().__init__(newest_first_offset=True, defer_all_clear=defer_all_clear, all_clear_native=native)

    def update(self, frame: dict, placed_sides: Sequence[str] = ()) -> dict | None:
        first = self.core.origin_sec is None
        placed = tuple(placed_sides) + tuple(s['side'] for s in frame['sides']
            if first and s.get('next_slide_motion'))
        return super().update(frame, placed)


def fork_ledger(core: ScoreLedgerV2) -> ScoreLedgerV2:
    """射影に必要な現状態だけコピーし、実観測台帳を変更しない。"""
    result = copy(core)
    result.st, result.batches = deepcopy(core.st), deepcopy(core.batches)
    result.chains, result._events = [], []
    result.mode = MODE_TIMING
    return result


def attack(core: ScoreLedgerV2, side: int, amount: int, stamp: float,
           chain_id: int = -1) -> None:
    """換算済み送りを既存相殺器へ渡す。端数換算はgross/ScoreLedgerが担当する。"""
    if amount < 0:
        raise ValueError('負の送り')
    remaining = amount-core._offset_own(side, amount)
    if remaining:
        core.batches[1-side].append(_Batch(remaining, side, chain_id, gen_t=stamp))


def conditional_landing(core: ScoreLedgerV2, side: int, trigger: float | None,
                        end: float | None) -> dict:
    """時刻未確定の置き終わりを捏造しない。既に開始した連鎖の終了だけ予測する。"""
    amount = core._pending(side)
    if not amount:
        return dict(side=side, count=0, at=None, gate='相殺で消滅', certain=True)
    eligible = sum(b.n for b in core.batches[side] if b.confirmed and
        b.confirmed_t is not None and trigger is not None and b.confirmed_t <= trigger)
    if trigger is not None and end is not None and eligible:
        return dict(side=side, count=min(eligible, OJAMA_MAX_DROP_PER_TURN),
            at=end, gate='開始済み連鎖の終了後', certain=True)
    return dict(side=side, count=min(amount, OJAMA_MAX_DROP_PER_TURN), at=None,
        gate='送り確定後の非連鎖置き終わり（連鎖なら終了後）', certain=False)


def project(core: ScoreLedgerV2, amounts: list[int], stamp: float,
            triggers: list[float | None], ends: list[float | None]) -> dict:
    """既発火の送りのみ相殺。未発火の打ち返し・追撃は一切ここに加えない。"""
    ledger = fork_ledger(core)
    for side in SIDES:
        attack(ledger, side, amounts[side], stamp)
    before = [ledger._pending(i) for i in SIDES]
    schedule = [conditional_landing(ledger, i, triggers[i], ends[i]) for i in SIDES]
    return dict(pending=before, landing_schedule=schedule,
        pending_observed=[core._pending(i) for i in SIDES],
        confirmed_observed=[core._pending(i, True) for i in SIDES],
        unresolved_response=[0, 0], response_is_committed=False)
