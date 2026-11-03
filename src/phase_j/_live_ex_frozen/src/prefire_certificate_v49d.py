"""死亡証明の保持と解除 (v4.9d)。保持は根拠と結び付け、反証が来たら解除する。

根拠 = 予告の量・受け側と送り側の連鎖世代・着弾の累計・受け側の置き終わり回数。
生状態 (STABLE/CHAIN など) の一時的なちらつきだけでは解除しない。
試合全体での無条件の保持はしない。純関数だけで構成し、状態は呼び出し側が持つ。
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

KEEP, REPROVE, DROP = 'keep', 'reprove', 'drop'
RECEIVER_CHAIN_STARTED = 'receiver_chain_started'
OPPONENT_NEW_CHAIN = 'opponent_new_chain'
PENDING_VANISHED = 'pending_vanished'
PENDING_DECREASED = 'pending_decreased'
LANDING_MISMATCH = 'landing_mismatch'
PLACEMENTS_EXCEEDED = 'placements_exceeded'
GRACE_EXPIRED = 'grace_expired'


@dataclass(frozen=True)
class Basis:
    """証明の根拠。発行時と現在で同じ形を比べる。"""
    pending: int
    receiver_seq: int
    sender_seq: int
    dropped: int
    placements: int
    hands: int
    min_drop: int
    stamp: float

    def as_dict(self) -> dict:
        return asdict(self)


def judge(base: Basis, now: Basis, admitted: bool, grace_sec: float) -> tuple[str, str | None]:
    """発行時の根拠 `base` と現在の根拠 `now` を比べ、(判定, 理由) を返す。

    DROP = 解除、REPROVE = 着弾で根拠が正当に変わったので契約を通る再証明が必要、KEEP = 保持。
    予告の増加は死亡を強めるだけなので反証にしない。
    """
    if now.receiver_seq != base.receiver_seq:
        return DROP, RECEIVER_CHAIN_STARTED
    if now.sender_seq != base.sender_seq:
        return DROP, OPPONENT_NEW_CHAIN
    if now.pending <= 0:
        return DROP, PENDING_VANISHED
    landed = now.dropped-base.dropped
    if landed > 0:
        if landed < base.min_drop:
            return DROP, LANDING_MISMATCH
        if base.pending-now.pending > landed:
            return DROP, PENDING_DECREASED
        return _reprove(base, now, admitted, grace_sec)
    if now.pending < base.pending:
        return DROP, PENDING_DECREASED
    if now.placements-base.placements > base.hands:
        return DROP, PLACEMENTS_EXCEEDED
    return KEEP, None


def _reprove(base: Basis, now: Basis, admitted: bool, grace_sec: float) -> tuple[str, str | None]:
    """着弾の後は契約を通る再証明を求める。通らない間は猶予の範囲だけ保持する。"""
    if admitted:
        return REPROVE, None
    return (KEEP, None) if now.stamp-base.stamp <= grace_sec else (DROP, GRACE_EXPIRED)
