"""死亡証明の入力契約 (v4.9d)。live と post が同じ入口を使うための純関数群。

証明に入れてよい入力は、次をすべて満たす場合だけ。1つでも欠ければ「対象外」とし、
理由を返す。探索が成立したことを、入力の確実性と取り違えない (v4.9c の誤死亡3試合の教訓)。

状態は持たない。台帳・盤面・得点から事実 (ProofFacts) を作るのは呼び出し側
(`src/prefire_facts_v49d.py`)、ここは判定と件数記録だけを担当する。
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

# 対象外の理由コード。判定順 = この並び (最初に偽になったものが「主理由」)。
UNKNOWN_BOARD = 'unknown_board'
CONFIRMED_SHORT = 'confirmed_short'
RECEIVER_CHAIN_OPEN = 'receiver_chain_open'
OPPONENT_UNFINISHED = 'opponent_unfinished'
RECEIVER_NOT_CURRENT = 'receiver_not_current'
BOARD_NOT_RESTING = 'board_not_resting'
BOARD_GUARD = 'board_guard'
ATTACK_UNVERIFIED = 'attack_unverified'
PHYSICS_MISMATCH = 'physics_mismatch'
ORIGIN_UNKNOWN = 'origin_unknown'
REASON_ORDER = (UNKNOWN_BOARD, CONFIRMED_SHORT, RECEIVER_CHAIN_OPEN, OPPONENT_UNFINISHED,
                RECEIVER_NOT_CURRENT, BOARD_NOT_RESTING, BOARD_GUARD, ATTACK_UNVERIFIED,
                PHYSICS_MISMATCH, ORIGIN_UNKNOWN)

# 攻撃の帰属を検証できない細目 (ATTACK_UNVERIFIED の detail)。
ATTACK_RECONCILE_MISMATCH = 'attack_reconcile_mismatch'
ATTACK_CANDIDATE_MISSING = 'attack_candidate_missing'
ATTACK_ALL_CLEAR_AMBIGUOUS = 'attack_all_clear_ambiguous'
ATTACK_CHAIN_OPEN = 'attack_chain_open'


@dataclass(frozen=True)
class ProofFacts:
    """証明対象 (受け側) 1つぶんの観測事実。値の意味は判定順の理由コードと対応する。"""
    side: int
    pending: int
    confirmed: int
    unknown_board: bool
    receiver_chain_open: bool
    opponent_unfinished: bool
    receiver_current: bool
    board_resting: bool
    guard_reason: str | None
    attack_reason: str | None
    physics_mismatch: bool
    origin_known: bool


def blockers(facts: ProofFacts) -> list[str]:
    """偽になった条件を判定順にすべて返す。空なら入力は証明に使える。"""
    failed = [
        (UNKNOWN_BOARD, facts.unknown_board),
        (CONFIRMED_SHORT, facts.confirmed < facts.pending),
        (RECEIVER_CHAIN_OPEN, facts.receiver_chain_open),
        (OPPONENT_UNFINISHED, facts.opponent_unfinished),
        (RECEIVER_NOT_CURRENT, not facts.receiver_current),
        (BOARD_NOT_RESTING, not facts.board_resting),
        (BOARD_GUARD, facts.guard_reason is not None),
        (ATTACK_UNVERIFIED, facts.attack_reason is not None),
        (PHYSICS_MISMATCH, facts.physics_mismatch),
        (ORIGIN_UNKNOWN, not facts.origin_known),
    ]
    return [name for name, bad in failed if bad]


def admit(facts: ProofFacts) -> str | None:
    """主理由 (判定順で最初に偽の条件)。None なら適格。予告なしは判定対象外として None を返す。"""
    if facts.pending <= 0:
        return None
    found = blockers(facts)
    return found[0] if found else None


def detail(facts: ProofFacts) -> dict[str, str | None]:
    """理由コードの細目 (受領票用)。"""
    return dict(guard=facts.guard_reason, attack=facts.attack_reason)


class ContractTally:
    """拒否理由ごとの件数。主理由 (最初に偽) と全該当 (重複あり) を別に数える。"""

    def __init__(self) -> None:
        self.evaluated = 0
        self.admitted = 0
        self.first: Counter = Counter()
        self.any: Counter = Counter()
        self.guard: Counter = Counter()
        self.attack: Counter = Counter()

    def add(self, facts: ProofFacts) -> str | None:
        """1回の判定を記録し、主理由を返す。"""
        if facts.pending <= 0:
            return None
        self.evaluated += 1
        found = blockers(facts)
        if not found:
            self.admitted += 1
            return None
        self.first[found[0]] += 1
        self.any.update(found)
        if facts.guard_reason is not None:
            self.guard[facts.guard_reason] += 1
        if facts.attack_reason is not None:
            self.attack[facts.attack_reason] += 1
        return found[0]

    def merge(self, other: 'ContractTally') -> None:
        self.evaluated += other.evaluated
        self.admitted += other.admitted
        for mine, theirs in ((self.first, other.first), (self.any, other.any),
                             (self.guard, other.guard), (self.attack, other.attack)):
            mine.update(theirs)

    def as_dict(self) -> dict:
        return dict(evaluated=self.evaluated, admitted=self.admitted, rejected=self.evaluated-self.admitted,
                    first=dict(self.first), any=dict(self.any), guard=dict(self.guard),
                    attack=dict(self.attack))
