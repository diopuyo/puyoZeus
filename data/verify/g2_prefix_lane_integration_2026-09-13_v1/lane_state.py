"""Modeごとの条件付き履歴。入力資格は別責務で、曖昧/prepopは反映しない。"""
from __future__ import annotations

from dataclasses import asdict
from typing import Any
import prefix_phase_math_v2 as P
import prefix_phase_saved_v4 as V
import prefix_commit as C


class Lane:
    def __init__(self, parts: Any, initial: Any, ledger: Any, assumption: str) -> None:
        parts.mode.L.check(ledger)
        parts.mode.B.validate(initial)
        require = parts.mode.B.require
        require(initial.scope == ledger.scope and initial.deadline == ledger.deadline
            and initial.frame == ledger.clock, 'lane_initial_scope_clock')
        require(bool(ledger.applied) and initial.tokens[-len(ledger.applied):] == ledger.applied,
            'lane_initial_applied')
        self.parts, self.initial, self.base = parts, initial, ledger.applied
        self.current = initial
        self.families = (P.Family(0, P.SETTLED, initial),)
        self.fired: frozenset[str] = frozenset()
        self.prior = parts.mode.C.T.H.uncalibrated_uniform(assumption)
        self.assumption = assumption
        self.observations: list[dict] = []
        self.committed: list[dict] = []

    def firing(self, ledger: Any, token: str) -> None:
        """呼び手が同scope原J/起点所有を検証してから通知する。ACKとは独立。"""
        require = self.parts.mode.B.require
        self.parts.mode.L.check(ledger)
        require(ledger.scope == self.initial.scope and ledger.deadline == self.initial.deadline,
            'lane_fire_scope')
        require(token in {a.token for a in ledger.arrivals[len(self.base):]}, 'lane_fire_unknown_token')
        self.fired |= frozenset((token,))

    def observe(self, ledger: Any, frame: int, observed: Any, evidence_key: str) -> dict:
        require = self.parts.mode.B.require
        self.parts.mode.L.check(ledger)
        require(ledger.scope == self.initial.scope and ledger.deadline == self.initial.deadline
            and ledger.applied[:len(self.base)] == self.base, 'lane_observe_scope_base')
        require(type(evidence_key) is str and bool(evidence_key)
            and all(row['evidence_key'] != evidence_key for row in self.observations), 'lane_evidence_key')
        require(ledger.clock <= frame <= ledger.deadline, 'lane_observe_ledger_clock')
        arrivals = ledger.arrivals[len(self.base):]
        families, lineage = V.advance(self.parts, self.families, arrivals, frame, observed, self.fired, self.prior)
        require(bool(families), 'lane_observation_zero_support')
        require(sum(len(f.value.worlds) for f in families) <= self.parts.mode.B.MAX_WORLDS,
            'lane_total_support_limit')
        row = dict(frame=frame, evidence_key=evidence_key, fired_tokens=sorted(self.fired),
            observed=self.parts.mode.B.grid(observed), arrivals=[asdict(a) for a in arrivals], lineage=lineage,
            families=[V.encode(self.parts.mode.BASE.V1.S, family, frozenset()) for family in families],
            source_qualification_connected=False, quality_gate_clear=False)
        self.families = families
        self.observations.append(row)
        return row

    def prepare(self, ledger: Any, call: str) -> C.Prepared | None:
        """単一履歴・settledかつ未反映prefixだけを準備。保留は別票に既に残る。"""
        if len(self.families) != 1 or self.families[0].phase != P.SETTLED:
            return None
        family = self.families[0]
        if len(self.base) + family.prefix <= len(ledger.applied):
            return None
        self.parts.mode.B.require(bool(self.observations), 'lane_observation_missing')
        return C.prepare(self.parts, self.current, ledger, self.base, self.families,
            family.value.frame, call, self.observations[-1]['evidence_key'])

    def accepted(self, prepared: C.Prepared, following: Any) -> None:
        """原Registryに一回反映できた後だけ呼ぶ。ここで元台帳は変更しない。"""
        require = self.parts.mode.B.require
        require(prepared.expected is self.current and prepared.following is following,
            'lane_commit_state_identity')
        require(prepared.receipt['evidence_key'] == self.observations[-1]['evidence_key']
            and not any(row['source_call_token'] == prepared.receipt['source_call_token']
                for row in self.committed), 'lane_commit_evidence_identity')
        self.current = following
        self.committed.append(prepared.receipt)
