"""落下開始から着地までの文脈を検査し、不適格な落下の借用を拒否する。"""
from __future__ import annotations
import gate_v2 as V2

V1 = V2.V1


class SettledBasisGate(V2.SettledBasisGate):
    def observe(self, obs: V1.CallObservation) -> V1.BasisCandidate | None:
        candidate = super().observe(obs)
        self.rows[-1].update(match_active=obs.match_active,
            effect_window=obs.effect_gate_window_active, origin_present=obs.origin_present,
            landing_grace_expired=obs.landing_grace_expired)
        return candidate

    def _advance(self, obs: V1.CallObservation) -> str | None:
        # リセット直後のMENUは許すが、落下開始以後の欠測・文脈切替は許さない。
        tracking = self.state == 'AWAIT_SETTLE' or obs.state == V1.STATE_TSUMO_FALL
        if tracking:
            if not obs.match_active or obs.effect_gate_window_active is not False or obs.origin_present:
                return self._break('fall_context_invalid')
            if not obs.board_present:
                return self._break('fall_board_missing')
        # 猶予は落下中に存在してよい。STABLE発行時の期限検査は既存実装を維持。
        return super()._advance(obs)
