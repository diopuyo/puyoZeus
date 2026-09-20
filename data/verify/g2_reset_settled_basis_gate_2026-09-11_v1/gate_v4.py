"""落下前の初期分布にも同じ文脈資格を課す。旧v3は比較用に保持する。"""
from __future__ import annotations
import gate_v3 as V3

V1 = V3.V1


class SettledBasisGate(V3.SettledBasisGate):
    def _advance(self, obs: V1.CallObservation) -> str | None:
        if self.state == 'AWAIT_FALL' and obs.board_present:
            if not obs.match_active or obs.effect_gate_window_active is not False or obs.origin_present:
                return self._break('entry_context_invalid')
        return super()._advance(obs)
