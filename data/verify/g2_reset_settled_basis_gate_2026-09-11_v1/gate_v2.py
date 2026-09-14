"""実resetと同frameのJから開始する版。旧対照のreset前frame起点を訂正。"""
from __future__ import annotations
import settled_basis_gate as V1


class SettledBasisGate(V1.SettledBasisGate):
    def observe(self,obs: V1.CallObservation) -> V1.BasisCandidate|None:
        candidate=super().observe(obs)
        if candidate is None and self.state not in ('ISSUED','BROKEN') and obs.frame==self.deadline:
            self._break('baseline_deadline')
            self.rows[-1]['gate_state']=self.state
            self.rows[-1]['reason']='baseline_deadline:'+str(self.rows[-1]['reason'])
        return candidate

    def _admit(self,obs: V1.CallObservation) -> str|None:
        V1.require(type(obs) is V1.CallObservation,'observation_type')
        if self.state in ('ISSUED','BROKEN'):
            return 'candidate_already_issued' if self.state=='ISSUED' else 'gate_broken'
        V1.require(type(obs.call_token) is str and bool(obs.call_token),'call_token_type')
        V1.require(type(obs.frame) is int and type(obs.epoch) is int and type(obs.generation) is int,'call_integer_type')
        V1.require(all(type(v) is bool for v in (obs.match_active,obs.origin_present,
            obs.landing_grace_expired,obs.board_present)),'call_boolean_type')
        V1.require(obs.effect_gate_window_active is None or type(obs.effect_gate_window_active) is bool,'effect_type')
        if obs.call_token in self.seen_tokens: return self._break('call_token_reused')
        self.seen_tokens.add(obs.call_token)
        if obs.observation_stage!=V1.ACTUAL_CALL_STAGE: return self._break('not_actual_J_call_stage')
        if obs.exception is not None: return self._break('exception_present')
        if obs.scope!=self.scope: return self._break('scope_changed')
        if obs.epoch!=self.scope[2] or obs.generation!=self.scope[5]: return self._break('epoch_or_generation_changed')
        expected=self.reset_frame if self.last_frame is None else self.last_frame+V1.STRIDE
        if obs.frame!=expected: return self._break('stride_broken')
        self.last_frame=obs.frame
        if obs.frame>self.deadline: return self._break('deadline_exceeded')
        return None
