"""既定OFFの比較用v4.9c。470試合の検収で誤死亡3試合があり、本番未採用。"""
from __future__ import annotations

from copy import deepcopy
from time import perf_counter
from typing import Any
import numpy as np
from src.phase_j._live_ex_frozen.src.prefire_physics_v49c import PhysicsV49C
from src.phase_j._live_ex_frozen.src.prefire_landing_v49c import LandingV49C
from src.phase_j._live_ex_frozen.src.prefire_post_b3 import forecast_features
from src.phase_j._live_ex_frozen.src.exchange_native_chain import native_chains
from src.phase_j._live_ex_frozen.src.prefire_worker import LOW_SPEC_RATIO
from src.phase_j._live_ex_frozen.src.production_config import PREFIRE_V49C_ENABLED_DEFAULT

MS = 1000.
LIMIT_MS = 500.
SIDES = (0, 1)


class PostV49C(PhysicsV49C):
    def __init__(self, predictor: Any, table: Any, policy: dict, native: Any,
                 enabled: bool = PREFIRE_V49C_ENABLED_DEFAULT, pre_model: dict | None = None) -> None:
        super().__init__(predictor, table, policy, native, enabled, pre_model)
        self.landing = LandingV49C()

    def synchronize(self, record: dict, raw: list[dict]) -> None:
        """到着済みの候補だけを読み、現在盤面で完走した側は元予測へ戻さない。"""
        self.fight = record.get('fight')
        for side in SIDES:
            trigger = record['chain_ids'][side]
            if trigger is None:
                self.chains[side] = None
                continue
            previous = self.chains[side]
            same = previous is not None and previous['started'] == trigger
            baseline = next((r for r in reversed(self.ledger_history) if r[0] < trigger), None)
            baseline = baseline or (self.state.game_start_t, 0., 0., 0., 0.)
            finished = not record['board_prediction_active'][side] or bool(same and previous['finished'])
            self.chains[side] = dict(signature=(trigger, 0, 0), started=trigger, external=True,
                finished=finished, external_sent=record['forecast_sent'][side] or 0,
                generated_at_fire=baseline[1+side], carry_at_fire=baseline[3+side],
                best=dict(scores=record['predicted_scores'][side]),
                result=dict(board=record['boards'][side], evidence_times=record['evidence_times']))
            self._finish_chain(side, self.state.t, raw[side])

    def calculate_record(self, record: dict, raw: list[dict]) -> dict | None:
        if not self.enabled:
            return None
        began = perf_counter()
        self.synchronize(record, raw)
        value = self.projection()
        before = deepcopy(value['boards'])
        with native_chains(self.native):
            self.landing.apply(self, value)
        dead = value['death_sides']
        if len(dead) == 1:
            probability = float(dead[0] == 1)
        elif value['physics_valid']:
            features = forecast_features(self.state, value['boards'], value['pending'], self.table, self.native)
            probability = self.predictor.predict(features)
        else:
            probability = None
        active = bool(probability is not None and (any(value['board_prediction_active']) or
            any(value['pending_observed']) or dead))
        measured = (perf_counter()-began)*MS
        reused = max(0., record['low_spec_ms']-record.get('inference_ms', 0.)*LOW_SPEC_RATIO)
        elapsed = measured*LOW_SPEC_RATIO+reused
        value.update(game=self.state.game_idx, input_sec=self.state.t,
            ready_sec=self.state.t+elapsed/MS, low_spec_ms=elapsed, measured_ms=measured,
            reused_physics_ms=reused, over_budget=elapsed>LIMIT_MS,
            fight=self.fight, v_post=probability, active=active, prelanding_boards=before,
            current_states=[s.state for s in self.state.sides],
            queues=[list(s.queue) for s in self.state.sides])
        assert max(value['evidence_times']) <= value['input_sec']
        return value


def display(b3: float, baseline: float, projection: dict | None, c: float,
            pre_delta: float, terminal: bool = False, menu: bool = False,
            enabled: bool = PREFIRE_V49C_ENABLED_DEFAULT) -> float:
    """表示は共通証明の結果を受け取るだけ。独自の個数閾値を持たない。"""
    if not enabled:
        return baseline
    if menu:
        return b3
    if projection is not None and projection['active'] and len(projection['death_sides']) == 1:
        return float(projection['death_sides'][0] == 1)
    if terminal:
        return b3
    if projection is None or not projection['active']:
        return b3
    return float(np.clip(b3+c*(projection['v_post']-b3)+pre_delta, 0., 1.))
