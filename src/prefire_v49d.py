"""既定OFFの比較用v4.9d。終盤の死亡判定を作り直した版 (死亡証明 (a) と連続値 (b) を分離)。

検収前・本番未採用。フラグは本番設定へ登録しない (採用時に `src/production_config.py` へ根拠つきで登録する)。
v4.9c で正しかった修正 (送り4個差の初手起点・端数の統一、物理完走の独立化、古い予測盤の完走終了) は
継承する。表示側での72個上書きや新しい閾値は持たない。
"""
from __future__ import annotations

from collections import Counter
from time import perf_counter
from typing import Any
import numpy as np
from src.board import COLOR_UNKNOWN
from src.exchange_native_chain import native_chains
from src.prefire_facts_v49d import FactsTracker
from src.prefire_landing_v49d import LandingV49D, StageClock, AGG_BASE, AGGREGATORS
from src.prefire_post_b3 import forecast_features
from src.prefire_v49c import PostV49C
from src.prefire_worker import LOW_SPEC_RATIO

# 既定OFF。ONにしても本番経路には接続しない (比較・検収専用)。
PREFIRE_V49D_ENABLED_DEFAULT: bool = False
MS = 1000.
LIMIT_MS = 500.
SIDES = (0, 1)
# 工程名 (時間の内訳)。同期・個数判定・応手列挙・楽観返し・着弾・B3・その他。
STAGES = ('sync', 'count', 'enumerate', 'optimistic', 'landing', 'b3')
ERROR_SAMPLES = 20


class PostV49D(PostV49C):
    """v4.9cの同期・物理完走を継承し、証明の入力契約と連続値を差し替える。"""

    def __init__(self, predictor: Any, table: Any, policy: dict, native: Any,
                 enabled: bool = PREFIRE_V49D_ENABLED_DEFAULT, pre_model: dict | None = None,
                 aggregator: str = AGG_BASE, all_variants: bool = True) -> None:
        super().__init__(predictor, table, policy, native, enabled, pre_model)
        if aggregator not in AGGREGATORS:
            raise ValueError(f'未登録の集約方法: {aggregator}')
        self.aggregator = aggregator
        # 実験では3案すべてを計算する。採用版の速度測定では選んだ1案だけを計算する。
        self.variants = AGGREGATORS if all_variants else (aggregator,)
        self.landing = LandingV49D()
        self.tracker = FactsTracker(native)
        # 例外は隠さず件数と先頭の内容を受領票へ残す。例外のあった更新は死亡を断定しない。
        self.errors: Counter = Counter()
        self.error_samples: list[str] = []
        self.tracker_totals: Counter = Counter()

    def _boundary(self, stamp: float, game: int) -> None:
        changed = self.state.game_idx != game
        super()._boundary(stamp, game)
        if changed:
            self.tracker_totals.update(self.tracker.counters)
            self.tracker.reset()
            self.landing.reset(game)

    def _current(self, stamp: float, game: int, raw: list[dict]) -> None:
        super()._current(stamp, game, raw)
        try:
            self.tracker.observe(self.state, raw, stamp)
        except Exception as exc:  # noqa: BLE001 - 件数を記録して継続する (受領票で0件を確認する)
            self._note('observe', stamp, exc)

    def tracker_summary(self) -> dict:
        """試合をまたいだ観測事実の件数 (試合境界で追跡器は作り直されるため、ここへ積み上げる)。"""
        return dict(self.tracker_totals+Counter(self.tracker.counters))

    def _note(self, where: str, stamp: float, exc: Exception) -> None:
        self.errors[where] += 1
        if len(self.error_samples) < ERROR_SAMPLES:
            self.error_samples.append(f'{where}@{stamp}: {exc!r}')

    def _evaluator(self, clock: StageClock) -> Any:
        def evaluate(boards: list, pending: list) -> float:
            with clock.time('b3'):
                features = forecast_features(self.state, boards, pending, self.table, self.native)
                return float(self.predictor.predict(features))
        return evaluate

    def _continuous(self, value: dict, clock: StageClock, errors: list[str]) -> dict:
        """連続値 (b)。未確定色を含む盤面は固定B3が評価できないため値なし (v4.9cと同じ)。
        例外のときは現在の盤面と予告のB3だけに戻し、例外を記録する。"""
        blank = dict(values={k: None for k in self.variants}, receiver=0, hands=0, immediate=False,
                     responses=0, strata=[], complete=False, nodes=0)
        if any(COLOR_UNKNOWN in board for board in value['boards']):
            return blank
        try:
            return self.landing.continuous(self, value, self._evaluator(clock), clock, self.variants)
        except Exception as exc:  # noqa: BLE001
            self._note('continuous', self.state.t, exc)
            errors.append('continuous')
            try:
                plain = self._evaluator(clock)(value['boards'], value['pending'])
            except Exception as again:  # noqa: BLE001
                self._note('continuous_plain', self.state.t, again)
                return blank
            return dict(blank, values={k: plain for k in self.variants})

    def calculate_record(self, record: dict, raw: list[dict]) -> dict | None:
        if not self.enabled:
            return None
        began, clock = perf_counter(), StageClock()
        with clock.time('sync'):
            self.synchronize(record, raw)
            value = self.projection()
            self.tracker.refresh_verdicts(self.chains, value['reconciliation'], self.state.sl.core)
        errors: list[str] = []
        with native_chains(self.native):
            try:
                self.landing.apply(self, self.tracker, value, clock)
            except Exception as exc:  # noqa: BLE001
                self._note('proof', self.state.t, exc)
                errors.append('proof')
                value.update(landing_proofs=[], death_sides=[], proof_nodes=0, optimistic_calls=0)
            values = self._continuous(value, clock, errors)
        dead = value['death_sides']
        v_post = values['values'][self.aggregator]
        active = bool(dead or (v_post is not None and (
            any(value['board_prediction_active']) or any(value['pending_observed']))))
        measured = (perf_counter()-began)*MS
        reused = max(0., record['low_spec_ms']-record.get('inference_ms', 0.)*LOW_SPEC_RATIO)
        elapsed = measured*LOW_SPEC_RATIO+reused
        stage = clock.snapshot()
        stage['ms']['other'] = max(0., measured-sum(stage['ms'].get(k, 0.) for k in STAGES))
        value.update(game=self.state.game_idx, input_sec=self.state.t, ready_sec=self.state.t+elapsed/MS,
            low_spec_ms=elapsed, measured_ms=measured, reused_physics_ms=reused,
            over_budget=elapsed > LIMIT_MS, fight=self.fight, active=active,
            v_post=v_post, v_post_variants=values['values'],
            continuous=dict(receiver=values['receiver'], hands=values['hands'],
                immediate=values['immediate'], responses=values['responses'],
                strata=values['strata'], complete=values['complete'], nodes=values['nodes']),
            errors=errors, stage=stage, current_states=[s.state for s in self.state.sides],
            queues=[list(s.queue) for s in self.state.sides])
        assert max(value['evidence_times']) <= value['input_sec']
        return value


def display(b3: float, baseline: float, projection: dict | None, c: float,
            pre_delta: float, terminal: bool = False, menu: bool = False,
            enabled: bool = PREFIRE_V49D_ENABLED_DEFAULT) -> float:
    """表示は証明結果と連続値を受け取るだけ。独自の個数閾値・72個の上書きを持たない。"""
    if not enabled:
        return baseline
    if menu:
        return b3
    if projection is not None and projection['active'] and len(projection['death_sides']) == 1:
        return float(projection['death_sides'][0] == 1)
    if terminal or projection is None or not projection['active'] or projection['v_post'] is None:
        return b3
    return float(np.clip(b3+c*(projection['v_post']-b3)+pre_delta, 0., 1.))
