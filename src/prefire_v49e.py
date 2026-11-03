"""既定OFFの比較用v4.9e。v4.9d の速度 (G4) を直す版。判定の中身は v4.9d と同じ。

検収前・本番未採用。フラグは本番設定へ登録しない (採用時に `src/production_config.py` へ根拠つきで登録する)。

- 出力不変の高速化 (`work_limit=None`): B3のまとめ推論と結果の再利用、特徴量複製の軽量化、応手列挙の軽量化。
  v4.9d と決定的な項目が全行一致する (`scripts/verify_prefire_v49e_equal.py`)。
- 作業量の打切り (`work_limit` を指定): 応手列挙の2手目を、連鎖シミュレーションの作業量で決定的に打ち切る。
  出力が変わるので別版として採点する。
"""
from __future__ import annotations
from src import prefire_work_budget as work_budget

from pathlib import Path
from typing import Any
from src.prefire_b3_v49e import B3Batch, code_digest, model_digest
from src.prefire_landing_v49d import AGG_BASE
from src.prefire_landing_v49e import LandingV49E
from src.prefire_responses_v49d import RESPONSE_MAX_HANDS
from src.prefire_v49d import PostV49D, display as display_v49d, MS, LIMIT_MS
from src.prefire_worker import LOW_SPEC_RATIO

# 既定OFF。ONにしても本番経路には接続しない (比較・検収専用)。
PREFIRE_V49E_ENABLED_DEFAULT: bool = False
ROOT = Path(__file__).resolve().parents[1]


class PostV49E(PostV49D):
    """v4.9d の同期・証明・保持を継承し、連続値 (b) のB3評価と応手列挙を高速版に差し替える。"""

    def __init__(self, predictor: Any, table: Any, policy: dict, native: Any,
                 enabled: bool = PREFIRE_V49E_ENABLED_DEFAULT, pre_model: dict | None = None,
                 aggregator: str = AGG_BASE, all_variants: bool = True,
                 work_limit: float | None = None, fast_b3: bool = True, lean: bool = True,
                 salt: str | None = None, max_hands: int = RESPONSE_MAX_HANDS) -> None:
        super().__init__(predictor, table, policy, native, enabled, pre_model, aggregator, all_variants)
        self.landing = LandingV49E(work_limit, lean, max_hands)
        if salt is None:
            salt = (model_digest(predictor)+code_digest(ROOT)) if enabled else ''
        self.b3 = B3Batch(predictor, table, native, salt, light=fast_b3, batch=fast_b3, reuse=fast_b3)
        self.work_limit, self.audit_seconds = work_limit, 0.

    def _evaluator(self, clock: Any) -> Any:
        """B3の評価関数。`.many` は複数候補を1回で評価する。照合に使った時間は工程時間から差し引く。"""
        def many(requests: list) -> list[float]:
            with clock.time('b3'):
                values = self.b3.evaluate(self.state, requests)
            audit = self.b3.take_audit_seconds()
            clock.seconds['b3'] = max(0., clock.seconds['b3']-audit)
            self.audit_seconds += audit
            return values

        def one(boards: list, pending: list) -> float:
            return many([(boards, pending)])[0]

        one.many = many
        return one

    def calculate_record(self, record: dict, raw: list[dict]) -> dict | None:
        self.audit_seconds, self.landing.last_enum = 0., {}
        value = super().calculate_record(record, raw)
        if value is None:
            return None
        value['continuous'] = dict(value['continuous'], **{k: self.landing.last_enum.get(k)
                                   for k in ('sims', 'steps', 'work', 'truncated', 'b3_evals')})
        if self.audit_seconds:
            # 抜き取り照合に使った時間は、計測にも工程時間にも含めない。
            audit_ms = self.audit_seconds*MS
            value['measured_ms'] = max(0., value['measured_ms']-audit_ms)
            elapsed = value['measured_ms']*LOW_SPEC_RATIO+value['reused_physics_ms']
            value.update(low_spec_ms=elapsed, ready_sec=self.state.t+elapsed/MS, over_budget=elapsed > LIMIT_MS)
            stage_ms = value['stage']['ms']
            stage_ms['other'] = max(0., stage_ms['other']-audit_ms)
        work_budget.publication(value)
        return value

    def b3_summary(self) -> dict:
        """B3まとめ評価・再利用・抜き取り照合の件数 (受領票へ記録する)。"""
        return dict(self.b3.stats, cache_size=len(self.b3.cache), salt=self.b3.salt)


def display(b3: float, baseline: float, projection: dict | None, c: float, pre_delta: float,
            terminal: bool = False, menu: bool = False, enabled: bool = PREFIRE_V49E_ENABLED_DEFAULT) -> float:
    """表示は v4.9d と同一 (証明結果と連続値を受け取るだけ)。既定OFFは基準値を返す。"""
    return display_v49d(b3, baseline, projection, c, pre_delta, terminal, menu, enabled)
