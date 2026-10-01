"""選択済み手の仮想着弾G_fe・S3・E35を既存本番部品で評価する。"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np

from src import prefire_best_play as fire
from src import prefire_best_play_layer as layer
from src import prefire_v5_search as base
from src.prefire_v5c_value import EvaluationCache


class ProductionValue:
    """静止末端はG_fe、発火末端はS3とのlogit合成。入力局面の文脈を固定する。"""

    def __init__(self, overlay: Any, states: tuple, elapsed: float,
                 cache: EvaluationCache | None = None) -> None:
        self.gfe = cache if cache is not None else EvaluationCache(overlay)
        self.gfe.bind(overlay)
        self.states, self.elapsed = states, elapsed
        self.latest = tuple(SimpleNamespace(board=base.sim._board(p.board),
            queue=np.asarray((*p.queue[:4], *(0 for _ in range(max(0, 4-len(p.queue))))))) for p in states)
        self.overlay = overlay
        self.context: layer.Context | None = None
        self.s3_calls = self.e35_calls = 0
        self.certainties: dict[base.Exchange, tuple[bool, bool]] = {}

    def __call__(self, exchange: base.Exchange, elapsed: float) -> float:
        """本番モデルは変えない。残し盤面と残組は5C遷移の結果を使う。"""
        static = self.gfe(exchange, elapsed)
        if not any(p.chains for p in exchange.sides):
            return static
        if self.context is None:
            self.context = layer.build_context(self.overlay, self.latest,
                tuple(p.queue for p in self.states), self.elapsed, {})
        self.s3_calls += 1
        scores = np.asarray([p.score for p in exchange.sides], dtype=float)
        result = layer.logit_mean(static, layer.s3_value(self.context, scores, elapsed))
        proven = [self.proven(side, exchange.sides[side], elapsed) for side in (0, 1)]
        self.certainties[exchange] = tuple(proven)
        if sum(proven) == 1:
            return layer._lethal_value(proven.index(True))
        return result

    def proven(self, side: int, attack: base.Position, elapsed: float) -> bool:
        """受け手に全色・演出中の猶予を認めた既存E35の保守的証明を使う。"""
        if not attack.chains:
            return False
        self.e35_calls += 1
        line = fire.FireLine(attack.consumed, attack.score, attack.chains, attack.board)
        sent = max(0, base.sim.send_ojama(attack.score, elapsed)-self.states[side].pending)
        return fire.lethal(self.states[1-side].board, int(sent), layer.counter_hands(line),
                           self.context.colors, elapsed)
