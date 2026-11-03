"""固定B3評価器の1行推論。学習時のfloat32丸めと列順を維持する。"""
from __future__ import annotations

from operator import itemgetter
from typing import TYPE_CHECKING, Mapping

import numpy as np
from threadpoolctl import ThreadpoolController

if TYPE_CHECKING:
    from src.phase_j._live_ex_frozen.src.b3_train_model import TwoStateModel


class RowPredictor:
    """固定モデル読込後に列取得とスレッドプール探索を一度だけ準備する。"""

    def __init__(self, model: TwoStateModel) -> None:
        self.evaluators = (model.static, model.exchange)
        self.getters = tuple(itemgetter(*ev.columns) if ev is not None else None
                             for ev in self.evaluators)
        # sklearnとモデルのロード後に生成し、dl_iterate_phdrの毎行走査を避ける。
        self.pools = ThreadpoolController()

    def predict(self, row: Mapping[str, object]) -> float:
        from src.phase_j._live_ex_frozen.src.b3_train_model import MODE_COLUMN, THREADS, hgb_threads

        mode = float(row[MODE_COLUMN])
        if np.isnan(mode):
            raise ValueError('mode_exchange に欠測 (状態を選べない)')
        if mode not in (0.0, 1.0):
            return float('nan')
        ev, getter = self.evaluators[int(mode)], self.getters[int(mode)]
        if ev is None:
            raise ValueError(f'mode={int(mode)} の行があるが評価器が無い')
        # pandas経路と同じfloat32化後、HGB自身がfloat64へ変換する。
        values = np.asarray(getter(row), dtype=np.float32).reshape(1, len(ev.columns))
        with self.pools.limit(limits=THREADS, user_api='blas'), \
                self.pools.limit(limits=hgb_threads(), user_api='openmp'):
            return float(ev.model.predict_proba(values)[0, 1])
